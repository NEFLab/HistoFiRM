"""Actual pinned HE2RNA architecture with explicit signed-input masks, cancer-specific inner selection and outer refit.
Paths and process-level isolation must be bound by the server adapter.
The trainer never accepts external/test labels. Checkpoints are locally produced,
hash-bound project artifacts; do not load downloaded/untrusted pickle files here.
"""
from __future__ import annotations
import argparse,fcntl,os,random,tempfile,time
from pathlib import Path
import numpy as np
import torch
from common import *
from support import TargetScale,masked_patient_mse
from he2rna_adapter import HE2RNAAdapter
from spatial_bags import SpatialBags
CORE_SHA="1f3fdb5e4ac401b76dde1212b7d73c70e51c13ec9549a73e9e12319e82705fae"

PROTOCOL='WSI_B3_ISOLATED_FINALMERGE_20260914_V1'

class BudgetReached(RuntimeError):pass

def setup(seed,threads=4):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(threads)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def save_torch(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent);os.close(fd)
    try:torch.save(obj,tmp);os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def set_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
    if r['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(r['cuda'])

def validation_mse(model,bags,indices,yz):
    model.eval();total=0.;n=0
    with torch.no_grad():
        for i in indices:
            if not np.isfinite(yz[i]).any():continue
            p=bags.forward(model,[i])
            t=torch.as_tensor(yz[i:i+1],dtype=torch.float32,device=bags.device)
            total+=float(masked_patient_mse(p,t));n+=1
    require(n>0,'No eligible inner-validation observations');return total/n

def phase(root,signature,model_cfg,optim_cfg,bags,y,fit_idx,val_idx,seed,epochs,deadline,selection):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    ready=root/'PHASE_COMPLETE.json'
    if ready.exists():
        r=read_json(ready);require(r['signature']==signature,'Phase identity drift');verified(r['checkpoint']);return r
    s=TargetScale.fit(y[fit_idx]);require(s.variable.any(),'No varying supervised targets in this phase')
    z=s.transform(y);setup(seed,threads=4)
    finite=np.isfinite(z[fit_idx]);bias=np.where(finite,z[fit_idx],0.).sum(0)/np.maximum(finite.sum(0),1)
    model=HE2RNAAdapter(output_dim=y.shape[1],bias_init=bias,**model_cfg).to(bags.device)
    opt=torch.optim.Adam(model.parameters(),lr=optim_cfg['lr'],betas=(0.9,0.999),eps=1e-8,weight_decay=0.,amsgrad=False)
    history=[];start_epoch=0;best=float('inf');best_epoch=0;stale=0;accumulated_wall=0.
    last=root/'LATEST.json'
    if last.exists():
        l=read_json(last);q=torch.load(verified(l['checkpoint']),map_location='cpu',weights_only=False)
        require(q['signature']==signature,'Checkpoint signature differs')
        model.load_state_dict(q['state_dict']);opt.load_state_dict(q['optimizer'])
        for st in opt.state.values():
            for k,v in st.items():
                if torch.is_tensor(v):st[k]=v.to(bags.device)
        history=q['history'];start_epoch=q['epoch'];best=q['best'];best_epoch=q['best_epoch'];stale=q['stale'];set_rng(q['rng'])
        accumulated_wall=q['phase_wall_seconds']
    started=time.monotonic();batch=optim_cfg['effective_patient_batch'];last_epoch_clock=started
    valid_fit=[i for i in fit_idx if np.isfinite(z[i]).any()]
    require(len(valid_fit)>0,'No training patients with supported labels')
    for epoch in range(start_epoch,epochs):
        if selection and stale>=optim_cfg['patience']:break
        if time.monotonic()>=deadline:raise BudgetReached('Resume saved phase at safe epoch boundary')
        model.train();order=np.random.permutation(valid_fit);loss_sum=0.;n=0;step_count=0
        for start in range(0,len(order),batch):
            group=order[start:start+batch];opt.zero_grad(set_to_none=True)
            # Patient rows remain equally weighted, irrespective of slide/patch counts.
            p=bags.forward(model,group)
            target=torch.as_tensor(z[group],dtype=torch.float32,device=bags.device)
            loss=masked_patient_mse(p,target);require(bool(torch.isfinite(loss)),'Nonfinite training loss')
            loss.backward();opt.step();step_count+=1
            loss_sum+=float(loss.detach())*len(group);n+=len(group)
        vm=validation_mse(model,bags,val_idx,z) if selection else None
        if selection:
            if vm<best:best=vm;best_epoch=epoch+1;stale=0
            else:stale+=1
        else:best_epoch=epoch+1
        history.append({'epoch':epoch+1,'train_scaled_mse':loss_sum/n,'inner_val_scaled_mse':vm,
                        'patients':n,'optimizer_steps':step_count,'epoch_seconds':time.monotonic()-last_epoch_clock,
                        'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else 0,
                        'GPU_peak_reserved_MiB':torch.cuda.max_memory_reserved()/1024**2 if torch.cuda.is_available() else 0})
        last_epoch_clock=time.monotonic()
        ck=root/('resume.'+str((epoch+1)%2)+'.pt')
        save_torch(ck,{'signature':signature,'state_dict':model.state_dict(),'optimizer':opt.state_dict(),
            'epoch':epoch+1,'history':history,'best':best,'best_epoch':best_epoch,'stale':stale,
            'rng':rng_state(),'phase_wall_seconds':accumulated_wall+time.monotonic()-started})
        write_json(last,{'checkpoint':record(ck),'signature':signature,'epoch':epoch+1},replace=True)
        # Point-by-point progress is durable before expensive downstream scoring/evaluation.
        write_json(root/'HISTORY.json',history,replace=True)
    require(best_epoch>0,'No completed epochs')
    # Selection checkpoints are not deployed; outer refit runs from scratch for best_epoch.
    final=root/'phase_final.pt';require(not final.exists(),'Unregistered phase checkpoint; reconcile')
    save_torch(final,{'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                     'scale':s.as_dict(),'architecture':model_cfg,'signature':signature})
    r={'status':'COMPLETE','signature':signature,'checkpoint':record(final),'best_epoch':best_epoch,
       'history':history,'scale_count':s.count,'supported':s.supported,'variable':s.variable,
       'phase_wall_seconds':accumulated_wall+time.monotonic()-started}
    write_json(ready,r);del model,opt
    if bags.device.startswith('cuda'):torch.cuda.empty_cache()
    return r

def train(job_path,protocol_path,out,device='cuda:0',max_seconds=43200):
    cfg=read_json(protocol_path);j=read_json(job_path);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    require(cfg['protocol_id']==PROTOCOL and cfg['B3_authorized'] is True,'Wrong/not released protocol')
    require(j['role']=='TCGA_OUTER_TRAIN_ONLY' and j['cohort']=='TCGA','External/test labels are not permitted')
    require(j['route']=='GENE' and j['fold'] in range(5),'Invalid unit')
    require(j['cancer'] in cfg['tcga_cancers'],'Unknown cancer')
    patients=j['patients'];ids=[p['patient_id'] for p in patients];unique(ids)
    require(all(p['cancer']==j['cancer'] for p in patients),'Cancer-specific model received other cancer')
    disjoint(ids,j['outer_test_patient_ids']);require(len(j['outer_test_patient_ids'])>0,'Missing heldout lineage')
    expected=set(j['inner_fit_patient_ids'])|set(j['inner_validation_patient_ids'])
    disjoint(j['inner_fit_patient_ids'],j['inner_validation_patient_ids']);require(expected==set(ids),'Inner partition does not equal outer train')
    label_path=verified(j['labels'])
    with np.load(label_path,allow_pickle=False) as z:
        require(set(z.files)=={'patient_id','target_id','values'},'Training archive has unexpected fields')
        same(z['patient_id'],ids);y=z['values'].astype(np.float64);targets=z['target_id'].astype(str)
    expected_width=18977
    require(y.shape==(len(ids),expected_width),'Production target shape differs');unique(targets)
    model_cfg=cfg['he2rna_architecture'];opts=cfg['he2rna_training'];require(model_cfg['input_dim']==1536,'Wrong input')
    require(device.startswith('cuda') and torch.cuda.is_available(),'Production training requires explicitly leased CUDA device')
    signature=digest({'job':j,'protocol':cfg,'core':CORE_SHA,'driver_sources':{n:sha(Path(__file__).parent/n) for n in ['train_he2rna.py','he2rna_adapter.py','spatial_bags.py','common.py','support.py']}})
    done=out/'TRAIN_COMPLETE.json'
    lockf=open(out/'unit.lock','a');fcntl.flock(lockf,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if done.exists():
            r=read_json(done);require(r['signature']==signature,'Completed unit belongs to other config');verified(r['checkpoint']);return r
        look={p:i for i,p in enumerate(ids)};fit=np.array([look[p] for p in j['inner_fit_patient_ids']]);val=np.array([look[p] for p in j['inner_validation_patient_ids']]);outer=np.arange(len(ids))
        bags=SpatialBags(patients,device);deadline=time.monotonic()+min(float(max_seconds),cfg['budget']['single_unit_wall_limit_hours']*3600)
        seed=2025+1009*j['fold']+int(__import__('hashlib').sha256(j['cancer'].encode()).hexdigest()[:6],16)
        sel=phase(out/'select',signature,model_cfg,opts,bags,y,fit,val,seed,opts['max_epochs'],deadline,True)
        refit=phase(out/'refit',signature,model_cfg,opts,bags,y,outer,[],seed,sel['best_epoch'],deadline,False)
        q=torch.load(verified(refit['checkpoint']),map_location='cpu',weights_only=False)
        q.update(target_id=targets,route=j['route'],cancer=j['cancer'],fold=j['fold'],training_patient_ids=ids,
                 source_core_sha256=CORE_SHA,protocol_id=PROTOCOL,unit_signature=signature)
        ck=out/'model.pt';require(not ck.exists(),'Unregistered final model; reconcile');save_torch(ck,q)
        r={'status':'FIT_AND_OUTER_REFIT_COMPLETE','signature':signature,'protocol_id':PROTOCOL,
           'method':'HE2RNA_UNI2H_GENE_FT01','cancer':j['cancer'],'fold':j['fold'],'route':j['route'],
           'checkpoint':record(ck),'fit_patient_ids':ids,'outer_test_patient_ids':j['outer_test_patient_ids'],
           'inner_validation_patient_ids':j['inner_validation_patient_ids'],'target_id':targets,
           'supported':refit['supported'],'train_label_counts':refit['scale_count'],
           'selected_epochs':sel['best_epoch'],'selection_history':sel['history'],'refit_history':refit['history'],
           'no_pretrained_TCGA_task_weights':True,'no_external_truth_used':True,
           'wall_seconds':sel['phase_wall_seconds']+refit['phase_wall_seconds']}
        write_json(done,r);return r
    except BudgetReached as e:
        write_json(out/'BUDGET_HOLD.json',{'status':'BUDGET_HOLD_RESUMABLE','reason':str(e)},replace=True);raise
    finally:fcntl.flock(lockf,fcntl.LOCK_UN);lockf.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--protocol',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--max-seconds',type=float,default=43200)
    a=p.parse_args();train(a.job,a.protocol,a.out,a.device,a.max_seconds)
