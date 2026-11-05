"""Author-core DeepPT, patient-grouped nested development and outer refit.
Production OS truth isolation and shared GPU lease are inherited from M1B.
Training receives ONLY outer-train gene labels, never CPTAC or heldout labels.
"""
from pathlib import Path
import argparse,os,time,tempfile,random,fcntl,hashlib
import numpy as np
import torch
from contracts import *
from deeppt_core import build_ae,build_reg,pooled_reg
from target_scale import Scale,patient_masked_mse
PROTOCOL='WSI_M1_B2_SCOPE_20260914_V1'
class BudgetStop(RuntimeError):pass

def setup(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(8)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def setrng(x):
 random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch'])
 if x['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(x['cuda'])
def torchsave(p,q):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=p.parent);os.close(fd)
 try:torch.save(q,t);os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def torchload(r):return torch.load(verified(r),map_location='cpu',weights_only=False)

class FeatureBags:
 def __init__(self,patients,dim=1536,chunk=512,max_cache_bytes=12*1024**3):
  self.patients=patients;self.dim=dim;self.chunk=chunk;self.cache={};self.bytes=0;self.limit=max_cache_bytes
  for p in patients:
   require(p.get('slides'),'Missing real patch inputs, tokens are not DeepPT raw features')
   for s in p['slides']:verified(s['feature'])
 def get(self,i):
  import h5py
  if i not in self.cache:
   aa=[]
   for s in self.patients[int(i)]['slides']:
    with h5py.File(s['feature']['path'],'r') as f:
     require(s['feature_key'] in f,'Unknown feature key');a=np.array(f[s['feature_key']],dtype=np.float32)
    require(a.ndim==2 and a.shape[1]==self.dim and len(a)>0 and np.isfinite(a).all(),'Bad actual patch data');aa.append(a)
   x=np.concatenate(aa);require(self.bytes+x.nbytes<=self.limit,'CPU patch cache budget exceeded; bind larger approved cache, not fewer patches');self.cache[i]=x;self.bytes+=x.nbytes
  return self.cache[i]
 def chunks(self,i,device):
  a=self.get(i)
  for j in range(0,len(a),self.chunk):yield torch.as_tensor(a[j:j+self.chunk],device=device)
 def ntiles(self,i):return len(self.get(i))

class EncodedBags:
 def __init__(self,ae,bags,indices,device):
  self.cache={};self.chunk=bags.chunk;ae.eval()
  with torch.no_grad():
   for i in indices:self.cache[int(i)]=torch.cat([ae.encoder(x).cpu() for x in bags.chunks(int(i),device)])
 def chunks(self,i,device):
  a=self.cache[int(i)]
  for j in range(0,len(a),self.chunk):yield a[j:j+self.chunk].to(device)


def validation(model,kind,bags,indices,device,y=None):
 model.eval();vals=[]
 with torch.no_grad():
  for i in indices:
   if kind=='ae':
    total=0.;n=0
    for x in bags.chunks(int(i),device):total+=float((model(x)-x).square().sum());n+=x.numel()
    vals.append(total/n)
   else:
    t=torch.as_tensor(y[int(i)],dtype=torch.float32,device=device);p=pooled_reg(model,bags.chunks(int(i),device));vals.append(float(patient_masked_mse(p[None],t[None])))
 require(vals and np.isfinite(vals).all(),'Non-estimable validation');return float(np.mean(vals))


def phase(kind,out,signature,bags,fit,val,cfg,seed,device,deadline,epochs,select=True,y=None,outputs=18977,dim=1536):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);done=out/'COMPLETE.json'
 if done.exists():
  r=load(done);require(r['signature']==signature,'Completed phase identity drift');verified(r['checkpoint']);return r
 setup(seed);model=(build_ae(dim,cfg['hidden']) if kind=='ae' else build_reg(outputs,cfg['hidden'],cfg['dropout'])).to(device)
 opt=torch.optim.Adam(model.parameters(),lr=cfg['lr']);start=0;best=float('inf');best_epoch=0;stale=0;history=[]
 resume=out/'RESUME.json'
 if resume.exists():
  rr=load(resume);q=torchload(rr['checkpoint']);require(q['signature']==signature,'Resume identity drift');model.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);start=q['epoch'];best=q['best'];best_epoch=q['best_epoch'];stale=q['stale'];history=q['history'];setrng(q['rng'])
  for st in opt.state.values():
   for k,v in st.items():
    if torch.is_tensor(v):st[k]=v.to(device)
 for ep in range(start,epochs):
  if select and stale>=cfg['patience']:break
  if time.monotonic()>=deadline:raise BudgetStop('Safe epoch budget boundary')
  model.train();order=np.random.permutation(fit);loss_sum=0.;npat=0;steps=0
  for b in range(0,len(order),cfg['batch_patients']):
   batch=order[b:b+cfg['batch_patients']];opt.zero_grad(set_to_none=True)
   for i in batch:
    if kind=='ae':
     denom=bags.ntiles(int(i))*dim;ploss=0.
     for x in bags.chunks(int(i),device):
      loss=(model(x)-x).square().sum()/denom;require(bool(torch.isfinite(loss)),'Nonfinite AE loss');(loss/len(batch)).backward();ploss+=float(loss.detach())
    else:
     p=pooled_reg(model,bags.chunks(int(i),device));t=torch.as_tensor(y[int(i)],dtype=torch.float32,device=device);loss=patient_masked_mse(p[None],t[None]);require(bool(torch.isfinite(loss)),'Nonfinite gene loss');(loss/len(batch)).backward();ploss=float(loss.detach())
    loss_sum+=ploss;npat+=1
   opt.step();steps+=1
  vm=validation(model,kind,bags,val,device,y) if select else None
  if select:
   if vm<best:
    best=vm;best_epoch=ep+1;stale=0;torchsave(out/'best.pt',{'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'signature':signature})
   else:stale+=1
  else:best_epoch=ep+1
  history.append({'epoch':ep+1,'train_mse':loss_sum/npat,'validation_mse':vm,'n_patients':npat,'optimizer_steps':steps})
  torchsave(out/'resume.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':ep+1,'best':best,'best_epoch':best_epoch,'stale':stale,'history':history,'rng':rng(),'signature':signature})
  write(resume,{'checkpoint':record(out/'resume.pt')},replace=True);write(out/'HISTORY.json',history,replace=True)
 require(best_epoch>0,'No complete epoch')
 if select:model.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=False)['model'])
 torchsave(out/'selected_or_refit.pt',{'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'signature':signature})
 r={'status':'COMPLETE','signature':signature,'kind':kind,'best_epoch':best_epoch,'epochs_executed':len(history),'selection':select,'checkpoint':record(out/'selected_or_refit.pt'),'history':history};write(done,r)
 del opt,model
 if device.startswith('cuda'):torch.cuda.empty_cache()
 return r


def train(job_path,protocol_path,out,device='cuda:0',max_seconds=43200):
 cfg=load(protocol_path);j=load(job_path);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 require(cfg['protocol_id']==PROTOCOL and cfg['release']['B2_DEEPPT']=='CONDITIONAL_RUN','Wrong release')
 require(j['cohort']=='TCGA' and j['role']=='OUTER_TRAIN_GENE_ONLY','Trainer must not receive external labels');require(device.startswith('cuda') and torch.cuda.is_available(),'Production training requires leased GPU')
 ps=j['patients'];ids=[p['patient_id'] for p in ps];unique(ids);require(all(p['cancer']==j['cancer'] for p in ps),'Cancer-specific scope drift');disjoint(ids,j['outer_test_ids']);disjoint(j['inner_fit_ids'],j['inner_validation_ids']);require(set(ids)==set(j['inner_fit_ids'])|set(j['inner_validation_ids']),'Invalid inner partition')
 with np.load(verified(j['labels']),allow_pickle=False) as z:
  require(set(z.files)=={'patient_id','target_id','values'},'Unexpected label fields');require(list(z['patient_id'].astype(str))==ids,'Patient axis mismatch');y=z['values'].astype(float);gene=z['target_id'].astype(str)
 require(y.shape==(len(ids),18977) and not np.isinf(y).any(),'Gene universe or nonfinite data differs');unique(gene);unique(j['outer_test_ids']);unique(j['inner_fit_ids']);unique(j['inner_validation_ids']);require(j['cancer'] in cfg['cancers'] and j['fold'] in cfg['folds'],'Unregistered cancer/fold')
 signature=digest({'job':j,'protocol':cfg,'code':{p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}})
 final=out/'TRAIN_COMPLETE.json';lock=open(out/'unit.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  if final.exists():r=load(final);require(r['signature']==signature,'Completed model mismatch');verified(r['checkpoint']);return r
  look={p:i for i,p in enumerate(ids)};fit=np.array([look[x] for x in j['inner_fit_ids']]);val=np.array([look[x] for x in j['inner_validation_ids']]);outer=np.arange(len(ids));deadline=time.monotonic()+min(max_seconds,cfg['budget']['unit_wall_seconds']);seed=2025+1009*int(j['fold'])+int(hashlib.sha256(j['cancer'].encode()).hexdigest()[:6],16)
  bags=FeatureBags(ps,chunk=cfg['deeppt']['patch_chunk']);ac=cfg['deeppt']['ae'];rc=cfg['deeppt']['regression']
  a=phase('ae',out/'AE_INNER',signature,bags,fit,val,ac,seed,device,deadline,ac['max_epochs'])
  ae=build_ae().to(device);ae.load_state_dict(torchload(a['checkpoint'])['model']);eb=EncodedBags(ae,bags,outer,device);del ae
  scale_inner=Scale.fit(y[fit]);require(scale_inner.supported.all(),'GENE background has unsupported head; do not zero-fill');zi=scale_inner.transform(y)
  h=phase('reg',out/'REG_INNER',signature,eb,fit,val,rc,seed,device,deadline,rc['max_epochs'],y=zi);del eb
  ar=phase('ae',out/'AE_OUTER_REFIT',signature,bags,outer,[],ac,seed,device,deadline,a['best_epoch'],select=False)
  ae=build_ae().to(device);ae.load_state_dict(torchload(ar['checkpoint'])['model']);eb=EncodedBags(ae,bags,outer,device);del ae
  scale_outer=Scale.fit(y);require(scale_outer.supported.all(),'Outer gene universe unsupported');zo=scale_outer.transform(y)
  hr=phase('reg',out/'REG_OUTER_REFIT',signature,eb,outer,[],rc,seed,device,deadline,h['best_epoch'],select=False,y=zo);del eb
  torchsave(out/'model.pt',{'ae_state':torchload(ar['checkpoint'])['model'],'reg_state':torchload(hr['checkpoint'])['model'],'target_scale':scale_outer.data(),'target_id':gene,'fit_ids':ids,'outer_test_ids':j['outer_test_ids'],'cancer':j['cancer'],'fold':j['fold'],'signature':signature,'protocol_id':PROTOCOL,'architecture':cfg['deeppt'],'source':load(Path(__file__).resolve().parents[1]/'config/AUTHOR_SOURCE.json')})
  r={'status':'COMPLETE','method':'DeepPT_UNI2H_ES_GENE_ADAPTED','signature':signature,'checkpoint':record(out/'model.pt'),'selected_AE_epochs':a['best_epoch'],'selected_reg_epochs':h['best_epoch'],'source_job':record(job_path),'train_n':len(ids),'test_n':len(j['outer_test_ids']),'AE_no_test_or_external_fit':True};write(final,r);return r
 except BudgetStop as e:write(out/'BUDGET_HOLD.json',{'reason':str(e),'status':'PARTIAL_RESUMABLE'},replace=True);raise
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':
 import hashlib
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--protocol',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--max-seconds',type=float,default=43200);a=p.parse_args();train(a.job,a.protocol,a.out,a.device,a.max_seconds)
