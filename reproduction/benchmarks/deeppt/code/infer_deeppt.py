"""Feature-only inference; deploy only self-produced, SHA-bound checkpoints."""
import argparse
from pathlib import Path
import numpy as np
import torch
from contracts import *
from deeppt_core import build_ae,build_reg,pooled_reg
from train_deeppt import FeatureBags,setup
from target_scale import Scale

def run(job_path,out,device='cuda:0'):
 j=load(job_path);require(j['role']=='FEATURE_ONLY_INFERENCE','Wrong inference role');require(j['cohort'] in ['TCGA_HELDOUT','CPTAC'],'Unknown cohort');ps=j['patients'];ids=[p['patient_id'] for p in ps];unique(ids)
 r=j['model'];p=verified(r);q=torch.load(p,map_location='cpu',weights_only=False);disjoint(ids,q['fit_ids']);require(all(x['cancer']==q['cancer'] for x in ps),'Wrong cancer model');require(j['fold']==q['fold'],'Fold identity mismatch')
 if j['cohort']=='TCGA_HELDOUT':require(set(ids)==set(q['outer_test_ids']),'Not exact heldout scope')
 root=Path(out);root.mkdir(parents=True,exist_ok=True);signature=digest({'job':j,'model':r})
 done=root/'COMPLETE.json'
 if done.exists():rr=load(done);require(rr['signature']==signature,'Inference identity drift');verified(rr['prediction']);return rr
 setup(2025);ae=build_ae().to(device);ae.load_state_dict(q['ae_state']);rg=build_reg(len(q['target_id'])).to(device);rg.load_state_dict(q['reg_state']);ae.eval();rg.eval();sc=Scale.from_data(q['target_scale']);bags=FeatureBags(ps,chunk=q['architecture']['patch_chunk']);pred=[]
 with torch.no_grad():
  for i,x in enumerate(ps):
   key=hashlib.sha256(x['patient_id'].encode()).hexdigest();rp=root/'rows'/(key+'.json')
   if rp.exists():
    old=load(rp);require(old['signature']==signature,'Shard identity drift')
    with np.load(verified(old['prediction']),allow_pickle=False) as z:require(str(z['patient_id'][0])==x['patient_id'],'Shard wrong patient');v=z['prediction'][0]
   else:
    encoded=(ae.encoder(a) for a in bags.chunks(i,device));v=sc.inverse(pooled_reg(rg,encoded).cpu().numpy());require(np.isfinite(v).all(),'Invalid prediction')
    pp=root/'rows'/(key+'.npz');save_npz(pp,patient_id=np.array([x['patient_id']]),prediction=v[None]);write(rp,{'signature':signature,'prediction':record(pp)})
   pred.append(v)
 pp=root/'GENE_PREDICTION.npz';save_npz(pp,patient_id=np.array(ids),cancer=np.array([q['cancer']]*len(ids)),target_id=q['target_id'],prediction=np.array(pred));rr={'signature':signature,'prediction':record(pp),'model':r,'cohort':j['cohort'],'fold':q['fold'],'n':len(ids),'scale':'LOG2_TPM_PLUS1_BEFORE_CLIP','status':'PREDICTION_FROZEN_BEFORE_TRUTH','not_yet_function_scored':True};write(done,rr);return rr
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();run(a.job,a.out,a.device)
