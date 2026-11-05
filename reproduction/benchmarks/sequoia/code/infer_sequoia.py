"""Separate image-only inference. No target/truth file is accepted by this job schema."""
import argparse,time
from pathlib import Path
import torch,numpy as np
from common import *
from author_model import build,CORE_SHA
from support import TargetScale
from tokens import patient_forward
from train_sequoia import Bags,setup,PROTOCOL

def infer(job_path,out,device='cuda:0'):
    j=read_json(job_path);allowed={'role','cohort','checkpoint','patients','fold','cancer','route'}
    require(set(j)==allowed,'Inference schema contains additional fields / missing bindings')
    require(j['role']=='IMAGE_ONLY_INFERENCE' and j['cohort'] in ['TCGA_OOF','CPTAC'],'Wrong inference role')
    out=Path(out);freeze=Path(str(out)+'.freeze.json');signature=digest(j)
    if freeze.exists():
        r=read_json(freeze);require(r['signature']==signature,'Existing inference differs');verified(r['artifact']);return r
    require(not out.exists(),'Unregistered prediction; reconcile before recomputation')
    ck=verified(j['checkpoint']);q=torch.load(ck,map_location='cpu',weights_only=False)
    require(q['source_core_sha256']==CORE_SHA and q['protocol_id']==PROTOCOL,'Wrong model provenance')
    require((q['fold'],q['cancer'],q['route'])==(j['fold'],j['cancer'],j['route']),'Wrong member')
    ids=[p['patient_id'] for p in j['patients']];unique(ids);disjoint(q['training_patient_ids'],ids)
    require(all(p['cancer']==q['cancer'] for p in j['patients']),'Wrong cancer inputs')
    require(all(set(p)=={'patient_id','cancer','tokens'} for p in j['patients']),'Inference patient includes extra data')
    setup(2025);s=TargetScale.from_dict(q['scale']);m=build(len(q['target_id']),device,architecture=q['architecture']);m.load_state_dict(q['state_dict']);m.eval();bags=Bags(j['patients'],device)
    pred=[];start=time.monotonic()
    with torch.no_grad():
        for i in range(len(ids)):
            x,w=bags.get(i);p=patient_forward(m,x,w,4).cpu().numpy();require(np.isfinite(p).all(),'Nonfinite native output');pred.append(s.inverse(p)[0])
    pred=np.asarray(pred);require(np.isfinite(pred[:,s.supported]).all(),'Nonfinite supported predictions')
    save_npz(out,prediction=pred,patient_id=np.asarray(ids),cancer=np.asarray([q['cancer']]*len(ids)),
        target_id=q['target_id'],supported=s.supported,train_label_counts=s.count)
    r={'status':'PREDICTIONS_FROZEN_BEFORE_TRUTH','signature':signature,'artifact':record(out),
       'fold':q['fold'],'cancer':q['cancer'],'route':q['route'],'cohort':j['cohort'],
       'checkpoint':j['checkpoint'],'n_patients':len(ids),'wall_seconds':time.monotonic()-start,
       'unsupported_outputs_preserved_as_NaN':True,'no_gene_projection_yet':q['route']=='GENE'}
    write_json(freeze,r);return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();infer(a.job,a.out,a.device)
