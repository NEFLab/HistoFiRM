"""Image-only HE2RNA inference; output inverse-standardized signed gene expression."""
from pathlib import Path
import time
import numpy as np
import torch
from common import *
from support import TargetScale
from he2rna_adapter import HE2RNAAdapter
from spatial_bags import SpatialBags
from train_he2rna import setup,BudgetReached

def infer(job_path,out_path,device='cuda:0',max_seconds=43200):
    deadline=time.monotonic()+float(max_seconds)
    j=read_json(job_path);require(j['role']=='IMAGE_ONLY_INFERENCE','Wrong inference role')
    signature=digest({'job':j,'code':{n:sha(Path(__file__).parent/n) for n in ['infer_he2rna.py','he2rna_adapter.py','spatial_bags.py','support.py']}})
    out_path=Path(out_path);freeze=Path(str(out_path)+'.freeze.json')
    if freeze.exists():
        r=read_json(freeze);require(r['signature']==signature,'Inference signature drift');verified(r['artifact']);return r
    require(not out_path.exists(),'Unregistered prediction; reconcile without repeating inference')
    setup(2025,threads=4)
    q=torch.load(verified(j['checkpoint']),map_location='cpu',weights_only=False)
    require(q['protocol_id']=='WSI_B3_ISOLATED_FINALMERGE_20260914_V1' and q['route']=='GENE','Wrong model protocol/route')
    require(q['cancer']==j['cancer'] and q['fold']==j['fold'],'Model cancer/fold mismatch')
    ids=[p['patient_id'] for p in j['patients']];unique(ids);disjoint(q['training_patient_ids'],ids)
    require(all(p['cancer']==j['cancer'] for p in j['patients']),'Mixed cancer inference')
    targets=np.asarray(q['target_id']).astype(str);require(len(targets)==18977,'Gene axis width')
    model=HE2RNAAdapter(output_dim=len(targets),**q['architecture']).to(device);model.load_state_dict(q['state_dict']);model.eval()
    scale=TargetScale.from_dict(q['scale']);bags=SpatialBags(j['patients'],device);rows=[]
    with torch.no_grad():
        for start in range(0,len(ids),16):
            if time.monotonic()>=deadline:raise BudgetReached("Inference deadline reached before publishing prediction")
            indices=list(range(start,min(len(ids),start+16)));p=bags.forward(model,indices).cpu().numpy()
            rows.append(scale.inverse(p))
    pred=np.concatenate(rows);require(pred.shape==(len(ids),18977) and not np.isinf(pred).any(),'Bad inference shape/value')
    save_npz(out_path,prediction=pred,patient_id=np.asarray(ids),cancer=np.asarray([j['cancer']]*len(ids)),target_id=targets)
    r={'status':'PREDICTION_FROZEN_BEFORE_TRUTH','signature':signature,'artifact':record(out_path),'model':j['checkpoint'],'cancer':j['cancer'],'fold':j['fold'],'cohort':j['cohort'],'input_space':'LOG2_TPM_PLUS1_INVERSE_STANDARDIZED','projection':'NONE_BEFORE_FIVE_FOLD_MEAN','no_truth_input':True,'patient_n':len(ids)}
    write_json(freeze,r);return r
