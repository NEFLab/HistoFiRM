import argparse
from pathlib import Path
from common import *
from tokens import make_patient

def run(spec_path,out):
    s=read_json(spec_path);require(s['role']=='FEATURE_ONLY_TOKEN_PREPARATION','Wrong role')
    require(set(s)=={'role','patients'},'Token preparation must not carry labels')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);result=[]
    for p in s['patients']:
        key=digest({'patient_id':p['patient_id'],'slides':p['slides']})[:24]
        r=make_patient(p,out/(key+'.npz'));result.append({'patient_id':p['patient_id'],'cancer':p['cancer'],'tokens':r['artifact']})
        write_json(out/'PRIVATE_INDEX_PROGRESS.json',result,replace=True)
    write_json(out/'PRIVATE_INDEX.json',{'patients':result,'n':len(result)})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.spec,a.out)
