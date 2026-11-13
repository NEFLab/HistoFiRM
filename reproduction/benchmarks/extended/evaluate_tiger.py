"""Evaluate frozen predictions using the original benchmark metric code."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np
from run_mosby_adapted import load,write,sha,ref

def csvwrite(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def evaluate(base,out):
    final=out/'evaluation';final.mkdir(exist_ok=False)
    assert all((out/f'F{f}/COMPLETE.json').exists() for f in range(5))
    for f in range(5):
        for r in load(out/f'F{f}/COMPLETE.json')['outputs']:assert sha(r['path'])==r['sha256']
    metric_root=Path(__file__).resolve().parent
    sys.path.insert(0,str(metric_root))
    from metrics import table,macro,coverage,KEYS
    catalog=load(base/'inputs/evaluation_sources.json')
    panelpath=base/'inputs/features/support_panel.json';panel=load(panelpath)
    overview=[];provenance=[ref(panelpath),ref(metric_root/'metrics.py')]
    for co in ['TCGA16','CPTAC8']:
        idx=load(out.parent/'mosby_adapted'/f'{co}_index.json');ids=idx['patient_ids'];cs=idx['cancers'];ts=idx['target_ids']
        p=np.full((len(ids),len(ts)),np.nan);by={x:i for i,x in enumerate(ids)}
        if co=='TCGA16':
            seen=set()
            for f in range(5):
                a=np.load(out/f'F{f}/{co}_prediction.npz');assert a['target_ids'].tolist()==ts
                ii=[by[x] for x in a['patient_ids']];assert not seen&set(ii);seen.update(ii)
                assert a['cancers'].tolist()==[cs[i] for i in ii];p[ii]=a['prediction']
            assert len(seen)==len(ids)
        else:
            preds=[]
            for f in range(5):
                a=np.load(out/f'F{f}/{co}_prediction.npz')
                assert a['patient_ids'].tolist()==ids and a['target_ids'].tolist()==ts and a['cancers'].tolist()==cs
                preds.append(a['prediction'])
            p=np.mean(preds,axis=0)
        np.savez(final/f'{co}_MERGED.npz',patient_ids=np.array(ids),cancers=np.array(cs),target_ids=np.array(ts),prediction=p)
        ent=catalog['sources'][co]['TRUTH'];src=ent['source'];assert sha(src['path'])==src['sha256'];provenance.append(src)
        a=np.load(src['path']);keys=ent['keys'];si=a[keys['patient_ids']].astype(str).tolist();st=a[keys['target_ids']].astype(str).tolist()
        assert set(si)==set(ids) and set(st)==set(ts) and len(set(si))==len(si)
        yi={x:i for i,x in enumerate(si)};ti={x:i for i,x in enumerate(st)}
        assert a[keys['cancers']][[yi[x] for x in ids]].astype(str).tolist()==cs
        y=a[keys['truth']][np.ix_([yi[x] for x in ids],[ti[x] for x in ts])]
        methods={'TIGER_UNI2H_ADAPTED':p}
        for label,pred in methods.items():
            rows=table(y,pred,cs,ts,panel,10);chosen=[r for r in rows if r['panel_membership']]
            summary={'cohort':co,'method':label,'patients_or_records':len(ids),'targets':len(ts),
                     'metrics':{k:macro(chosen,k) for k in KEYS},'coverage':coverage(chosen),'all_target_coverage':coverage(rows)}
            csvwrite(final/f'{co}_{label}_ALL_FUNCTIONS.csv',rows)
            pc=[]
            for c in sorted(set(cs)):
                cr=[r for r in chosen if r['cancer']==c]
                pc.append({'cohort':co,'method':label,'cancer':c,'n':cs.count(c),**{k:macro(cr,k)['value'] for k in KEYS}})
            csvwrite(final/f'{co}_{label}_PER_CANCER.csv',pc);write(final/f'{co}_{label}_SUMMARY.json',summary)
            overview.append({'cohort':co,'method':label,'n':len(ids),**{k:summary['metrics'][k]['value'] for k in KEYS},'status':summary['metrics']['pearson']['status']})
            print(json.dumps(overview[-1]),flush=True)
    csvwrite(final/'OVERALL.csv',overview);write(final/'OVERALL.json',overview);write(final/'EVALUATION_PROVENANCE.json',provenance)
    write(final/'COMPLETE.json',{'status':'COMPLETE','new_method':'TIGER_UNI2H_ADAPTED','no_new_CI_or_significance':True,'outputs':[ref(p) for p in final.iterdir() if p.is_file()]})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();evaluate(a.base,a.out)
