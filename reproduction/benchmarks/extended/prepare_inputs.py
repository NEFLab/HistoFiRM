"""Bind user-supplied arrays to the public study partitions for extended comparators."""
from pathlib import Path
import argparse,copy,importlib.util,json,os,shutil
import numpy as np
from run_mosby_adapted import load,write,sha,ref

def bind(tcga,cptac,study,output):
    output=output.resolve();study=study.resolve()
    if output.exists():raise ValueError('Choose a new output directory.')
    script=Path(__file__).resolve().parents[2]/'scripts/build_fixed_manifests.py'
    spec=importlib.util.spec_from_file_location('fixed_manifests',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tc=load(tcga);cp=load(cptac);axes=load(study/'axes.json')
    records=load(study/'cptac_records.json')
    expected={p['patient_id']:p for p in records};actual={p['patient_id']:p for p in cp['patients']}
    assert len(actual)==len(cp['patients']) and set(actual)==set(expected)
    for key in ['target_ids','input_dim']:assert cp[key]==axes[key]
    assert set(cp['cancer_names']).issubset(axes['cancer_names'])
    for pid,p in actual.items():
        e=expected[pid];assert p['cancer']==e['cancer']
        assert {s['slide_id']:s['n_patches'] for s in p['slides']}=={s['slide_id']:s['n_patches'] for s in e['slides']}
    output.mkdir(parents=True);temp=output/'bound_folds'
    module.build(tcga,study,temp)
    for co,obj in [('TCGA16',load(temp/'morphology.json')),('CPTAC8',cp)]:
        obj.pop('labels',None)
        # Canonical order follows the existing public register.
        register=load(study/('tcga_patients.json' if co=='TCGA16' else 'cptac_records.json'))
        lookup={p['patient_id']:p for p in obj['patients']};obj['patients']=[lookup[p['patient_id']] for p in register]
        write(output/f'inputs/features/{co}_MORPHOLOGY_MMAP.json',obj)
    for fold in range(5):
        for role in ['fit','validation','outer_train','test']:
            obj=load(temp/f'fold_{fold}/{role}.json')
            if role!='test':
                src=Path(obj['labels']['path']);dest=output/f'inputs/folds/F{fold}/{role}_labels.npz'
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
                obj['labels']={'path':str(dest),'sha256':sha(dest)}
            write(output/f'inputs/folds/F{fold}/{role}.json',obj)
    # Persist user-specific references relative to each generated JSON.
    for p in output.glob('inputs/**/*.json'):
        obj=load(p)
        def relative(v):
            if isinstance(v,list):return [relative(x) for x in v]
            if isinstance(v,dict):
                out={k:relative(x) for k,x in v.items()}
                if isinstance(v.get('path'),str):out['path']=Path(os.path.relpath(v['path'],p.parent)).as_posix()
                return out
            return v
        write(p,relative(obj))
    # The temporary builder outputs are retained as provenance, not inputs to training.
    write(output/'BINDING.json',{'tcga_manifest':ref(tcga),'cptac_manifest':ref(cptac),'fixed_study':str(study),'n_tcga':len(tc['patients']),'n_cptac':len(cp['patients'])})

def prepare_method(method,cache,output):
    output=output.resolve();cache=cache.resolve()
    if output.exists():raise ValueError('Choose a new output directory.')
    assert (cache/'PREPARED.json').exists();output.mkdir(parents=True)
    if method=='abmil':
        cfg=load(Path(__file__).parent/'abmil_protocol.json');cfg['cache']=str(cache)
        write(output/'PROTOCOL.json',cfg);write(output/'PREPARED.json',{'status':'COMPLETE','cache':ref(cache/'PREPARED.json')})
    else:
        shutil.copy2(Path(__file__).parent/'gene_protocol.json',output/'PROTOCOL.json')
        (output/'input').mkdir()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='mode',required=True)
    b=sp.add_parser('bind');b.add_argument('--tcga',type=Path,required=True);b.add_argument('--cptac',type=Path,required=True);b.add_argument('--study',type=Path,required=True);b.add_argument('--output',type=Path,required=True)
    m=sp.add_parser('method');m.add_argument('--method',choices=['abmil','gene'],required=True);m.add_argument('--cache',type=Path,required=True);m.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='bind':bind(a.tcga,a.cptac,a.study,a.output)
    else:prepare_method(a.method,a.cache,a.output)
