"""Use the exact historical FT01 raw-ES/PROGENy functions, never the old NES API."""
from __future__ import annotations
import ast, inspect
from pathlib import Path
import numpy as np
import pandas as pd
from scorer_contracts import *
NAMES={'normalize_symbol','parse_gmt','_gseapy_ssgsea','score_ssgsea_batched','score_progeny','filter_mapped_sets'}


def load_pure():
    import functional_scores as f
    return {name:getattr(f,name) for name in NAMES}


def decode_gene_ensemble(values):
    """Predictions are decoded log2(TPM+1). Project negatives to 0 AFTER fold mean."""
    require(len(values)>0,'Empty ensemble')
    shape=values[0].shape
    require(all(v.shape==shape and np.isfinite(v).all() for v in values),'Gene ensemble invalid')
    avg=sum(np.asarray(v,dtype=np.float64) for v in values)/len(values)
    rate=float((avg<0).mean());return np.maximum(avg,0.),rate

class FT01Scorer:
    def __init__(self,binding_path):
        cfg=read_json(binding_path);self.cfg=cfg
        require(cfg['input_space']=='HGNC_ORDERED_LOG2_TPM_PLUS1','Do not log twice or rerank standardized genes')
        self.genes=unique(read_json(verify_record(cfg['genes']))['ids'],'gene IDs')
        self.targets=unique(read_json(verify_record(cfg['targets']))['ids'],'target IDs')
        require(len(self.genes)==18977 and len(self.targets)==1564,'Production FT01 scope changed')
        require(cfg['scoring']['ordinary_primary']=='ssgsea_raw_es','NES prohibited for frozen task')
        locked=read_json(Path(__file__).resolve().parent/'scoring_lock.json')
        require(cfg['scoring']==locked,'FT01 numerical parameters changed')
        self.ns=load_pure()
        import gseapy
        require(gseapy.__version__==cfg['scoring']['required_gseapy_version'],'GSEApy version mismatch')
        sets={}
        for ref in cfg['gmt']:
            parsed,_=self.ns['parse_gmt'](verify_record(ref),ref['source'])
            require(not set(parsed)&set(sets),'Duplicate source term');sets.update(parsed)
        dummy=pd.DataFrame(np.zeros((len(self.genes),1)),index=self.genes,columns=['dummy'])
        self.sets,_=self.ns['filter_mapped_sets'](dummy,sets,cfg['scoring']['score_all_min_mapped_genes'],cfg['scoring']['score_all_max_mapped_genes'])
        require(len(self.sets)==3086,'Historical ordinary scoreable universe is not 3086')
        self.progeny=pd.read_csv(verify_record(cfg['progeny']))
        self.progeny['gene']=self.progeny['gene'].astype(str).str.strip().str.upper()
        ordinary=[t for t in self.targets if not t.startswith('PROGENY::')]
        require(set(ordinary).issubset(self.sets),'Missing target in frozen GMT')
    def score(self,values,patient_ids,gene_ids):
        same(gene_ids,self.genes,'scorer gene order');unique(patient_ids)
        a=np.asarray(values,dtype=np.float64)
        require(a.shape==(len(patient_ids),len(self.genes)) and np.isfinite(a).all() and (a>=0).all(),'Phi requires finite nonnegative decoded log2 expression')
        expr=pd.DataFrame(a.T,index=self.genes,columns=np.asarray(patient_ids).astype(str))
        es=self.ns['score_ssgsea_batched'](expr,self.sets,self.cfg['scoring'])
        pro,_=self.ns['score_progeny'](expr,self.progeny)
        out=pd.concat([es,pro],axis=1).reindex(index=patient_ids,columns=self.targets)
        require(np.isfinite(out.to_numpy()).all(),'Missing or nonfinite FT01 outputs')
        return out.to_numpy(dtype=np.float64)
