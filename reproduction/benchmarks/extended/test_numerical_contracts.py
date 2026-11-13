"""CPU checks for pooling order, fixed-support scoring and relative file binding."""
from pathlib import Path
import json,tempfile
import numpy as np
import torch
from run_gene_models import RNAPath,Path2Omics
from run_mosby_adapted import load
from statistics_fixed import selection
from metrics import table,macro

def test():
    torch.manual_seed(19);x=torch.randn(11,6);lens=torch.tensor([4,7]);mu=np.zeros(6);sd=np.ones(6)
    for cls in [RNAPath,Path2Omics]:
        net=cls(6,3,mu,sd).eval();got=net(x,lens)
        if cls is RNAPath:tile=torch.relu(net.linear(x))
        else:tile=net.last(net.first(x))
        expected=torch.stack([tile[:4].mean(0),tile[4:].mean(0)])
        torch.testing.assert_close(got,expected,atol=1e-6,rtol=1e-6)
    y=np.arange(40,dtype=float).reshape(20,2);p=np.ones_like(y);cs=['A']*20
    s=selection(y,p,cs,{'A':np.ones(2,bool)})
    assert s['pearson']==0 and s['per_cancer'][0]['constant_predictions']==2
    rows=table(y,p,cs,['t1','t2'],{'A':['t1','t2']});assert macro(rows)['status']=='INCOMPLETE_FIXED_SUPPORT'
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);p=root/'config.json';p.write_text(json.dumps({'labels':{'path':'input.npz'}}))
        assert Path(load(p)['labels']['path'])==(root/'input.npz').resolve()
    return {'status':'PASS','checks':['RNAPath_relu_before_mean','Path2Omics_affine_pooling','constant_predictions_selection_zero','constant_predictions_evaluation_unavailable','relative_manifest_paths']}

if __name__=='__main__':print(json.dumps(test(),indent=2))
