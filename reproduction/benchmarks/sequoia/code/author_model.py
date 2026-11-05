"""Load the exact MIT author core returned by M1A. No surrogate Transformer."""
import importlib.util,sys
from pathlib import Path
from common import require,sha
PIN='044780b49a7bff3bf4d29d63b807b4471c32f0ba'
CORE_SHA='4a0fdc074f8ed9f44afe9f59a7cd5a7e793fdad077c93bb9ab5069abc072071e'
DEFAULT_CORE=Path(__file__).resolve().parents[1]/'vendor/sequoia/tformer_lin.py'

def author_module(path=None):
    p=Path(path or DEFAULT_CORE);require(sha(p)==CORE_SHA,'Not the pinned SEQUOIA core')
    name='wsi_author_sequoia_'+CORE_SHA[:12]
    if name in sys.modules:return sys.modules[name]
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m)
    return m

def build(outputs,device='cpu',core=None,architecture=None):
    a=dict(architecture or dict(input_dim=1536,depth=6,nheads=16,dimensions_f=64,dimensions_s=64,dimensions_c=64,num_clusters=100))
    model=author_module(core).ViS(num_outputs=int(outputs),device=device,**a).to(device)
    return model
