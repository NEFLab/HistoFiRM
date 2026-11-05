"""Inference command for the adapted HE2RNA model."""
import argparse
from infer_he2rna import infer
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job",required=True);p.add_argument("--out",required=True);p.add_argument("--device",default="cuda:0")
    a=p.parse_args();infer(a.job,a.out,a.device)
