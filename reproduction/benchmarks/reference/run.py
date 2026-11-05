"""Training and evaluation interface for the adapted Ridge and MLP comparators."""
import argparse
from m1ref.pipeline import train_cv,export_oof,predict_external,score_gene_prediction,evaluate,compare,METHODS
def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest="command",required=True)
    a=s.add_parser("train");a.add_argument("--dataset",required=True);a.add_argument("--method",choices=sorted(METHODS),required=True);a.add_argument("--release",required=True);a.add_argument("--out",required=True);a.add_argument("--device",default="cpu");a.add_argument("--fold",type=int,choices=range(5),required=True);a.add_argument("--max-seconds",type=float,default=43200)
    a=s.add_parser("oof");a.add_argument("--models",required=True);a.add_argument("--metadata",required=True);a.add_argument("--out",required=True)
    a=s.add_parser("predict");a.add_argument("--models",required=True);a.add_argument("--features",required=True);a.add_argument("--release",required=True);a.add_argument("--out",required=True);a.add_argument("--device",default="cpu")
    a=s.add_parser("score");a.add_argument("--prediction",required=True);a.add_argument("--binding",required=True);a.add_argument("--out",required=True)
    a=s.add_parser("evaluate");a.add_argument("--prediction",required=True);a.add_argument("--truth",required=True);a.add_argument("--release",required=True);a.add_argument("--out",required=True)
    a=p.parse_args()
    if a.command=="train":train_cv(a.dataset,a.method,a.release,a.out,a.device,[a.fold],a.max_seconds)
    elif a.command=="oof":export_oof(a.models,a.metadata,a.out)
    elif a.command=="predict":predict_external(a.models,a.features,a.release,a.out,a.device)
    elif a.command=="score":score_gene_prediction(a.prediction,a.binding,a.out)
    elif a.command=="evaluate":evaluate(a.prediction,a.truth,a.out,a.release)
if __name__=="__main__":main()
