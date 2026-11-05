"""Bind supplied feature/label files to the fixed study patient partitions."""
import argparse,copy,hashlib,json,os
from pathlib import Path
import numpy as np

def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def rebase(value,source_dir,out_dir):
    if isinstance(value,dict):
        result={k:rebase(v,source_dir,out_dir) for k,v in value.items()}
        if isinstance(value.get("path"),str):
            p=Path(value["path"])
            p=(source_dir/p).resolve() if not p.is_absolute() else p.resolve()
            result["path"]=Path(os.path.relpath(p,out_dir)).as_posix()
        return result
    if isinstance(value,list): return [rebase(x,source_dir,out_dir) for x in value]
    return value
def build(manifest,study,output):
    source=Path(manifest).resolve();study=Path(study).resolve();out=Path(output).resolve()
    if out.exists(): raise ValueError("Use a new output directory.")
    m=read(source);axes=read(study/"axes.json")
    for k in ["input_dim","target_ids","cancer_names","source_ids"]:
        if m[k]!=axes[k]: raise ValueError("Ordered study axis differs: "+k)
    records=read(study/"tcga_patients.json");expected={x["patient_id"]:x for x in records}
    rows={x["patient_id"]:x for x in m["patients"]}
    if len(rows)!=len(m["patients"]) or set(rows)!=set(expected):
        raise ValueError("Supply exactly the fixed TCGA patient set, without duplicate patients.")
    for pid,row in rows.items():
        e=expected[pid]
        if row["cancer"]!=e["cancer"] or row["outer_fold"]!=e["outer_fold"]:
            raise ValueError("Patient cancer or fold mismatch: "+pid)
        want={s["slide_id"]:s["n_patches"] for s in e["slides"]}
        have={s["slide_id"]:s["n_patches"] for s in row["slides"]}
        if have!=want or len(have)!=len(row["slides"]):
            raise ValueError("Slide membership or patch count mismatch: "+pid)
    lab=m["labels"];lp=Path(lab["path"]);lp=source.parent/lp if not lp.is_absolute() else lp
    if digest(lp)!=lab["sha256"]: raise ValueError("Label file hash mismatch.")
    with np.load(lp,allow_pickle=False) as z:
        ids=z["patient_ids"].astype(str).tolist();targets=z["target_ids"].astype(str).tolist()
        cancers=z["cancers"].astype(str).tolist();truth=z["truth"].copy()
    if len(ids)!=len(set(ids)) or set(ids)!=set(rows) or targets!=axes["target_ids"]:
        raise ValueError("Label axes differ.")
    lookup={v:i for i,v in enumerate(ids)}
    if truth.shape!=(len(ids),len(targets)) or any(cancers[lookup[p]]!=rows[p]["cancer"] for p in rows):
        raise ValueError("Label shape or cancer mismatch.")
    folds=[read(study/"splits"/("fold_"+str(f)+".json")) for f in range(5)]
    for f in folds:
        a=f["roles"];fit=set(a["fit"]);val=set(a["validation"]);outer=set(a["outer_train"]);test=set(a["test"])
        if fit&val or fit|val!=outer or outer&test or outer|test!=set(rows):
            raise ValueError("Invalid fixed partition.")
    out.mkdir(parents=True)
    for f in folds:
        d=out/("fold_"+str(f["fold"]));d.mkdir()
        for role,pids in f["roles"].items():
            obj={**axes,"cohort":"TCGA16","patients":[rebase(rows[p],source.parent,d) for p in pids]}
            if role!="test":
                p=d/(role+"_labels.npz");ix=[lookup[x] for x in pids]
                np.savez_compressed(p,truth=truth[ix],patient_ids=np.asarray(pids),cancers=np.asarray([rows[x]["cancer"] for x in pids]),target_ids=np.asarray(targets))
                obj["labels"]={"path":p.name,"sha256":digest(p)}
            (d/(role+".json")).write_text(json.dumps(obj,indent=2),encoding="utf-8")
    morphology={**axes,"cohort":"TCGA16","patients":[rebase(rows[p],source.parent,out) for p in sorted(rows)]}
    (out/"morphology.json").write_text(json.dumps(morphology,indent=2),encoding="utf-8")
    (out/"binding_report.json").write_text(json.dumps({"patients":len(rows),"folds":5,"training_test_overlap":False,"input_manifest_sha256":digest(source)},indent=2),encoding="utf-8")
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True);p.add_argument("--study",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();build(a.manifest,a.study,a.output)
