"""Verify and install the two upstream DeepPT class files from a user download."""
import argparse,hashlib,json,shutil
from pathlib import Path
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",required=True,help="Directory containing model_AE.py and model_MLP.py from Zenodo record 11125591")
    a=p.parse_args();root=Path(__file__).resolve().parents[1]/"benchmarks"/"deeppt"
    lock=json.loads((root/"config"/"AUTHOR_SOURCE.json").read_text())
    src=Path(a.source);dst=root/"vendor"/"deeppt";pending=[]
    for name in ["model_AE.py","model_MLP.py"]:
        fp=src/name
        if hashlib.sha256(fp.read_bytes()).hexdigest()!=lock["files"][name]["sha256"]:
            raise ValueError("Upstream file does not match the pinned version: "+name)
        pending.append((fp,dst/name))
    dst.mkdir(parents=True,exist_ok=True)
    for source,dest in pending:
        if dest.exists() and dest.read_bytes()!=source.read_bytes():raise FileExistsError(dest)
        shutil.copyfile(source,dest)
    print("Two pinned author files installed. Their original license applies.")
if __name__=="__main__":main()
