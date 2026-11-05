import importlib.util,json,tempfile,hashlib,unittest
from pathlib import Path
import numpy as np
SOURCE=Path(__file__).resolve().parents[1]/"scripts"/"build_fixed_manifests.py"
spec=importlib.util.spec_from_file_location("binding",SOURCE)
binding=importlib.util.module_from_spec(spec);spec.loader.exec_module(binding)

class BindingTests(unittest.TestCase):
    def fixture(self,root):
        study=root/"study";(study/"splits").mkdir(parents=True)
        axes={"input_dim":2,"target_ids":["t0","t1"],"source_ids":["A","A"],"cancer_names":["BRCA"]}
        rows=[{"patient_id":"p"+str(i),"cancer":"BRCA","outer_fold":i%5,"slides":[{"slide_id":"s"+str(i),"n_patches":2,"features":{"path":"arrays/s"+str(i)+".npy","sha256":"fixture"}}]} for i in range(10)]
        (study/"axes.json").write_text(json.dumps(axes))
        (study/"tcga_patients.json").write_text(json.dumps(rows))
        for f in range(5):
            test=[x["patient_id"] for x in rows if x["outer_fold"]==f];outer=[x["patient_id"] for x in rows if x["outer_fold"]!=f]
            (study/"splits"/("fold_"+str(f)+".json")).write_text(json.dumps({"fold":f,"roles":{"fit":outer[2:],"validation":outer[:2],"outer_train":outer,"test":test}}))
        # Deliberately reverse label order. Binding must align by patient identity.
        ordered=list(reversed(rows));lp=root/"labels.npz"
        np.savez(lp,truth=np.asarray([[int(x["patient_id"][1:]),-int(x["patient_id"][1:])] for x in ordered]),patient_ids=np.asarray([x["patient_id"] for x in ordered]),cancers=np.asarray(["BRCA"]*10),target_ids=np.asarray(axes["target_ids"]))
        m={**axes,"patients":rows,"labels":{"path":lp.name,"sha256":hashlib.sha256(lp.read_bytes()).hexdigest()}}
        manifest=root/"manifest.json";manifest.write_text(json.dumps(m))
        return study,manifest,m
    def test_alignment_roles_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);study,m,_=self.fixture(root);out=root/"bound";binding.build(m,study,out)
            for f in range(5):
                part=json.loads((out/("fold_"+str(f))/"fit.json").read_text())
                with np.load(out/("fold_"+str(f))/"fit_labels.npz") as z:
                    self.assertEqual(z["patient_ids"].tolist(),[x["patient_id"] for x in part["patients"]])
                    self.assertEqual(z["truth"][:,0].tolist(),[int(x["patient_id"][1:]) for x in part["patients"]])
                held=json.loads((out/("fold_"+str(f))/"test.json").read_text())
                self.assertNotIn("labels",held)
                self.assertFalse(Path(part["patients"][0]["slides"][0]["features"]["path"]).is_absolute())
    def test_rejects_changed_fold(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);study,m,obj=self.fixture(root);obj["patients"][0]["outer_fold"]=4;m.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError,"fold mismatch"):binding.build(m,study,root/"bound")
    def test_rejects_duplicate_patient(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);study,m,obj=self.fixture(root);obj["patients"].append(obj["patients"][0]);m.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError,"duplicate"):binding.build(m,study,root/"bound")
if __name__=="__main__":unittest.main()
