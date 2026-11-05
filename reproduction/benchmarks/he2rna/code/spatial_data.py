"""B3 feature-only spatial supertile binding and resumable single-writer cache.

Raw inputs and registrations are read-only. This module never opens label arrays.
HE2RNA fixed-author spatial clustering; no SEQUOIA token reuse.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, json, os, time, uuid
from pathlib import Path

AUTHOR_COMMIT = "d47f481fbcaab41a49635e4a35e810bf30eae0e0"
KMCUDA_COMMIT = "7dcd28c49a6c76bb9c5e1b19c23976a1f76e1b3b"
GPU_UUID = "GPU-e6a7c285-286f-5d3e-c488-3247ea3fc5e4"
PINS = {
 "metadata": "0d310ad33deb07b86c6ac31ffc763cf45e2f596748cbed9afc65a1d6677af548",
 "splits": "0e5dbcd41527196dbb17e1525df7f5ebe660b0682f5727250db1d66db8b16244",
 "label_index": "1a68c0f89b6c28cca222ada338685ff1f754495778993a92b5193cb1391d842d",
 "patch_manifest": "3ea4c0b94e6bafaed44cb1cf89a7e138ef3a9a6594542194d457796d77eed896",
 "external_registry": "3d6937daad885b6f3843899f4892562c3748a00c922a44f47eabeac306cbf270",
}
RECIPE = {
 "author_commit": AUTHOR_COMMIT,
 "dependency_commit": KMCUDA_COMMIT,
 "backend": "libKMCUDA.kmeans_cuda",
 "input": "REGISTERED_SLIDE_XY_COORDINATES_FLOAT32",
 "clusters": "min(100,original_slide_patch_count)",
 "tolerance": 0.01, "init": "k-means++", "metric": "L2",
 "yinyang_t": 0, "verbosity": 0,
 "seed_policy": "AUTHOR_TIME_SECONDS_DEFAULT_EXPLICITLY_RECORDED_PER_SLIDE",
 "device": "FIRST_AND_ONLY_VISIBLE_GPU5",
 "feature_reduction": "MEAN_FLOAT32_PER_NONEMPTY_SPATIAL_CLUSTER",
 "slide_aggregation": "TRAINER_PREDICTION_WEIGHTED_BY_ORIGINAL_PATCH_COUNT",
 "padding": "ZERO_VALUES_WITH_EXPLICIT_VALID_LENGTHS_ONLY",
 "singleton_slide": "ONE_EXACT_CLUSTER_WITHOUT_NATIVE_K_LESS_THAN_2_ERROR",
 "maximum_supertiles": 100, "feature_dim": 1536,
 "workers": 1, "threads": 8,
}

def require(v, msg):
 if not v: raise ValueError(msg)
def read_json(p):
 return json.loads(Path(p).read_text(encoding="utf8"))
def digest(x):
 return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def record(p):
 p=Path(p);return {"path":str(p),"bytes":p.stat().st_size,"sha256":sha(p)}
def verify(ref):
 p=Path(ref["path"])
 require(p.is_file() and p.stat().st_size==ref["bytes"],"Source size changed")
 require(sha(p)==ref["sha256"],"Source hash changed")
 return p
def pinned(p,key):
 p=Path(p); require(sha(p)==PINS[key],"Frozen "+key+" changed");return p

def atomic_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+".partial."+uuid.uuid4().hex)
 with tmp.open("x",encoding="utf8") as f:
  json.dump(x,f,sort_keys=True,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)

def copy_unchanged(source,destination):
 source=Path(source);destination=Path(destination)
 if destination.exists():
  require(sha(destination)==sha(source),"Existing frozen B3 metadata copy differs")
  return
 tmp=destination.with_name(destination.name+".partial."+uuid.uuid4().hex)
 with source.open("rb") as fi, tmp.open("xb") as fo:
  for block in iter(lambda:fi.read(1024*1024),b""):fo.write(block)
  fo.flush();os.fsync(fo.fileno())
 os.rename(tmp,destination)

def contained(p,root):
 p=Path(p).resolve();p.relative_to(Path(root).resolve());return p

@contextlib.contextmanager
def one_writer(root):
 import fcntl
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 with (root/"spatial_cache.writer.lock").open("a") as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)

def bound_cpu():
 for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:os.environ[k]="8"
 if hasattr(os,"sched_getaffinity"):
  cores=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cores[32:40] if len(cores)>=40 else cores[:8])

def bind_plan(project_parent,b3_root,write=True):
 """Emit private identities only to B3; stdout is aggregate counts/schema."""
 import csv
 A=Path(project_parent);B=A/"WSI_M1B_AUTHOR_MODELS_20260913_V1"
 P=A.parent;W=Path(b3_root)
 require(W.name=="WSI_B3_ISOLATED_20260914_V1","Wrong B3 write root")
 mp=pinned(B/"datasets/PRIVATE_PATIENT_METADATA.json","metadata")
 sp=pinned(B/"datasets/PRIVATE_SPLITS.json","splits")
 lp=pinned(B/"datasets/LabelPartitionIndex.json","label_index")
 pm=pinned(P/"04_visibility_models/M01A_ExactAdditiveB00Lift_BaselineBoundaryClosure_20260714/canonical_interface/patch_manifest.csv","patch_manifest")
 er=pinned(A/"WSI_M0_V3_R2_20260912/runtime/CPTACInputRegistry.server_only.json","external_registry")
 meta=read_json(mp);splits=read_json(sp);label_index=read_json(lp)
 require({k:len(v) for k,v in meta.items()}=={"TCGA":6781,"CPTAC":1128},"Patient population changed")
 require([f["fold"] for f in splits["folds"]]==list(range(5)),"Original five folds changed")
 refs={k:record(p) for k,p in [("metadata",mp),("splits",sp),("label_index",lp),("patch_manifest",pm),("external_registry",er)]}
 with pm.open(newline="") as f:patch=list(csv.DictReader(f))
 require(len(patch)==6834,"TCGA slide count changed")
 pmap={}
 for p in patch:
  key=(p["patient_id"],p["slide_id"]);require(key not in pmap,"Duplicate registered TCGA slide")
  require(p["feature_path"]==p["coordinate_path"],"Separate feature/coordinate source requires explicit lineage")
  require(p["feature_key"]=="features" and p["coordinate_key"]=="coords_target10" and p["patch_id_key"]=="patch_index","Registered H5 schema changed")
  require(p["row_order_contract"]=="PROVEN_SAME_H5_CODE9A_SHUFFLE_FALSE_FEATURE_ROWS_AND_COPIED_COORDS_WITH_SEQUENTIAL_PATCH_INDEX","Unproven feature-coordinate row order")
  require(p["coordinate_units"]=="TARGET10_GRID_PIXELS_ONLY","Coordinate units changed")
  pmap[key]=p
 plans={};plan_refs=[]
 for cohort in ["TCGA","CPTAC"]:
  pts=[]
  for slot in [0,1]:
   planp=B/f"datasets/PRIVATE_TOKEN_PLAN.{cohort}.{slot}.json";plan=read_json(planp)
   require(plan["cohort"]==cohort,"Plan cohort mismatch");plan_refs.append(record(planp));pts.extend(plan["patients"])
  require(len(pts)==len(meta[cohort]),"Feature plan count mismatch")
  pts_by_id={p["patient_id"]:p for p in pts};require(len(pts_by_id)==len(pts),"Duplicate patient feature plan")
  require(set(pts_by_id)=={m["patient_id"] for m in meta[cohort]},"Feature plan metadata identity mismatch")
  rows=[]
  for m in meta[cohort]:
   p=pts_by_id[m["patient_id"]];require(p["cancer"]==m["cancer"] and p["cohort"]==cohort,"Feature identity mismatch")
   slides=[]
   for s in p["slides"]:
    if cohort=="TCGA":
     r=pmap[(m["patient_id"],s["slide_id"])];require(r["cancer"]==m["cancer"] and s["path"]==r["feature_path"] and s["feature_key"]==r["feature_key"],"TCGA feature plan drift")
     fp=Path(r["feature_path"]);require(fp.stat().st_size==s["bytes_at_binding"],"TCGA source byte size drift")
     slides.append({"slide_id":s["slide_id"],"format":"H5","path":str(fp),"bytes":fp.stat().st_size,"feature_key":"features","coordinate_key":"coords_target10","patch_id_key":"patch_index","coordinate_units":r["coordinate_units"],"row_order_contract":r["row_order_contract"],"source_hash_policy":"HASH_AND_PIN_ON_FIRST_RAW_READ"})
    else:
     cp=verify(s["conversion"]);c=read_json(cp);q=c["proof"]
     require(c["patient_id"]==m["patient_id"] and c["cancer"]==m["cancer"],"CPTAC conversion identity mismatch")
     require(c["RNA_reads"]==0 and q["storage_widening_lossless"] and q["feature_coordinate_patch_index_exact"] and q["embedded_coordinates_exact"],"CPTAC exact coordinate identity not established")
     require(str(q["single_registered_slide"])==s["slide_id"]==str(c["source_row"]["feature_slide_id"]),"CPTAC single slide identity mismatch")
     require(q["coordinate_scale_factor"]==1 and q["coordinate_key"]=="coords_target10","CPTAC coordinate units changed")
     require(all(c["artifact"][k]==s["feature"][k] for k in ["path","bytes","sha256"]) and s["feature"]["format"]=="npz" and s["feature"]["key"]=="features","CPTAC registered artifact mismatch")
     slides.append({"slide_id":s["slide_id"],"format":"NPZ_REGISTERED_SINGLE_SLIDE","feature":s["feature"],"conversion":s["conversion"],"feature_key":"features","coordinate_key":"coordinates","slide_index_key":"slides","n_patches":int(c["source_row"]["n_features"]),"coordinate_units":"TARGET10_GRID_PIXELS_ONLY","row_order_contract":"FROZEN_CONVERSION_EXACT_FEATURE_COORDINATE_PATCH_INDEX"})
   require(slides and len({s["slide_id"] for s in slides})==len(slides),"Invalid per-patient slide identities")
   rows.append({"patient_id":m["patient_id"],"cancer":m["cancer"],"cohort":cohort,"outer_fold":int(m["outer_fold"]),"slides":slides})
  plans[cohort]=rows
 for f in splits["folds"]:
  sets=[set(f[k]) for k in ["inner_fit_patient_ids","inner_validation_patient_ids","outer_test_patient_ids"]]
  require(not any(sets[i]&sets[j] for i in range(3) for j in range(i)),"Inherited split overlap")
  require(set.union(*sets)=={m["patient_id"] for m in meta["TCGA"]},"Inherited split coverage")
  require(sets[2]=={m["patient_id"] for m in meta["TCGA"] if m["outer_fold"]==f["fold"]},"Metadata original fold mismatch")
 obj={"schema":"B3_PRIVATE_SPATIAL_SUPERTILE_PLAN_V1","sources":refs,"source_feature_plans":plan_refs,"recipe":RECIPE,"recipe_sha256":digest(RECIPE),"patients":plans,"label_partitions":[p for p in label_index["partitions"] if p["route"]=="GENE"],"truth_reads":0}
 dst=contained(W/"datasets/PRIVATE_SUPERTILE_PLAN.json",W)
 if not write:
  return {"status":"READONLY_BINDING_VERIFIED","patient_counts":{k:len(v) for k,v in plans.items()},"slide_counts":{k:sum(len(p["slides"]) for p in v) for k,v in plans.items()},"source_hashes":{k:v["sha256"] for k,v in refs.items()},"recipe_sha256":obj["recipe_sha256"],"truth_reads":0}
 with one_writer(dst.parent):
  for original in [mp,sp,lp]:copy_unchanged(original,dst.parent/original.name)
  if dst.exists():require(read_json(dst)==obj,"Existing B3 spatial plan identity differs")
  else:atomic_json(dst,obj)
 return {"status":"SPATIAL_FEATURES_AND_ORIGINAL_SPLITS_BOUND","artifact":record(dst),"patient_counts":{k:len(v) for k,v in plans.items()},"slide_counts":{k:sum(len(p["slides"]) for p in v) for k,v in plans.items()},"schema":obj["schema"],"recipe_sha256":obj["recipe_sha256"],"truth_reads":0,"RNA_array_reads":0}

def read_slide(slide):
 import numpy as np
 if slide["format"]=="H5":
  import h5py
  p=Path(slide["path"]);before=p.stat();require(before.st_size==slide["bytes"],"Registered H5 size changed")
  ref=record(p);ref["mtime_ns"]=before.st_mtime_ns
  with h5py.File(p,"r") as h:
   x=np.asarray(h[slide["feature_key"]][:],dtype=np.float32)
   xy=np.asarray(h[slide["coordinate_key"]][:])
   ix=np.asarray(h[slide["patch_id_key"]][:])
  require(ix.dtype.kind in "ui" and np.array_equal(ix,np.arange(len(x),dtype=ix.dtype)),"H5 patch_index is not exact sequential feature row identity")
  after=p.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),"Frozen H5 changed during read")
 else:
  p=verify(slide["feature"]);ref=dict(slide["feature"]);ref["mtime_ns"]=p.stat().st_mtime_ns
  verify(slide["conversion"])
  with np.load(p,allow_pickle=False) as z:
   x=np.asarray(z[slide["feature_key"]],dtype=np.float32);xy=np.asarray(z[slide["coordinate_key"]]);idx=z[slide["slide_index_key"]]
  require(len(x)==slide["n_patches"] and idx.shape==(len(x),) and np.all(idx==0),"CPTAC registered single slide cardinality differs")
 require(x.ndim==2 and x.shape[1]==1536 and len(x)>0,"Invalid UNI2H bag shape")
 require(xy.shape==(len(x),2) and np.isfinite(x).all() and np.isfinite(xy).all(),"Feature-coordinate shape or finiteness differs")
 xy32=np.asarray(xy,dtype=np.float32)
 require(np.array_equal(xy.astype(np.float64),xy32.astype(np.float64)),"Spatial coordinate FP32 conversion is not lossless")
 return x,xy32,ref

def native_cluster(coords,k,seed):
 if k==1:
  import numpy as np
  return np.zeros(len(coords),dtype=np.uint32)
 require(os.environ.get("CUDA_VISIBLE_DEVICES")==GPU_UUID,"Author CUDA preprocessing must expose only authorized GPU5 UUID")
 from libKMCUDA import kmeans_cuda
 _,assignment=kmeans_cuda(coords,k,tolerance=.01,init="k-means++",yinyang_t=0,metric="L2",seed=seed,device=1,verbosity=0)
 return assignment

def pool_spatial(features,coords,cluster=native_cluster,seed=None):
 """Return an unpadded valid bag; feature values never define padding."""
 import numpy as np
 require(features.ndim==2 and coords.shape==(len(features),2) and len(features)>0,"Spatial input alignment")
 seed=int(time.time()) if seed is None else int(seed)
 k=min(100,len(features));labels=np.asarray(cluster(coords,k,seed))
 require(labels.shape==(len(features),) and labels.dtype.kind in "ui","Cluster assignment schema")
 require(np.all(labels>=0) and np.all(labels<k),"Invalid cluster assignment")
 live=np.unique(labels)
 out=np.asarray([np.mean(features[labels==c],axis=0,dtype=np.float32) for c in live],dtype=np.float32)
 require(1<=len(out)<=k and np.isfinite(out).all(),"Invalid spatial means")
 return out,{"seed":seed,"requested_clusters":k,"nonempty_clusters":len(out),"patch_count":len(features),"assignment_sha256":hashlib.sha256(labels.astype("<u4").tobytes()).hexdigest()}

def cache_patient(patient,plan,cache_root,cluster=native_cluster):
 import numpy as np
 cache_root=Path(cache_root);identity=digest({"patient":patient,"sources":plan["sources"],"recipe_sha256":plan["recipe_sha256"]})
 token=digest({"cohort":patient["cohort"],"patient_id":patient["patient_id"]})
 folder=cache_root/patient["cohort"]/patient["cancer"]/token
 receipt=folder/"receipt.json"
 if folder.exists():
  require(receipt.exists(),"Incomplete visible patient cache; explicit reconciliation required")
  r=read_json(receipt);require(r["identity_sha256"]==identity,"Existing cache input/recipe differs")
  verify(r["artifact"])
  for detail in r["slides"]:
   ref=detail["source"];st=Path(ref["path"]).stat()
   require(st.st_size==ref["bytes"] and st.st_mtime_ns==ref["mtime_ns"],"Pinned raw feature source stat changed; explicit hash reconciliation required")
  return {"patient_id":patient["patient_id"],"cancer":patient["cancer"],"supertiles":r["artifact"]}
 folder.parent.mkdir(parents=True,exist_ok=True)
 tmp=folder.with_name(folder.name+".partial."+uuid.uuid4().hex);tmp.mkdir()
 xs=[];lengths=[];counts=[];details=[]
 for s in patient["slides"]:
  x,xy,source=read_slide(s);y,info=pool_spatial(x,xy,cluster=cluster)
  xs.append(y);lengths.append(len(y));counts.append(len(x));details.append({"slide_id":s["slide_id"],"source":source,"clustering":info});del x,xy
 bags=np.zeros((len(xs),100,1536),dtype=np.float32)
 for i,x in enumerate(xs):bags[i,:len(x)]=x
 file=tmp/"supertiles.npz"
 with file.open("xb") as f:
  np.savez(f,supertiles=bags,valid_lengths=np.asarray(lengths,dtype=np.int64),patch_counts=np.asarray(counts,dtype=np.int64),slide_id=np.asarray([s["slide_id"] for s in patient["slides"]]),patient_id=np.asarray([patient["patient_id"]]))
  f.flush();os.fsync(f.fileno())
 artifact=record(file);artifact["path"]=str(folder/file.name)
 r={"schema":"B3_PATIENT_SPATIAL_SUPERTILES_V1","identity_sha256":identity,"recipe_sha256":plan["recipe_sha256"],"cohort":patient["cohort"],"cancer":patient["cancer"],"patient_id":patient["patient_id"],"artifact":artifact,"slides":details,"valid_lengths":lengths,"patch_counts":counts,"feature_dim":1536,"truth_reads":0}
 atomic_json(tmp/"receipt.json",r)
 os.rename(tmp,folder)
 return {"patient_id":patient["patient_id"],"cancer":patient["cancer"],"supertiles":artifact}

def prepare_cache(plan_path,b3_root,cohort,cancer,limit=None,allow_gpu=False,max_seconds=None,deadline=None):
 """Stop at patient boundaries; deadline uses the time.monotonic clock."""
 import math,resource
 started=time.monotonic();u0=resource.getrusage(resource.RUSAGE_SELF)
 if max_seconds is not None:
  require(math.isfinite(max_seconds) and max_seconds>=0,"max_seconds must be finite and nonnegative")
  relative_deadline=started+max_seconds
 else:relative_deadline=None
 if deadline is not None:require(math.isfinite(deadline),"deadline must be a finite monotonic timestamp")
 deadlines=[value for value in [relative_deadline,deadline] if value is not None]
 effective_deadline=min(deadlines) if deadlines else None
 budget_exhausted=False
 require(allow_gpu,"Native author preprocessing uses GPU; explicit bounded launch authorization required")
 W=Path(b3_root);require(W.name=="WSI_B3_ISOLATED_20260914_V1","Wrong B3 root")
 plan=read_json(plan_path);require(plan["recipe"]==RECIPE and plan["recipe_sha256"]==digest(RECIPE),"Spatial recipe changed")
 for ref in plan["sources"].values():verify(ref)
 selected=[p for p in plan["patients"][cohort] if p["cancer"]==cancer]
 require(selected,"Empty cohort/cancer")
 target_n=len(selected)
 if limit is not None:require(limit>0,"limit must be positive");selected=selected[:limit]
 cache=contained(W/"cache/spatial_supertiles",W);results=[]
 with one_writer(cache):
  for p in selected:
   if effective_deadline is not None and time.monotonic()>=effective_deadline:
    budget_exhausted=True;break
   results.append(cache_patient(p,plan,cache))
   if effective_deadline is not None and time.monotonic()>=effective_deadline:
    budget_exhausted=True;break
  complete=len(results)==target_n and not budget_exhausted
  basename=f"SUPERTILE_INDEX.{cohort}.{cancer}.json" if complete else f"PRIVATE_SUPERTILE_PROGRESS.{cohort}.{cancer}.json"
  out=contained(W/"datasets"/basename,W)
  payload={"cohort":cohort,"cancer":cancer,"records":results,"n_expected":target_n,"complete":complete,"plan":record(plan_path),"recipe_sha256":plan["recipe_sha256"]}
  if budget_exhausted:payload["stop_reason"]="PREPROCESSING_DEADLINE_REACHED"
  if complete:
   expected={p["patient_id"] for p in plan["patients"][cohort] if p["cancer"]==cancer}
   require({p["patient_id"] for p in results}==expected,"Complete index patient coverage mismatch")
   if out.exists():require(read_json(out)==payload,"Frozen complete cache index identity differs")
   else:atomic_json(out,payload)
  else:atomic_json(out,payload)
 u=resource.getrusage(resource.RUSAGE_SELF)
 result={"status":"INCOMPLETE" if budget_exhausted else ("COMPLETE" if complete else "BOUNDED_PARTIAL"),"cohort":cohort,"cancer":cancer,"completed_patients":len(results),"expected_patients":target_n,"index":record(out),"cpu_seconds":u.ru_utime+u.ru_stime-u0.ru_utime-u0.ru_stime,"wall_seconds":time.monotonic()-started,"truth_reads":0,"GPU_runtime_metering":"PARENT_LAUNCHER_RESERVED_WALL"}
 if effective_deadline is not None:
  result["budget"]={"max_seconds":max_seconds,"deadline_monotonic":effective_deadline,"exhausted":budget_exhausted,"check_granularity":"COMPLETED_PATIENT_BOUNDARY"}
 if budget_exhausted:result["stop_reason"]="PREPROCESSING_DEADLINE_REACHED"
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument("action",choices=["bind","cache"]);p.add_argument("--b3-root",required=True);p.add_argument("--project-parent");p.add_argument("--plan");p.add_argument("--cohort",choices=["TCGA","CPTAC"]);p.add_argument("--cancer");p.add_argument("--limit",type=int);p.add_argument("--allow-gpu-preprocessing",action="store_true");p.add_argument("--max-seconds",type=float);p.add_argument("--deadline-monotonic",type=float);a=p.parse_args();bound_cpu()
 if a.action=="bind":r=bind_plan(a.project_parent,a.b3_root)
 else:r=prepare_cache(a.plan,a.b3_root,a.cohort,a.cancer,a.limit,a.allow_gpu_preprocessing,a.max_seconds,a.deadline_monotonic)
 print(json.dumps(r,sort_keys=True))
if __name__=="__main__":main()
