"""Real reference training/refit, blinded prediction, freeze and separate evaluation."""
from __future__ import annotations
import pickle,time
from pathlib import Path
import numpy as np
import pandas as pd
from .contracts import *
from .data import load_dataset,inner_indices
from .models import Ridge,ScopeRidge,MLP,Scale,scaled_mse
from .metrics import table,summarize,paired_bootstrap
from .scoring import FT01Scorer,decode_gene_ensemble

METHODS={'RIDGE_DIRECT_SHARED':('RIDGE','TCGA_F','shared'),
         'RIDGE_GENE_PHI_SHARED':('RIDGE','TCGA_G','shared'),
         'MLP_DIRECT_SHARED':('MLP','TCGA_F','shared'),
         'MLP_GENE_PHI_SHARED':('MLP','TCGA_G','shared'),
         'RIDGE_DIRECT_SINGLE':('RIDGE','TCGA_F','per_cancer'),
         'RIDGE_DIRECT_HIER_REFERENCE':('RIDGE','TCGA_F','hierarchical')}

def source_digest():
    return digest({p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})

def candidate_scores(x,y,c,ix,iv,alphas,scope):
    """Single-cancer models select alpha within that cancer only; shared uses pooled fit labels."""
    s=Scale().fit(y[ix],True);trials=[]
    for alpha in alphas:
        model=ScopeRidge(scope,alpha).fit(x[ix],y[ix],c[ix])
        value=scaled_mse(model.predict(x[iv],c[iv]),y[iv],s)
        trials.append({'alpha':alpha,'inner_scaled_mse':value})
    idx=int(np.argmin([r['inner_scaled_mse'] for r in trials]));return trials[idx]['alpha'],trials


def train_cv(dataset_path,method,release_path,out_dir,device='cpu',folds=range(5),max_seconds=43200):
    r=check_release(release_path,'fit_reference');require(method in METHODS,'Unapproved reference method')
    family,kind,scope=METHODS[method];cfg,m,x,y,targets=load_dataset(dataset_path,kind)
    if r.get('production',True):
        protocol=read_json(Path(__file__).resolve().parents[1]/'config/M1_PROTOCOL.json')
        require(len(m)==6781 and set(m.cancer)==set(protocol['TCGA_cancers']) and x.shape[1]==1536,'Production TCGA scope changed')
        require(len(targets)==(18977 if kind=='TCGA_G' else 1564),'Production output dimension changed')
    params=r['reference_config'];out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    clock=Clock(min(float(max_seconds),float(r['max_unit_wall_seconds'])))
    signature=digest({'dataset':cfg,'method':method,'reference_config':params,'code':source_digest(),'release':r['protocol_id']})
    c=m.cancer.to_numpy(str)
    for fold in list(folds):
        require(fold in range(5),'Invalid outer fold')
        root=out/f'fold_{fold}';root.mkdir(exist_ok=True)
        with exclusive_lock(root/'unit.lock'):
            recpath=root/'PRIVATE_FOLD_RECEIPT.json'
            if recpath.exists():
                old=read_json(recpath);require(old['signature']==signature,'Completed unit has different signature')
                verify_record(old['checkpoint']);verify_record(old['oof_prediction']);continue
            require(not (root/'model.pkl').exists() and not (root/'oof.npz').exists(),'Partial output requires evidence reconciliation, not overwrite')
            clock=Clock(min(float(max_seconds),float(r['max_unit_wall_seconds'])))
            tstart=time.monotonic();ix,iv,it=inner_indices(m,fold,params['seed'],params['inner_fraction'])
            outer=np.sort(np.r_[ix,iv]);clock.check()
            if family=='RIDGE':
                if scope=='per_cancer':
                    final=ScopeRidge('per_cancer');final.local={};final.global_model=None;final.output_dim=y.shape[1];trials={}
                    for cancer in sorted(set(c)):
                        ci=ix[c[ix]==cancer];cv=iv[c[iv]==cancer];co=outer[c[outer]==cancer]
                        alpha,trial=candidate_scores(x,y,c,ci,cv,params['ridge_alphas'],'shared');clock.check()
                        final.local[cancer]=(1.,Ridge(alpha).fit(x[co],y[co]));trials[cancer]=dict(selected_alpha=alpha,trials=trial)
                elif scope=='hierarchical':
                    # Prespecified mechanism comparator, not separately optimized on external data.
                    final=ScopeRidge('hierarchical',100.,500.,50.).fit(x[outer],y[outer],c[outer])
                    trials={'preset':{'global_alpha':100,'cancer_alpha':500,'shrinkage':50},'not_the_frozen_anchor':True}
                else:
                    alpha,trials=candidate_scores(x,y,c,ix,iv,params['ridge_alphas'],'shared');clock.check()
                    final=ScopeRidge('shared',alpha).fit(x[outer],y[outer],c[outer])
                pred=final.predict(x[it],c[it]);selection={'criterion':'INNER_TRAIN_SCALED_VALIDATION_MSE','trials':trials}
            else:
                k=dict(params['mlp']);seed=params['seed']+fold*1009
                tune=MLP(k,seed,device).train(x[ix],y[ix],clock,valid=(x[iv],y[iv]))
                best=tune.best_epoch;history=tune.history;del tune
                final=MLP(k,seed,device).train(x[outer],y[outer],clock,epochs=best)
                pred=final.predict(x[it]);selection={'criterion':'INNER_TRAIN_SCALED_VALIDATION_MSE','best_epoch':best,
                                                    'inner_history':history,'outer_refit_history':final.history}
                final.portable()
            require(pred.shape==(len(it),len(targets)) and np.isfinite(pred).all(),'Bad heldout predictions')
            checkpoint=root/'model.pkl';atomic_bytes(checkpoint,pickle.dumps(final,protocol=5))
            op=root/'oof.npz';save_npz(op,prediction=pred,patient_id=m.iloc[it].patient_id.to_numpy(str),
                cancer=c[it],target_id=targets)
            elapsed=time.monotonic()-tstart
            rec={'status':'FOLD_FIT_REFIT_AND_OOF_COMPLETE','signature':signature,'method':method,'fold':fold,'kind':kind,
                 'scope':scope,'checkpoint':record(checkpoint),'oof_prediction':record(op),'dataset_signature':digest(cfg),
                 'target_digest':digest(targets.tolist()),'input_dim':x.shape[1],
                 'fit_patient_ids':m.iloc[outer].patient_id.tolist(),'selection_patient_ids':m.iloc[iv].patient_id.tolist(),
                 'test_patient_ids':m.iloc[it].patient_id.tolist(),'test_fold':fold,'selection':selection,
                 'actual_fit_scope':'OUTER_TRAIN_AFTER_INNER_SELECTION_REFIT',
                 'elapsed_wall_seconds':elapsed,'device':device,'gpu_hours_upper':elapsed/3600 if device.startswith('cuda') else 0.,
                 'no_CPTAC_in_training_or_selection':True,'P07_selection_history_not_rewritten':True}
            write_json(recpath,rec)
    return out


def export_oof(model_dir,expected_meta_path,out_path):
    from .data import metadata
    meta=metadata(expected_meta_path,'TCGA');pieces=[];folds=[]
    for fold in range(5):
        rec=read_json(Path(model_dir)/f'fold_{fold}'/'PRIVATE_FOLD_RECEIPT.json')
        p=load_prediction(verify_record(rec['oof_prediction']))
        disjoint(rec['fit_patient_ids'],p['patient_id'])
        expected=meta[meta.outer_fold==fold].patient_id.to_numpy(str);same(p['patient_id'],expected,'heldout fold roster')
        pieces.append(p);folds.append(rec)
    for piece in pieces: same(pieces[0]['target_id'],piece['target_id'])
    patient=np.concatenate([p['patient_id'] for p in pieces]);unique(patient)
    lookup={v:i for i,v in enumerate(patient)};order=[lookup[v] for v in meta.patient_id]
    require(set(lookup)==set(meta.patient_id),'OOF coverage incomplete')
    pred=np.concatenate([p['prediction'] for p in pieces])[order]
    save_npz(out_path,prediction=pred,patient_id=meta.patient_id.to_numpy(str),cancer=meta.cancer.to_numpy(str),target_id=pieces[0]['target_id'])
    write_json(str(out_path)+'.freeze.json',{'status':'FROZEN_OOF_BEFORE_EVALUATION','artifact':record(out_path),
        'patient_n':len(meta),'members':[r['oof_prediction'] for r in folds],'target_digest':digest(pieces[0]['target_id'].tolist())})


def predict_external(model_dir,features_path,release_path,out_dir,device='cpu'):
    release=check_release(release_path,'infer_reference');cfg,m,x,_,_=load_dataset(features_path,'EXTERNAL_FEATURES')
    if release.get('production',True):
        protocol=read_json(Path(__file__).resolve().parents[1]/'config'/'M1_PROTOCOL.json')
        require(len(m)==1128 and x.shape[1]==1536,'Production CPTAC scope changed')
        require(m.cancer.value_counts().to_dict()==protocol['cancer_counts'],'CPTAC cancer composition changed')
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);members=[];route=None;first_signature=None;first_targets=None
    for fold in range(5):
        root=Path(model_dir)/f'fold_{fold}';rec=read_json(root/'PRIVATE_FOLD_RECEIPT.json')
        require(rec['fold']==fold,'Wrong fold member')
        if first_signature is None:first_signature=rec['signature'];route=rec['kind']
        require(rec['signature']==first_signature and rec['kind']==route,'Mixed methods/data scopes in ensemble')
        disjoint(rec['fit_patient_ids'],m.patient_id)
        ck=verify_record(rec['checkpoint']);target=load_prediction(verify_record(rec['oof_prediction']))['target_id']
        if first_targets is None:first_targets=target
        same(first_targets,target,'fivefold target order')
        op=out/f'fold_{fold}.npz';receipt=out/f'fold_{fold}.freeze.json'
        signature=digest({'checkpoint':rec['checkpoint']['sha256'],'input':cfg,'fold':fold,'code':source_digest()})
        if receipt.exists():
            rr=read_json(receipt);require(rr['signature']==signature,'Existing external unit differs')
            p=load_prediction(verify_record(rr['artifact']));same(p['patient_id'],m.patient_id);same(p['cancer'],m.cancer);same(p['target_id'],target);members.append(p['prediction']);continue
        require(not op.exists(),'Unregistered output; reconcile before running')
        # Only this project-created, hash-verified pickle is accepted; never arbitrary downloaded pickle.
        with ck.open('rb') as f: model=pickle.load(f)
        if isinstance(model,MLP):model.device=device;model.net=model.net.to(device);pred=model.predict(x)
        else:pred=model.predict(x,m.cancer.to_numpy(str))
        require(np.isfinite(pred).all(),'External inference failure')
        save_npz(op,prediction=pred,patient_id=m.patient_id.to_numpy(str),cancer=m.cancer.to_numpy(str),target_id=target)
        write_json(receipt,{'status':'FROZEN_EXTERNAL_NO_TRUTH','signature':signature,'fold':fold,'artifact':record(op)})
        members.append(pred)
    mean=sum(np.asarray(v,dtype=np.float64) for v in members)/5.
    final=out/'ensemble_raw.npz'
    if not final.exists():
        save_npz(final,prediction=mean,patient_id=m.patient_id.to_numpy(str),cancer=m.cancer.to_numpy(str),target_id=target)
        write_json(str(final)+'.freeze.json',{'status':'FIVE_FOLD_EQUAL_MEAN_BEFORE_PHI' if route=='TCGA_G' else 'FIVE_FOLD_FUNCTION_MEAN',
            'artifact':record(final),'members':[record(out/f'fold_{i}.npz') for i in range(5)],'no_truth_input':True})
    else:
        old=load_prediction(final);require(np.array_equal(old['prediction'],mean),'Unregistered ensemble mismatch')
        require(Path(str(final)+'.freeze.json').exists(),'Ensemble lacks freeze; reconcile existing result, do not repeat training')
        verify_record(read_json(str(final)+'.freeze.json')['artifact'])
    return final


def score_gene_prediction(pred_path,binding_path,out_path):
    """Input is OOF gene prediction or ALREADY AVERAGED external gene predictions."""
    freeze=read_json(str(pred_path)+'.freeze.json');verify_record(freeze['artifact'])
    require(sha(pred_path)==freeze['artifact']['sha256'],'Prediction freeze does not bind supplied file')
    p=load_prediction(pred_path);scorer=FT01Scorer(binding_path)
    gene,neg=decode_gene_ensemble([p['prediction']]);f=scorer.score(gene,p['patient_id'],p['target_id'])
    save_npz(out_path,prediction=f,patient_id=p['patient_id'],cancer=p['cancer'],target_id=scorer.targets)
    write_json(str(out_path)+'.freeze.json',{'status':'FT01_FUNCTION_FROM_FROZEN_GENE_PREDICTION',
        'artifact':record(out_path),'upstream_prediction':record(pred_path),'phi_binding':record(binding_path),
        'negative_gene_fraction_before_projection':neg,'gene_projection':'max(mean_decoded_log2_TPM_plus1,0)',
        'ordinary':'FT01_RAW_ES','PROGENy':'SIGNED_WEIGHTED_MEAN','no_external_fit':True})


def evaluate(pred_path,truth_path,out_dir,release_path):
    r=check_release(release_path,'evaluate_frozen');p=load_prediction(pred_path)
    freeze=read_json(str(pred_path)+'.freeze.json');verify_record(freeze['artifact'])
    require(sha(pred_path)==freeze['artifact']['sha256'],'Prediction changed after freeze')
    with np.load(truth_path,allow_pickle=False) as z:
        require(set(z.files)=={'patient_id','cancer','target_id','truth'},'Explicit truth schema required')
        t={k:z[k] for k in z.files}
    for key in ['patient_id','cancer','target_id']:same(p[key],t[key],key)
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    cells=table(t['truth'],p['prediction'],p['cancer'],p['target_id'],r['statistics']['min_n']);s=summarize(cells)
    atomic_bytes(out/'ALL_CELLS.csv',cells.to_csv(index=False).encode())
    write_json(out/'POINT_ESTIMATES.json',dict(status='POINT_ESTIMATES_COMPLETE',**s))
    write_json(out/'EVALUATION_RECEIPT.json',{'status':'POINT_ESTIMATES_COMPLETE','prediction':record(pred_path),
        'truth':record(truth_path),'n_patients':len(p['patient_id']),'n_targets':len(p['target_id']),
        'historical_external_access':'ALREADY_USED_M0_NO_VIRGIN_CLAIM','no_calibration':True,
        'point_estimates_before_CI':True})
    return s


def compare(pred1_path,pred0_path,truth_path,out_path,release_path):
    r=check_release(release_path,'evaluate_frozen');a=load_prediction(pred1_path);b=load_prediction(pred0_path)
    for key in ['patient_id','cancer','target_id']:same(a[key],b[key],key)
    for path in [pred1_path,pred0_path]:
        frozen=read_json(str(path)+'.freeze.json')['artifact'];verify_record(frozen)
        require(sha(path)==frozen['sha256'],'Comparison prediction freeze mismatch')
    with np.load(truth_path,allow_pickle=False) as z:
        for key in ['patient_id','cancer','target_id']:same(a[key],z[key],key)
        truth=z['truth']
    st=r['statistics'];result=paired_bootstrap(truth,a['prediction'],b['prediction'],a['cancer'],st['B'],st['seed'],st['min_n'])
    write_json(out_path,dict(comparison={'first':record(pred1_path),'reference':record(pred0_path)},**result));return result
