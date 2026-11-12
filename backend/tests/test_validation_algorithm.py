"""逐批留出检验：防数据泄漏、失败折说明原因、阈值/未来m不变性、旧字段兼容。"""
import numpy as np
import pytest
from scipy.stats import betabinom
from src.services import validation_algorithm as va
from src.services.analysis_service import analyze_germination
from src.services.eb_algorithm import fit_hyperparameters

def make_counts(b=12,seed=7):
    rng=np.random.default_rng(seed);n=rng.integers(20,181,size=b).tolist();p=rng.beta(8,4,size=b)
    return n,[int(rng.binomial(int(ni),float(pi))) for ni,pi in zip(n,p)]

def codes(b):return [f'T-{i+1:02d}' for i in range(b)]

def make_snapshot(n,k):
    b=len(n)
    lots=[{'id':i+1,'study_id':1,'code':codes(b)[i],'name':f'批次{i+1}','origin':'合成'} for i in range(b)]
    counts=[{'id':i+1,'lot_id':i+1,'n':int(n[i]),'k':int(k[i]),'note':''} for i in range(b)]
    return {'study':{'id':1,'code':'T-01','name':'留出检验合成研究','description':'','source':'pytest'},'lots':lots,'counts':counts}

def test_fold_priors_match_refit_without_held_out_batch():
    n,k=make_counts();v=va.leave_one_batch_out(n,k,codes(len(n)))
    assert all(f['status']=='ok' for f in v['folds'])
    for j,fold in enumerate(v['folds']):
        refit=fit_hyperparameters([x for i,x in enumerate(n) if i!=j],[x for i,x in enumerate(k) if i!=j])
        assert fold['alpha']==refit['alpha'] and fold['beta']==refit['beta']
        assert fold['train_batches']==len(n)-1

def test_held_out_count_never_enters_own_fold():
    n,k=make_counts();j=3;k2=list(k);k2[j]=0 if k[j]>0 else 1
    v1=va.leave_one_batch_out(n,k,codes(len(n)));v2=va.leave_one_batch_out(n,k2,codes(len(n)))
    f1,f2=v1['folds'][j],v2['folds'][j]
    assert (f1['alpha'],f1['beta'],f1['lower'],f1['upper'])==(f2['alpha'],f2['beta'],f2['lower'],f2['upper'])
    assert f1['observed_log_probability']!=f2['observed_log_probability']
    assert any(v1['folds'][i]['alpha']!=v2['folds'][i]['alpha'] for i in range(len(n)) if i!=j)

def test_predictive_interval_and_log_probability_match_betabinom():
    n,k=make_counts();v=va.leave_one_batch_out(n,k,codes(len(n)))
    for f in v['folds']:
        a,b,m,obs=f['alpha'],f['beta'],f['n'],f['k']
        assert f['lower']==int(betabinom.ppf(.025,m,a,b)) and f['upper']==int(betabinom.ppf(.975,m,a,b))
        assert 0<=f['lower']<=f['upper']<=m
        assert f['observed_log_probability']==pytest.approx(float(betabinom.logpmf(obs,m,a,b)),abs=1e-9)
        assert f['covered']==(f['lower']<=obs<=f['upper'])
        assert f['mean']==pytest.approx(m*a/(a+b))

def test_summary_scores_all_folds_when_complete():
    n,k=make_counts();v=va.leave_one_batch_out(n,k,codes(len(n)));s=v['summary']
    assert s['total_folds']==len(n)==s['scored_folds'] and s['failed_folds']==0 and s['complete'] is True
    assert s['mean_log_probability']==pytest.approx(float(np.mean([f['observed_log_probability'] for f in v['folds']])))
    assert s['coverage']==pytest.approx(float(np.mean([f['covered'] for f in v['folds']])))
    assert '完整逐批留出评分' in s['score_note']

def test_eight_batch_study_folds_all_fail_with_reason():
    n,k=make_counts(b=8,seed=11);v=va.leave_one_batch_out(n,k,codes(8));s=v['summary']
    assert s['scored_folds']==0 and s['failed_folds']==8 and s['complete'] is False
    assert s['mean_log_probability'] is None and s['coverage'] is None
    assert '不是完整评分' in s['score_note']
    for f in v['folds']:
        assert f['status']=='failed' and f['reason_kind']=='insufficient_batches'
        assert '8–200' in f['reason'] and 'alpha' not in f and 'observed_log_probability' not in f

def test_fold_fails_when_remaining_counts_all_zero():
    v=va.leave_one_batch_out([50]*9,[5]+[0]*8,codes(9));f0=v['folds'][0];s=v['summary']
    assert f0['status']=='failed' and f0['reason_kind']=='degenerate_counts' and '全部0或全部1' in f0['reason']
    assert s['failed_folds']==1 and s['scored_folds']==8 and s['complete'] is False
    assert s['mean_log_probability'] is not None and '不是完整评分' in s['score_note']

def test_fold_fails_when_remaining_counts_all_one():
    v=va.leave_one_batch_out([50]*9,[5]+[50]*8,codes(9));f0=v['folds'][0];s=v['summary']
    assert f0['status']=='failed' and f0['reason_kind']=='degenerate_counts' and '全部0或全部1' in f0['reason']
    assert s['failed_folds']==1 and s['scored_folds']==8 and s['complete'] is False

def test_fit_failure_folds_carry_reason_and_block_complete_score(monkeypatch):
    n,k=make_counts()
    def boom(x,y):raise ValueError("所有起点未达到收敛与投影梯度要求，拒绝拟合")
    monkeypatch.setattr(va,'fit_hyperparameters',boom)
    v=va.leave_one_batch_out(n,k,codes(len(n)));s=v['summary']
    assert all(f['status']=='failed' and f['reason_kind']=='fit_failure' and '拒绝拟合' in f['reason'] for f in v['folds'])
    assert s['scored_folds']==0 and s['complete'] is False and s['mean_log_probability'] is None

def test_validation_invariant_to_threshold_and_future_n():
    snap=make_snapshot(*make_counts())
    r1=analyze_germination(snap,threshold=.5,future_n=10);r2=analyze_germination(snap,threshold=.9,future_n=200)
    assert r1['validation']==r2['validation'] and r1['validation_hash']==r2['validation_hash']
    assert r1['fit_hash']==r2['fit_hash'] and r1['input_hash']!=r2['input_hash']

def test_result_keeps_legacy_fields_and_conditional_posterior():
    r=analyze_germination(make_snapshot(*make_counts()))
    for key in ('algorithm_version','study','fit','lots','threshold','future_n','input_snapshot','fit_hash','input_hash','assumptions','interpretation'):assert key in r
    for row in r['lots']:
        for key in ('post_alpha','post_beta','raw_rate','mean','lower','upper','prob_gt_threshold','data_weight','prior_weight','cdf','prediction','conditional_note'):assert key in row
    assert r['validation']['summary']['total_folds']==len(r['lots'])
    legacy={k:v for k,v in r.items() if k not in ('validation','validation_hash')}
    assert legacy['fit']==r['fit'] and legacy['lots']==r['lots'] and legacy['input_hash']==r['input_hash']

def test_analyze_marks_incomplete_when_training_batches_insufficient():
    n,k=make_counts(b=8,seed=11);r=analyze_germination(make_snapshot(n,k));s=r['validation']['summary']
    assert s['complete'] is False and s['scored_folds']==0 and s['failed_folds']==8
