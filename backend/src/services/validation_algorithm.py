"""逐批留出(leave-one-batch-out)预测检验：每折只用其余批次拟合先验，预测留出批次的原试种萌发数。"""
import numpy as np
from scipy.special import betaln,gammaln,logsumexp
from .eb_algorithm import fit_hyperparameters,validate_counts
VALIDATION_VERSION='leave-one-batch-out-betabinomial-predictive-v1'
INTERVAL_LEVEL=.95
MIN_TRAIN_BATCHES=8

def beta_binomial_logpmf(k,n,a,b):
    return float(gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)+betaln(k+a,n-k+b)-betaln(a,b))

def predictive_count_interval(n,a,b,level=INTERVAL_LEVEL):
    tail=(1-level)/2;j=np.arange(n+1)
    logp=gammaln(n+1)-gammaln(j+1)-gammaln(n-j+1)+betaln(j+a,n-j+b)-betaln(a,b)
    cdf=np.cumsum(np.exp(logp-float(logsumexp(logp))))
    return min(int(np.searchsorted(cdf,tail,side='left')),n),min(int(np.searchsorted(cdf,1-tail,side='left')),n)

def failure_kind(train_n,train_k):
    if len(train_n)<MIN_TRAIN_BATCHES:return 'insufficient_batches'
    if int(np.sum(train_k))==0 or int(np.sum(train_k))==int(np.sum(train_n)):return 'degenerate_counts'
    return 'fit_failure'

def leave_one_batch_out(n,k,codes=None):
    N,K=validate_counts(n,k);B=len(N)
    if codes is None:codes=[f'#{i+1}' for i in range(B)]
    if len(codes)!=B:raise ValueError("批次编号数须与批数一致")
    folds=[]
    for j in range(B):
        mask=np.arange(B)!=j;tn,tk=N[mask],K[mask]
        fold={'index':j,'lot_code':str(codes[j]),'n':int(N[j]),'k':int(K[j]),'train_batches':B-1}
        try:fit=fit_hyperparameters(tn,tk)
        except ValueError as e:
            folds.append({**fold,'status':'failed','reason':str(e),'reason_kind':failure_kind(tn,tk)});continue
        a,b=fit['alpha'],fit['beta'];m=int(N[j]);mu=a/(a+b);lower,upper=predictive_count_interval(m,a,b)
        folds.append({**fold,'status':'ok','reason':'','reason_kind':None,'alpha':a,'beta':b,'fit_loglik':fit['loglik'],'weak_identification':fit['weak_identification'],'mean':float(m*mu),'variance':float(m*mu*(1-mu)*(a+b+m)/(a+b+1)),'lower':lower,'upper':upper,'level':INTERVAL_LEVEL,'observed_log_probability':beta_binomial_logpmf(int(K[j]),m,a,b),'covered':bool(lower<=int(K[j])<=upper)})
    scored=[f for f in folds if f['status']=='ok'];failed=[f for f in folds if f['status']=='failed'];complete=not failed
    mean_lp=float(np.mean([f['observed_log_probability'] for f in scored])) if scored else None
    coverage=float(np.mean([f['covered'] for f in scored])) if scored else None
    if complete:score_note=f'全部{B}折留一拟合成功，平均对数概率与覆盖率覆盖所有批次，是完整逐批留出评分。'
    else:score_note=f'{len(failed)}折因剩余训练批数不足、剩余计数全零全一或拟合失败未评分（原因见各折）；汇总只覆盖{len(scored)}个成功折，不是完整评分，不能与完整评分直接比较。'
    return {'method':VALIDATION_VERSION,'level':INTERVAL_LEVEL,'folds':folds,'summary':{'total_folds':B,'scored_folds':len(scored),'failed_folds':len(failed),'complete':complete,'mean_log_probability':mean_lp,'coverage':coverage,'level':INTERVAL_LEVEL,'score_note':score_note},'leakage_note':'每折先验只用其余批次按同一有界多起点流程重新拟合；留出批次的k不进入该折训练，仅用于对数概率评分与覆盖判定。','invariant_note':'留一检验只使用各批n/k与固定拟合流程；研究阈值与同批未来试种数不参与，改变它们检验结果不变。'}
