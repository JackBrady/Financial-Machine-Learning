
# Snippets from Advances in Financial Machine Learning by Dr. Marcos Lopez de Prado

def clfHyperFit(feat,lbl,times,classifier,param_grid,weights,cv=10,bagging=[0,None,1.],
                rndSearchIter=0,n_jobs=-1,pctEmbargo=0.01):
    if set(lbl.values)=={0,1}:scoring='f1' # f1 for meta-labeling
    else:scoring='accuracy' # symmetric towards all cases
    #1) hyperparameter search, on train data
    inner_cv=PurgedKFold(n_splits=cv,t1=times,pctEmbargo=pctEmbargo) # purged
    if rndSearchIter==0:
        gs=GridSearchCV(estimator=classifier,param_grid=param_grid,
            scoring=scoring,cv=inner_cv,n_jobs=n_jobs,iid=False)
    else:

        gs=RandomizedSearchCV(estimator=classifier,param_distributions= \
            param_grid,scoring=scoring,cv=inner_cv,n_jobs=n_jobs,
            iid=False,n_iter=rndSearchIter)
    gs=gs.fit(feat,lbl,weights)

    return gs



import numpy as np,pandas as pd,matplotlib.pyplot as mpl
from scipy.stats import rv_continuous,kstest

class logUniform_gen(rv_continuous):

    def _cdf(self,x):
        return np.log(x/self.a)/np.log(self.b/self.a)
def logUniform(a=1,b=np.exp(1)):return logUniform_gen(a=a,b=b,name='logUniform')



def clfHyperFitnn(feat,lbl,t1,clf,param_grid,sample_weights,cv=10,
                rndSearchIter=0,n_jobs=-1,pctEmbargo=0.01):
    if set(lbl.values)=={0,1}:scoring='f1' # f1 for meta-labeling
    else:scoring='accuracy' # symmetric towards all cases
    #1) hyperparameter search, on train data
    inner_cv=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged
    if rndSearchIter==0:
        gs=GridSearchCV(estimator=clf,param_grid=param_grid,
            scoring=scoring,cv=inner_cv,n_jobs=n_jobs,iid=False)
    else:

        gs=RandomizedSearchCV(estimator=clf,param_distributions= \
            param_grid,scoring=scoring,cv=inner_cv,n_jobs=n_jobs,
            iid=False,n_iter=rndSearchIter)
    gs=gs.fit(feat,lbl,sample_weights) 

    return gs
