#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:44:41 2026

@author: WauWter
"""
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def multiple_correlation(X,y):
    # X = np array [nfactors,ntime]
    # y = np array [ntime]
    c=np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        c[i]=np.corrcoef(X[i,:],y)[0][1]
    R = np.corrcoef(X)
    Rsq = c.T @ np.linalg.inv(R) @ c
    return (Rsq)

def pearson_cc(x,y):
    numerator=np.sum((x-np.mean(x))*(y-np.mean(y)))
    denominator=np.sqrt(np.sum((x-np.mean(x))**2) * np.sum((y-np.mean(y))**2))
    r=numerator/denominator
    return r

def linreg(x,y):
    X = np.asarray([np.ones(len(x)), x]).T
    beta_0, beta_1 = np.linalg.inv(X.T @ X) @ X.T @ y
    return np.array([beta_0, beta_1])
    
def quick_pca(data,k=2):
    data -= data.mean(axis=0) # data needs to be in shape [n_observations,n_features]
    U, S, Vt = np.linalg.svd(data, full_matrices=False)
    pc = U[:, :k] * S[:k]
    return pc

def convert_t_to_z(t_data, dof):
    # Convert t-values to p-values, then to Z-scores
    z_scores = stats.norm.ppf(stats.t.cdf(t_data, dof))
    z_scores[np.isnan(z_scores)] = 0
    z_scores[np.isinf(z_scores)] = 0
    return z_scores

def cosine_dist(x,y):
    cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return 1-cos_sim

def rval2tval(R,N):
    numerator = R * np.sqrt(N - 2)
    denominator = np.sqrt(1 - R**2)
    t = numerator / denominator
    return(t)

def mult_R2(designmatrix):
    mcc=np.zeros(designmatrix.shape[0],dtype=np.float32)
    clist=np.arange(designmatrix.shape[0],dtype=np.int16)
    reg = LinearRegression()
    for i in range(designmatrix.shape[0]):
        reg.fit(designmatrix[clist[clist != i],:].T, designmatrix[i,:])
        tmp = reg.predict(designmatrix[clist[clist != i],:].T)
        mcc[i] = r2_score(designmatrix[i,:],tmp)
    return mcc

def multiple_comparison(statistic,dfn=None,dfd=None,alpha=0.05,method='fdr_bh',test='t',tail=2,cutoff_only=True):
    stat0=statistic[statistic!=0]
    if test=='t':
        uncorr_pval = stats.t.sf(np.abs(stat0), dfn-1)*tail
    if test=='f':
        uncorr_pval = 1-stats.f.cdf(stat0, dfn, dfd)
    corr_pval=multipletests(uncorr_pval,alpha,method=method)[1]
    if ((cutoff_only) & (np.min(corr_pval[stat0>0])<alpha)):
        return np.min(stat0[(corr_pval<alpha) & (stat0>0)])
    else:
        return uncorr_pval,corr_pval

def goodness_of_fit_F(modelfit,data,mask,dof,chisq=None):
    #modelfit & data should be of the same size (datapoints,time)
    #mask is of shape datapoints
    #dof is 2-element list: [model DoF,ErrorDoF]
    #optional: provide chisq values of size datapoints
    #returns array size datapoints with F-statistics
    fval=np.zeros(mask.shape)
    msm=np.zeros(mask.shape)
    for i in tqdm(range(mask.shape[0])):
        if mask[i]!=0:
            msm[i]=np.sum((np.mean(data[i,:])-modelfit[i,:])**2)/dof[0]
    if chisq==None:
        chisq=np.sum((data-modelfit)**2,axis=1)
    mse=chisq/dof[1]
    fval[mse!=0]=msm[mse!=0]/mse[mse!=0]
    return(fval)