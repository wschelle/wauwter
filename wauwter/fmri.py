#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 14:00:08 2026

@author: WauWter
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.linalg import toeplitz
from tqdm import tqdm
 
def cosfilt(nf,time):
    fm=np.zeros([nf,time])
    for i in range(nf):
        fm[i,:]=np.cos(np.arange(time)*(i+1)*math.pi/time)
    return(fm)

def dispmat(matr):
    msize=matr.shape
    fig, axes = plt.subplots(msize[0], 1, figsize=(msize[0],msize[0]*2))
    for i in range(msize[0]):
        axes[i].plot(matr[i,:])
    plt.show()

def loadmp(mpfile,csv=None):
    if csv:
        mp=np.genfromtxt(mpfile,delimiter=',')
    else:
        mp=np.loadtxt(mpfile)
        
    mp=mp.T
    return(mp)

def hpfilt(dat,tr,cut,addfilt=0,mask=0,convperc=False,showfiltmat=True):
    #reg = LinearRegression()
    datsize=dat.shape
    nfac=datsize[0]
    ntime=datsize[1]
    dat2=np.zeros([int(nfac),int(ntime)],dtype=np.float32)
    nfilt=int(np.floor(2*(ntime*(tr/1.)*cut)))
    fm=cosfilt(nfilt,int(ntime))
    yhat=np.zeros(dat.shape,dtype=np.float32)
    if np.sum(addfilt) != 0:
        fm=np.concatenate([fm,addfilt])
        nfilt+=addfilt.shape[0]
    if np.sum(mask) == 0:
        mask=np.ones(nfac)
    for i in range(nfilt):
        fm-=np.mean(fm[i,:])
    fm=np.vstack((fm,np.ones(ntime)))
    nfilt+=1
    if showfiltmat:
        dispmat(fm)
    beta=np.zeros([nfac,nfilt],dtype=np.float32)
    fm=fm.T
    for i in tqdm(range(nfac)):
        if mask[i] != 0:
            #reg.fit(fm.T, dat[i,:])
            beta[i,:] = np.linalg.inv(fm.T @ fm) @ fm.T @ dat[i,:]
            yhat[i,:] = fm @ beta[i,:]
            if convperc:
                dat2[i,:]=(dat[i,:]-yhat[i,:])/beta[i,-1]*100
            else:
                dat2[i,:]=dat[i,:]-yhat[i,:]
    return(dat2)

def mreg(dm,dat,mask):
    dmsize=dm.shape
    datsize=dat.shape
    betas=np.zeros([datsize[0],dmsize[0]+1],dtype=np.float32)
    msres=np.zeros([datsize[0]],dtype=np.float32)
    reg = LinearRegression()
    for i in range(dmsize[0]):
        dm[i,:]-=np.mean(dm[i,:])
    for i in tqdm(range(datsize[0])):
        if mask[i] != 0:
            reg.fit(dm.T,dat[i,:])
            betas[i,0:dmsize[0]]=reg.coef_
            betas[i,dmsize[0]]=reg.intercept_
            msres[i]=(np.sum((dat[i,:]-reg.predict(dm.T))**2))/(datsize[1]-dmsize[0]-1)
    return(betas,msres)

def GLM(X, Y, mask, norm_X=True, add_const=True, beta_only=False, betaresid_only=True):
    if norm_X:
        for i in range(X.shape[0]):
            X-=np.mean(X[i,:])
    if add_const:
        X=np.vstack((X,np.ones(X.shape[1])))
        
    beta=np.zeros([Y.shape[0],X.shape[0]],dtype=np.float32)
    yhat=np.zeros(Y.shape,dtype=np.float32)
    
    X=X.T #because my X is normally of shape [factor,time], which is transposed here

    for i in tqdm(range(Y.shape[0])):
        if mask[i] != 0:
            beta[i,:] = np.linalg.inv(X.T @ X) @ X.T @ Y[i,:]
            yhat[i,:] = X @ beta[i,:]
        
    residuals=Y-yhat
    msres=np.sum(residuals**2,axis=1)/(X.shape[0]-X.shape[1])
    
    if beta_only:
        return beta
    elif betaresid_only:
        return beta,msres
    else:
        return beta,msres,yhat
    
def GLS(X, Y, mask, yhat_0, norm_X=True, add_const=True, beta_only=False, betaresid_only=True):
    if norm_X:
        for i in range(X.shape[0]):
            X-=np.mean(X[i,:])
            
    if add_const:
        X=np.vstack((X,np.ones(X.shape[1])))
    
    beta=np.zeros([Y.shape[0],X.shape[0]],dtype=np.float32)
    yhat=np.zeros(Y.shape,dtype=np.float32)
    phi=np.zeros(Y.shape[0],dtype=np.float32)
    
    X=X.T
    
    tpl=toeplitz(np.arange(Y.shape[1]))
    tpl=tpl.astype(np.int32)
    
    for i in tqdm(range(Y.shape[0])):
        if mask[i] != 0:
            res0 = Y[i,:] - yhat[i,:]
            res1 = np.roll(res0,1)
            phi[i] = (res0 - res0.mean()) @ (res1 - res1.mean()) / np.sqrt(np.sum((res0 - res0.mean()) ** 2) * np.sum((res1 - res1.mean()) ** 2))
    
    for i in tqdm(range(Y.shape[0])):
        if mask[i] != 0:
            V = phi[i] ** tpl
            V = np.linalg.inv(V)
            beta[i,:] = np.linalg.inv(X.T @ V @ X) @ X.T @ V @ Y[i,:]
            yhat[i,:] = X @ beta[i,:]
    
    residuals = Y - yhat
    msres=np.sum(residuals**2,axis=1)/(X.shape[0]-X.shape[1])
    
    if beta_only:
        return beta
    elif betaresid_only:
        return beta,msres
    else:
        return beta,msres,yhat

def tcon(contrast,dm,betas,msres,mask):
    bsize=betas.shape
    dmsize=dm.shape
    if len(dmsize)==2:
        ntime=dmsize[1]
        nfac=dmsize[0]
    else:
        ntime=dmsize[0]
        nfac=1
    c=np.zeros([nfac+1,1])
    c[:,0]=contrast
    tval=np.zeros(bsize[0])
    t1=np.zeros([nfac+1,1])
    #term2=transpose(c)#invert(x#transpose(x))#c
    t2=c[0:nfac,0].T @ np.linalg.inv(dm @ dm.T) @ c[0:nfac,0]
    for i in tqdm(range(bsize[0])):
        if mask[i] != 0:
            t1[:,0]=betas[i,:]
            tval[i]=c.T @ t1 / (np.sqrt(msres[i]*t2))
    return(tval)
    
def make_fir(fironset,firlength,maxtime):
    firmatrix=np.zeros([int(np.max(fironset[:,0])+1)*firlength,maxtime])
    for i in range(len(fironset[:,0])):
        for j in range(firlength):
            if fironset[i,1]+j < maxtime:
                firmatrix[fironset[i,0]*firlength+j,fironset[i,1]+j]=1
    firmatrix=firmatrix[np.sum(firmatrix,axis=1)>1,:]
    #dispmat(firmat4[0::4,:])
    return firmatrix

def calc_fir(firmatrix,datamatrix,mask):
    b=GLM(firmatrix,datamatrix,mask,norm_X=False,beta_only=True)
    return b[:,:-1]