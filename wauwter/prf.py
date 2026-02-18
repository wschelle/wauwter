#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 11:00:00 2026

@author: WauWter
"""

import numpy as np
from tqdm import tqdm
from Python.python_scripts.wauwterfmri import GLM
import copy

def prf_est(dm,dat,mask):
    nfac=dm.shape[0]
    nv=dat.shape[0]
    prfc=np.zeros(nv)
    prfs=np.zeros(nv)
    prfa=np.zeros(nv)
    prfi=np.zeros(nv)
    prfr=np.zeros(nv)
    fwhm=2*np.sqrt(2*np.log(2))
    for i in range(nfac):
        dm[i,:]-=np.mean(dm[i,:])
    
    be,re,yh=GLM(dm,dat,mask,betaresid_only=False)
    co=np.squeeze(be[:,-1])
    be=be[:,:-1]
    
    for i in tqdm(range(nv)):
        if mask[i] != 0:
            prfa[i]=np.max(be[i,:])
            prfi[i]=co[i]
            prfc[i]=np.where(be[i,:] == np.max(be[i,:]))[0][0]
            prfs[i]=np.sum(be[i,:] > prfa[i]/2)/fwhm
            prfr[i]=np.corrcoef(dat[i,:],yh[i,:])[0][1]**2
    return prfc,prfs,prfa,prfi,prfr

def lmfit_prf(params, dm, ydata):
    cons = params['cons'].value
    ampl = params['ampl'].value
    center = params['center'].value
    sigma = params['sigma'].value
    
    ymodel=np.zeros(dm.shape)
    for i in range(dm.shape[0]):
        ymodel[i,:]=dm[i,:] * np.exp(-(i-center)**2 / (2*sigma**2))
    ymodel=np.sum(ymodel,axis=0)
    ymodel*=ampl
    ymodel+=cons
    return (ymodel - ydata)

def prf2d(designmatrix,p0,xgrid,ygrid):
    xs=designmatrix.shape
    x2=np.zeros(xs,dtype=np.float32)
    # xdiff = np.arange(xs[0],dtype=np.float32)
    # ydiff = np.arange(xs[1],dtype=np.float32)
    # xdiff, ydiff = np.meshgrid(xdiff, ydiff)
    xdiff=xgrid-p0[2]
    ydiff=ygrid-p0[3]
    xdiff**=2
    ydiff**=2
    xdiff/=(p0[4]**2)
    ydiff/=((p0[4]/p0[5])**2)
    num=xdiff+ydiff
    expo=np.exp(-num/2)
    x2 = np.tensordot(designmatrix,expo, axes=([0,1],[0,1]))
    x2/=np.nanmax(x2)
    x2[np.isnan(x2)]=0
    x2*=p0[1]
    x2+=p0[0]
    return x2

def lmfit_prf2d(params,designmatrix,ydata,xgrid,ygrid):
    p0=np.zeros(6,dtype=np.float32)
    p0[0]=params['cons'].value
    p0[1]=params['amp'].value
    p0[2]=params['centerX'].value
    p0[3]=params['centerY'].value
    p0[4]=params['XYsigma'].value
    p0[5]=params['XYsigmaRatio'].value
    ymodel=prf2d(designmatrix,p0,xgrid,ygrid)
    return(ymodel-ydata)
    
def gaussian1D(xrange,const,amplitude,center,sigma):
    numerator=(xrange-center)**2
    denominator=2*sigma**2
    gauss=amplitude*np.exp(-numerator/denominator)+const
    return gauss

def lmfit_1DGauss(params,xrange,ydata):
    gauss=gaussian1D(xrange,params['cons'],params['ampl'],params['center'],params['sigma'])
    return(gauss-ydata)

def nrprf_est(beta_matrix,nrprf_center=3.5,nrprf_sigma=1,stepsize=100):
    gauss_pos=gaussian1D(np.arange(-nrprf_center,nrprf_center,1/stepsize),0,1,nrprf_center,nrprf_sigma)
    gauss_neg=gaussian1D(np.arange(-nrprf_center,nrprf_center,1/stepsize),0,-1,-nrprf_center,nrprf_sigma)
    gauss=gauss_pos+gauss_neg
    gauss/=np.max(gauss)
    nr_factors=beta_matrix.shape[1]-1
    nr_datapoints=beta_matrix.shape[0]
    ordinal_regress=np.zeros([nr_datapoints,nr_factors])
    nrprf_matrix=np.zeros([nr_datapoints,nr_factors+4])
    for i in tqdm(range(nr_datapoints)):
        if np.std(beta_matrix[i,:])!=0:
            ordinal_regress[i,:]=copy.deepcopy(beta_matrix[i,:-1])
            ordinal_regress[i,:]/=np.max(ordinal_regress[i,:])
            ordinal_regress[i,ordinal_regress[i,:]<gauss[0]]=gauss[0]
            nrprf_matrix[i,0]=nrprf_center
            nrprf_matrix[i,1]=nrprf_sigma
            nrprf_matrix[i,2]=np.max(beta_matrix[i,:-1])
            nrprf_matrix[i,3]=beta_matrix[i,-1]
            for j in range(nr_factors):
                nrprf_matrix[i,j+4]=np.max(np.where(ordinal_regress[i,j] >= gauss))
    nrprf_matrix[:,4:]/=stepsize
    return nrprf_matrix

def nrprf(params,dm):
    ymodelpos=np.zeros(dm.shape)
    ymodelneg=np.zeros(dm.shape)
    for i in range(dm.shape[0]):
        ymodelpos[i,:]=dm[i,:] * np.exp(-(params['p'+str(i)].value - (params['center'].value*2))**2 / (2*params['sigma'].value**2))
        ymodelneg[i,:]=dm[i,:] * np.exp(-(params['p'+str(i)].value - 0)**2 / (2*params['sigma'].value**2))
    ymodelpos=np.sum(ymodelpos,axis=0)    
    ymodelneg=np.sum(ymodelneg,axis=0)
    ymodel=ymodelpos+(-ymodelneg)
    ymodel*=params['ampl'].value
    ymodel+=params['cons'].value
    return ymodel
    
def lmfit_nrprf(params, dm, ydata):
    ymodel=nrprf(params,dm)
    return (ymodel - ydata)

def nrprf_center_sigma(fitparams,paramstart=4,paramstop=8):
    fp=fitparams[:,paramstart:paramstop]
    centercurve=np.max(fitparams[:,0])
    sigmacurve=np.max(fitparams[:,1])
    hwhm=np.sqrt(2*np.log(2))*sigmacurve
    fp2=copy.deepcopy(fp)
    fp2[fp < centercurve*2-hwhm]=0
    fp2/=centercurve*2
    centerpos=np.zeros(fp.shape[0],dtype=np.float32)
    sigmapos=np.zeros(fp.shape[0],dtype=np.float32)
    com=np.zeros(fp.shape[1],dtype=np.float32)
    for i in tqdm(range(fp.shape[0])):
        if np.std(fp2[i,:]) != 0:
            maxfit=np.where(fp2[i,:] == fp2[i,:].max())[0]
            if len(maxfit) == 1:
                centerpos[i] = maxfit + 1
            else:
                com*=0
                for j in range(fp.shape[1]):
                    com[j]+= (j+1)*fp2[i,j]
                    centerpos[i] = np.sum(com) / np.sum(fp2[i,:])
            sigmapos[i] = np.sum(fp2[i,:])*hwhm
    return centerpos,sigmapos

