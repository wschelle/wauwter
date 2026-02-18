#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:44:35 2023

@author: WauWter
"""
import numpy as np
import os
import sys
from .wauwternifti import readnii, savenii
from .wauwterfmri import loadmp, hpfilt
from copy import deepcopy

## Getting parameters from input line
fdir=sys.argv[1]
suff0=sys.argv[2]
TR=float(sys.argv[3])
cutoff=float(sys.argv[4])

## Getting filenames
files = [f for f in os.listdir(fdir) if os.path.isfile(os.path.join(fdir, f))]
filenames = [s for s in files if suff0 in s]
ntask=len(filenames)

## Loading the niftis
nt=np.zeros(ntask,dtype=np.int16)
hdrs={}
for i in range(ntask):
    nii,hdr=readnii(fdir+filenames[i])
    if i==0:
        sm=deepcopy(nii)
        fx=nii.shape[0]
        fy=nii.shape[1]
        fz=nii.shape[2]
    else:
        sm=np.concatenate((sm,nii),axis=3)
    nt[i]=nii.shape[3]
    hdrs[str(i)]=hdr
    del nii,hdr

## Getting length per run
nv=fx*fy*fz
ntt=np.sum(nt)
nti=np.zeros(ntask+1,dtype=np.int32)
for i in range(ntask+1):
    nti[i]=np.sum(nt[:i])

## Reshaping data matrix to 2D
sm=np.reshape(sm,[nv,ntt])

## Making a mask
mask=np.zeros([nv,ntask],dtype=np.int16)
for i in range(ntask):
    mask[np.mean(sm[:,nti[i]:nti[i+1]],axis=1)>50,i]=1
m2=np.prod(mask,axis=1)

## Getting corresponding filebase name
filebase=[None]*ntask
for i in range(ntask):
    filebase[i]=filenames[i].split(suff0)[0]

## factor for different runs
mp0=np.zeros([ntask-1,ntt],dtype=np.float32)
for i in range(ntask-1):
    mp0[i,nti[i]:nti[i+1]]=1

## Loading motion parameters
mp=np.zeros([6,ntt],dtype=np.float32)
for i in range(ntask):
    mp[:,nti[i]:nti[i+1]]=loadmp(fdir+filebase[i]+'POCS_NORDIC_MCMOCOparams_mp.csv',csv=1)

## High-pass filtering motion params
mp=hpfilt(mp,TR,cutoff,mp0,0,0,showfiltmat=False)

## Getting framewise displacement
mpmd=np.zeros([4,ntt],dtype=np.float32)
for i in range(ntask):
    tmp=loadmp(fdir+filebase[i]+'POCS_NORDIC_MCMOCOparams_mp_disp.csv',csv=1)
    mpmd[:,nti[i]:nti[i+1]]=tmp[1:,:]

## Adding it all to 1 motion param matrix
mp=np.vstack([mp,mpmd])
mp=np.concatenate([mp,mp0],axis=0)

## High-pass filtering the fmri data
sm2=hpfilt(sm,TR,cutoff,mp,m2,1,showfiltmat=False)

## Write to nifti
for i in range(ntask):
    savenii(np.reshape(sm2[:,nti[i]:nti[i+1]],[fx,fy,fz,nt[i]]),hdrs[str(i)],fdir+filebase[i]+suff0+'_hpfilt.nii.gz')

