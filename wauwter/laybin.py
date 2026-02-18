#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 14:37:40 2023

@author: WauWter
"""
#%% Initialize

import numpy as np
from .wauwternifti import readnii,savenii
import sys

#%% Read layfile

layfile=sys.argv[1]
nrlayz=sys.argv[2]
# sub='sub-014'
# rdir='/project/3017081.01/bids7T/'+sub+'/'
# ddir=rdir+'derivatives/'
# anatdir=ddir+'pipe/anat/'

# layfile=anatdir+'layers_equidist-0.5mm.nii.gz'

lay,hlay=readnii(layfile)

lay[lay==1]+=1
lay[lay==20]-=1
lay[lay>0]-=1
nlay=int(nrlayz)
laycorr=np.max(lay)/nlay
lay=np.ceil(lay/laycorr).astype(np.int16)

#%% make separate layer files
fx=lay.shape[0]
fy=lay.shape[1]
fz=lay.shape[2]
nvox=fx*fy*fz
binarylayers=np.zeros((nvox,nlay,dtype=np.int16)
lay=np.reshape(lay,nvox)

for i in range(nlay):
    binarylayers[lay==i+1,i]=1
binarylayers=np.reshape(binarylayers,(fx,fy,fz,nlay),dtype=np.int16)

hlay['datatype']=4
hlay['bitpix']=16
hlay['dim']=(4,)+hlay['dim'][1:4]+(nlay,)+hdr['dim'][5:]

layfilebase,layfilesuffix=layfile.split('.nii.gz')
savenii(binarylayers,hlay,layfilebase+'_'+str(nlay)+'depth.nii.gz')
