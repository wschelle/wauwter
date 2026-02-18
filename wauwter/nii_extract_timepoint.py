#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:29:31 2024

@author: WauWter
"""
#%% Initialize
import numpy as np
from .wauwternifti import readnii,savenii
import sys

#%% Read layfile

niifile=sys.argv[1]
extract_nr=int(sys.argv[2])

nii,hdr=readnii(niifile)

nii_write=np.squeeze(nii[:,:,:,extract_nr]).astype(np.float32)

hdr['dim']=(3,)+hdr['dim'][1:4]+(1,)+hdr['dim'][5:]
hdr['pixdim']=hdr['pixdim'][:4]+(0.0,)+hdr['pixdim'][5:]
hdr['datatype']=16
hdr['bitpix']=32
hdr['vox_offset']=352

niifilebase,niifilesuffix=niifile.split('.nii.gz')

savenii(nii_write,hdr,niifilebase+'_n'+str(extract_nr)+'.nii.gz')
