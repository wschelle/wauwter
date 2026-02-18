#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 12:00:00 2026

@author: WauWter
"""
#%% Initialize
import numpy as np
from .wauwternifti import readnii,savenii
import sys

#%% Read layfile

niifile=sys.argv[1]
remove_nr=int(sys.argv[2])

nii,hdr=readnii(niifile)
new_nr_dyn=int(nii.shape[3]-remove_nr)

nii_write=np.squeeze(nii[:,:,:,:new_dyn_nr]).astype(np.float32)

hdr['dim']=hdr['dim'][0:4]+(new_nr_dyn,)+hdr['dim'][5:]
hdr['datatype']=16
hdr['bitpix']=32
hdr['vox_offset']=352

niifilebase,niifilesuffix=niifile.split('.nii.gz')

savenii(nii_write,hdr,niifilebase+'_del-'+str(remove_nr)+'dyn.nii.gz')
