#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:55:00 2024

@author: WauWter
"""
#%% Initialize
#import os
#os.chdir('/project/3017081.01/required/')
import numpy as np
from .wauwternifti import readnii,savenii
from .wauwtermisc import gaussian_filter_3D
from copy import deepcopy
from multiprocessing import Pool, cpu_count


def smooth_area_layer(ii, jj, nii, mask, lay, ks, ksa, ksb, hdr, gf, nscans):
    """
    Function to perform smoothing for a single area (nvar) and single layer (nlay)
    """
    niic_layer = np.zeros([hdr['dim'][1], hdr['dim'][2], hdr['dim'][3], nscans], dtype=np.float32)
    
    niitmp = deepcopy(nii)
    niipad = np.empty([hdr['dim'][1] + (2 * ks), hdr['dim'][2] + (2 * ks), hdr['dim'][3] + (2 * ks), nscans], dtype=np.float32)
    niipad[:] = np.nan
    
    niitmp[(mask != ii) | (lay != jj), :] = np.nan
    niitmp = np.reshape(niitmp, [hdr['dim'][1], hdr['dim'][2], hdr['dim'][3], nscans])
    niipad[ks:-ks, ks:-ks, ks:-ks, :] = deepcopy(niitmp)
    
    for kk in range(ks, hdr['dim'][1] + ks):
        for ll in range(ks, hdr['dim'][2] + ks):
            for mm in range(ks, hdr['dim'][3] + ks):
                if np.sum(np.isnan(niipad[kk, ll, mm, :])) == 0:
                    tmpdat = niipad[kk-ksa:kk+ksb, ll-ksa:ll+ksb, mm-ksa:mm+ksb, :]
                    mult = tmpdat * gf
                    mult = np.nansum(mult, axis=(0, 1, 2))
                    norm = np.sum(gf[~np.isnan(tmpdat)]) / nscans
                    niic_layer[kk-ks, ll-ks, mm-ks, :] = mult / norm

    return ii, jj, niic_layer

def layersmooth(niifile, layfile, fwhm=1, kernelsize=7, nlay=6, layedge=0, parcfile='', lowcut=10, minparc=None, maxparc=None, suffix='ls'):
    nii, hdr = readnii(niifile)
    lay, hdrlay = readnii(layfile, scaling=False)
    
    nvox = hdr['dim'][1] * hdr['dim'][2] * hdr['dim'][3]
    nscans = hdr['dim'][4]
    
    mask = np.zeros([hdr['dim'][1], hdr['dim'][2], hdr['dim'][3]], dtype=np.int16)
    mask[np.nanmean(nii, axis=3) > lowcut] = 1
    nii = np.reshape(nii, [nvox, nscans])
    lay = np.reshape(lay, nvox)
    
    if parcfile:
        parc, hdrparc = readnii(parcfile, scaling=False)
        if not minparc:
            minparc = 1
        if not maxparc:
            maxparc = parc.max()
        parc[(parc < minparc) | (parc > maxparc)] = 0
        mask *= parc.astype(np.int16)
    mask = np.reshape(mask, nvox)
    del parc

    layz = np.max(lay)
    if layedge == 1:
        lay[lay == 1] += 1
        lay[lay == layz] -= 1
        lay[lay > 0] -= 1
    elif layedge == 2:
        lay[lay == 1] = 0
        lay[lay == layz] = 0
        lay[lay > 0] -= 1
    layz = np.max(lay)
    lay = np.ceil(lay / (layz / float(nlay)))
    
    ks = kernelsize
    ksa = ks // 2
    ksb = int(np.ceil(ks / 2))
    voxfwhm = fwhm / np.array(hdr['pixdim'][1:4])
    gsigma = voxfwhm / (2 * np.sqrt(2 * np.log(2)))
    gf = gaussian_filter_3D(gsigma, ks)
    gf = np.repeat(gf, nscans)
    gf = np.reshape(gf, [ks, ks, ks, nscans]).astype(np.float32)
    
    del voxfwhm, gsigma
    
    niic = np.zeros([hdr['dim'][1], hdr['dim'][2], hdr['dim'][3], nscans], dtype=np.float32)
    nvar = np.max(mask)
    
    print('Start layer smoothing of ' + niifile)
    
    # Parallelize across both nvar (areas) and nlay (layers)
    with Pool(cpu_count()) as pool:
        results = pool.starmap(smooth_area_layer, [(ii, jj, nii, mask, lay, ks, ksa, ksb, hdr, gf, nscans) 
                                                   for ii in range(1, nvar + 1)
                                                   for jj in range(1, nlay + 1)])
    
    # Aggregate results from all processes
    for ii, jj, niic_layer in results:
        niic += niic_layer
    
    del gf, nii, mask, lay
    
    hdr['datatype'] = 16
    hdr['bitpix'] = 32
    hdr['scl_slope'] = 1
    hdr['scl_inter'] = 0
    hdr['vox_offset'] = 352
    niistring = niifile.split('.nii')
    newnii = niistring[0] + '_' + suffix + '.nii' + niistring[1]
    savenii(niic.astype(np.float32), hdr, newnii)

def layersmooth_seq(niifile,layfile,fwhm=1,kernelsize=7,nlay=6,layedge=0,parcfile='',lowcut=10,minparc=None,maxparc=None,suffix='ls'):
    # Minimum input:
    #     path to niifile
    #     path to layerfile
    #     optionally give path to parcellation file (parcfile)
    # layedge has 3 options to deal with the first and last layers in the layer volume:
    #       0: do nothing (default)
    #       1: combine first 2 layers and combine last 2 layers
    #       2: remove first and last layers
    
    nii,hdr=readnii(niifile)
    lay,hdrlay=readnii(layfile,scaling=False)
    
    # nvox=hdr['dim'][1]*hdr['dim'][2]*hdr['dim'][3]
    nscans=hdr['dim'][4]
    mask=np.zeros([hdr['dim'][1],hdr['dim'][2],hdr['dim'][3]],dtype=np.int16)
    mask[np.nanmean(nii,axis=3) > lowcut]=1
    # nii=np.reshape(nii,[nvox,nscans])
    # lay=np.reshape(lay,nvox)
    
    if parcfile:
        parc,hdrparc=readnii(parcfile,scaling=False)
        if not minparc:
            minparc=1
        if not maxparc:
            maxparc=parc.max()
        parc[(parc<minparc) | (parc>maxparc)]=0
        mask*=parc.astype(np.int16)
    # mask=np.reshape(mask,nvox)
    del parc

    layz=np.max(lay)
    if layedge==1:
        lay[lay==1]+=1
        lay[lay==layz]-=1
        lay[lay>0]-=1
    elif layedge==2:
        lay[lay==1]=0
        lay[lay==layz]=0
        lay[lay>0]-=1
    layz=np.max(lay)
    lay=np.ceil(lay/(layz/float(nlay)))
    
    ks=kernelsize
    ksa=ks//2
    ksb=int(np.ceil(ks/2))
    voxfwhm=fwhm/np.array(hdr['pixdim'][1:4])
    gsigma=voxfwhm/(2*np.sqrt(2*np.log(2)))
    gf=gaussian_filter_3D(gsigma,ks)
    gf=np.repeat(gf,nscans)
    gf=np.reshape(gf,[ks,ks,ks,nscans]).astype(np.float32)
    
    del voxfwhm,gsigma
    
    niipad=np.empty([hdr['dim'][1]+(2*ks),hdr['dim'][2]+(2*ks),hdr['dim'][3]+(2*ks),nscans],dtype=np.float32)
    niic=np.zeros([hdr['dim'][1],hdr['dim'][2],hdr['dim'][3],nscans],dtype=np.float32)
    nvar=np.max(mask)
    
    print('Start layer smoothing of '+niifile)
    for ii in range(1,nvar+1):
        for jj in range(1,nlay+1):
            lx,ly,lz=np.where((lay==jj) & (mask==ii))
            if len(lx) > 0:
                niipad[:]=np.nan
                niipad[ksa+lx,ksa+ly,ksa+lz,:]=deepcopy(nii[lx,ly,lz,:])
                for kk in range(len(lx)):
                    mult = niipad[lx[kk]:lx[kk]+ks,ly[kk]:ly[kk]+ks,lz[kk]:lz[kk]+ks,:] * gf
                    mult = np.nansum(mult,axis=(0,1,2))
                    norm = np.sum(gf[~np.isnan(niipad[lx[kk]:lx[kk]+ks,ly[kk]:ly[kk]+ks,lz[kk]:lz[kk]+ks,:])])/nscans
                    niic[lx[kk],ly[kk],lz[kk],:]= mult / norm
            print('smoothed '+str(jj) +' of '+str(nlay)+' layers in '+str(ii)+' of '+str(nvar)+' areas')
                    
    del gf,niipad,mult,norm,lay,mask,nii
    
    hdr['datatype']=16
    hdr['bitpix']=32
    hdr['scl_slope']=1
    hdr['scl_inter']=0
    hdr['vox_offset']=352
    niistring=niifile.split('.nii')
    newnii=niistring[0]+'_'+suffix+'.nii'+niistring[1]
    savenii(niic.astype(np.float32),hdr,newnii)
