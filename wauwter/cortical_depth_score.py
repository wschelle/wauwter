#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 12:12:51 2026

@author: WauWter
"""

#%%Initiliaze
import os
os.chdir('/project/3017081.01/required/')
import numpy as np
from Python.python_scripts.wauwternifti import readnii,savenii
from Python.python_scripts.wauwtermisc import gaussian_filter_3D
from copy import deepcopy

def cortical_depth_score(layer_data,score_data,nlay=3,fwhm=2,ks=7,percentile=0.333,hdr_file=None,save_data=False,calc_negative=False,suffix='cds_pattern'):
    # Minimum input:
    #     layer_data (path or numpy array)
    #     score_data (path or numpy array)
    #     layer_data will be resampled to nlay
    #     cortical depth score will be smoothed with Gauss kernel (width fwhm and kernelsize ks)
    #     inclusion threshold for cortical depth score in percentiles
    #     if both layer_data and score_data are numpy arrays, the path to an adequate nifti file must be given for the header data

    if isinstance(layer_data, str):
        lay,hdr=readnii(layer_data,scaling=False)
        s=layer_data.split("/")[0:-1]
        filebase="/"
        filebase=filebase.join(s)
    else:
        lay=deepcopy(layer_data)
    if isinstance(score_data, str):
        score,hdr=readnii(score_data,scaling=False)
    else:
        score=deepcopy(score_data)
    if hdr_file:
        s=hdr_file.split("/")[0:-1]
        hdr=readnii(hdr_file,header_only=True)
        filebase="/"
        filebase=filebase.join(s)
    
    fx,fy,fz=lay.shape


    #mask simple mask
    mask=np.zeros((fx,fy,fz),dtype=np.int16)
    mask[(np.std(score)!=0)]=1

    #adjust layers
    if lay.max()==20:
        lay[lay==1]+=1
        lay[lay==20]-=1
        lay[lay>0]-=1
    lmx,lmy,lmz=np.where(lay>0)
    ilay=deepcopy(lay)
    laycorr=np.max(lay)/nlay
    ilay=np.ceil(lay/laycorr).astype(np.int16)
    ilay_cd=deepcopy(ilay)
    ilay_cd[lmx,lmy,lmz]-=2
    
    #create cortical depth score
    cd_score = score * ilay_cd
    
    #make gaussian smoothing kernel
    ksa=ks//2
    voxfwhm=fwhm/np.array(hdr['pixdim'][1:4])
    gsigma=voxfwhm/(2*np.sqrt(2*np.log(2)))
    gf=gaussian_filter_3D(gsigma,ks)
    
    #gaussian smoothing cortical depth score
    niipad=np.empty([hdr['dim'][1]+(2*ks),hdr['dim'][2]+(2*ks),hdr['dim'][3]+(2*ks)],dtype=np.float32)
    scd_score=np.zeros([hdr['dim'][1],hdr['dim'][2],hdr['dim'][3]],dtype=np.float32)
    
    lx,ly,lz=np.where(mask==1)
    niipad[:]=np.nan
    niipad[ksa+lx,ksa+ly,ksa+lz]=deepcopy(cd_score[lx,ly,lz])
    for kk in range(len(lx)):
        mult = niipad[lx[kk]:lx[kk]+ks,ly[kk]:ly[kk]+ks,lz[kk]:lz[kk]+ks] * gf
        mult = np.nansum(mult,axis=(0,1,2))
        norm = np.sum(gf[~np.isnan(niipad[lx[kk]:lx[kk]+ks,ly[kk]:ly[kk]+ks,lz[kk]:lz[kk]+ks])])
        scd_score[lx[kk],ly[kk],lz[kk]]= mult / norm
    
    scd_score_p = scd_score[scd_score > 0]
    scd_score_ps = np.sort(scd_score_p)[::-1]  # Sort in descending order
    target_count = percentile * len(scd_score_p)
    cutoff_p = scd_score_ps[int(target_count) - 1] if target_count > 0 else 0
    
    #make a mask of relevant cd scores
    cdmask=np.zeros((fx,fy,fz),dtype=np.int16)
    cdmask[scd_score>=cutoff_p]=1
    cd_lay=ilay*cdmask
    cd_lay_score_p=np.zeros((fx,fy,fz),dtype=np.float32)
    lx,ly,lz=np.where(ilay==2)
    for i in range(len(lx)):
        cwin=deepcopy(cd_lay[lx[i]-1:lx[i]+2,ly[i]-1:ly[i]+2,lz[i]-1:lz[i]+2])
        nsup=np.sum(cwin==3)
        ndeep=np.sum(cwin==1)
        if ((nsup>0) & (ndeep>0)):
            # print(i,nsup,ndeep)
            sx,sy,sz=np.where(cwin==3)
            dx,dy,dz=np.where(cwin==1)
            for j in range(len(sx)):
                cd_lay_score_p[lx[i]-1+sx[j],ly[i]-1+sy[j],lz[i]-1+sz[j]]=3
            for k in range(len(dx)):
                cd_lay_score_p[lx[i]-1+dx[k],ly[i]-1+dy[k],lz[i]-1+dz[k]]=1
            cd_lay_score_p[lx[i],ly[i],lz[i]]=2
    
    #write smoothed cd score
    hdr['datatype']=16
    hdr['bitpix']=32
    hdr['scl_slope']=1
    hdr['scl_inter']=0
    hdr['vox_offset']=352
    if save_data:
        savenii(cd_lay_score_p,hdr,filebase+'/'+suffix+'-positive.nii.gz')
    
    if calc_negative:
        scd_score_n = scd_score[scd_score < 0]
        scd_score_ns = np.sort(scd_score_n)  # Sort in ascending order
        target_count = percentile* len(scd_score_n)
        cutoff_n = scd_score_ns[int(target_count) - 1] if target_count > 0 else 0
        
        cdmask=np.zeros((fx,fy,fz),dtype=np.int16)
        cdmask[scd_score<=cutoff_n]=1
        cd_lay=ilay*cdmask
        cd_lay_score_n=np.zeros((fx,fy,fz),dtype=np.float32)
        lx,ly,lz=np.where(ilay==2)
        for i in range(len(lx)):
            cwin=deepcopy(cd_lay[lx[i]-1:lx[i]+2,ly[i]-1:ly[i]+2,lz[i]-1:lz[i]+2])
            nsup=np.sum(cwin==3)
            ndeep=np.sum(cwin==1)
            if ((nsup>0) & (ndeep>0)):
                sx,sy,sz=np.where(cwin==3)
                dx,dy,dz=np.where(cwin==1)
                for j in range(len(sx)):
                    cd_lay_score_n[lx[i]-1+sx[j],ly[i]-1+sy[j],lz[i]-1+sz[j]]=3
                for k in range(len(dx)):
                    cd_lay_score_n[lx[i]-1+dx[k],ly[i]-1+dy[k],lz[i]-1+dz[k]]=1
                cd_lay_score_n[lx[i],ly[i],lz[i]]=2
                
        if save_data:
            savenii(cd_lay_score_n,hdr,filebase+'/'+suffix+'-negative.nii.gz')
    
    if calc_negative:
        return cd_lay_score_p, cd_lay_score_n
    else:
        return cd_lay_score_p

#%%testing this shit on all my partcis

# rdir='/home/control/wousch/project/bids7T/'
# gdir=rdir+'group/'

# subs=['sub-001','sub-002','sub-003','sub-004','sub-005','sub-006',
#       'sub-007','sub-008','sub-009','sub-010','sub-011','sub-012',
#       'sub-013','sub-014','sub-015','sub-016','sub-017','sub-018',
#       'sub-019','sub-020','sub-021','sub-022','sub-023','sub-024',
#       'sub-025']
# nsub=len(subs)

# scorefiles=['tmap_0in','tmap_1in','tmap_2in','tmap_3in',
#          'tmap_0out','tmap_1out','tmap_2out','tmap_3out',
#          'tmap_yesmatch','tmap_nomatch',
#          'tmap_in_vs_base','tmap_out_vs_base','tmap_in_vs_0in','tmap_out_vs_0out','tmap_in_vs_out']
# nbeta1=len(blabels1)
# ntmap1=len(tlabels1)
