#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 00:23:03 2023

@author: WauWter
"""

import numpy as np
from .wauwternifti import readnii, savenii
from .wauwtermisc import butterworth_filter_3D, gaussian_filter_3D, convol_nan_3D
from .FrangiFilter2D import Hessian2D, eig2image, FrangiFilter2D
import os
from scipy import ndimage, signal
import json
import copy
from tqdm import tqdm
import math

def mp2rage_norm(mp2r_dir,mp2r_file):
    nii,hdr=readnii(mp2r_dir+mp2r_file)
    nii-=hdr['scl_inter']
    
    hg,bins=np.histogram(nii,bins=10000)
    maxbin=np.where(hg==np.max(hg))[0]
    
    mask=np.zeros(nii.shape,dtype=np.int16)
    for i in range(1,nii.shape[0]-1):
        for j in range(1,nii.shape[1]-1):
            for k in range(1,nii.shape[2]-1):
                if (nii[i,j,k]>bins[maxbin-1]) & (nii[i,j,k]<bins[maxbin+1]):
                    if np.sum((nii[i-1:i+2,j-1:j+2,k-1:k+2]>bins[maxbin-1]) & (nii[i-1:i+2,j-1:j+2,k-1:k+2]<bins[maxbin+1])) >= 14:
                        mask[i,j,k]=1
    mask[0,:,:]=1
    mask[:,0,:]=1
    mask[:,:,0]=1
    mask[-1,:,:]=1
    mask[:,-1,:]=1
    mask[:,:,-1]=1

    t1b=nii.astype(np.float32)
    t1b[mask==0]+=bins[maxbin]
    t1b-=bins[maxbin]

    hdr['scl_slope']=1
    hdr['scl_inter']=0
    hdr['bitpix']=32
    hdr['datatype']=16
    
    savenii(t1b,hdr,mp2r_dir+'i'+mp2r_file)

    m2=np.zeros(nii.shape,dtype=np.int16)
    m2[t1b>10]=1
    hdr['bitpix']=16
    hdr['datatype']=4
    
    savenii(m2,hdr,mp2r_dir+'mask-i'+mp2r_file)

def survey_norm(survey_dir,survey_file,cutoff=3,order=1.75,minmask=150):
    nii,hdr=readnii(survey_dir+survey_file)
    
    filt=butterworth_filter_3D(nii.shape[0],nii.shape[1],nii.shape[2],cutoff,order)
    nii_filt = np.abs(np.fft.ifftn(np.fft.fftn(nii) * filt))
    
    mask=np.zeros(nii.shape,dtype=np.int16)
    mask[nii>minmask]=1
    
    nii_adjusted=np.zeros(nii.shape)
    nii_adjusted[mask==1]=nii[mask==1] / nii_filt[mask==1]
    nii_adjusted = (nii_adjusted**2)
    nii_adjusted[nii_adjusted >= 3.5]/=8
    nii_adjusted*=400
    nii_adjusted=nii_adjusted.astype(np.float32)
    hdr['datatype']=16
    hdr['bitpix']=32
    hdr['scl_slope']=1
    savenii(nii_adjusted,hdr,survey_dir+'i'+survey_file)

def fillgaps(nii_dir, nii_file, gap=0, fillthres=None, boxsize=3, minv=1, maxv=None, helperfile=None):
    nii,hdr=readnii(nii_dir+nii_file,scaling=False)
    
    boxmin=int((boxsize-1)//2)
    boxmax=int(np.ceil((boxsize-1)/2))
    if maxv==None:
        maxv=np.max(nii)
    
    boxtotal=boxsize**3
    if not fillthres:
        fillthres=boxtotal//2
    maxgaps=boxtotal-fillthres
    
    nii3=copy.deepcopy(nii).astype(np.float32)
    
    if helperfile:
        nii2,hdr2=readnii(helperfile,scaling=False)
        for i in tqdm(range(boxmin,nii.shape[0]-boxmax)):
            for j in range(boxmin,nii.shape[1]-boxmax):
                for k in range(boxmin,nii.shape[2]-boxmax):
                    if (nii[i,j,k]==gap) & (nii2[i,j,k]!=0) & (np.sum(nii[i-boxmin:i+boxmax+1,j-boxmin:j+boxmax+1,k-boxmin:k+boxmax+1]!=gap)>0):
                        hg,loc=np.histogram(nii[i-boxmin:i+boxmax+1,j-boxmin:j+boxmax+1,k-boxmin:k+boxmax+1],bins=100)
                        hg=hg[loc[:-1]!=gap]
                        loc=loc[loc!=gap]
                        if hg.max() > 0:
                            maxloc=loc[np.where(hg == hg.max())[0][0]+1]
                        else:
                            maxloc=np.max(nii[i-boxmin:i+boxmax+1,j-boxmin:j+boxmax+1,k-boxmin:k+boxmax+1])
                        nii3[i,j,k]=maxloc
    else:
        for i in range(boxmin,nii.shape[0]-boxmax):
            for j in range(boxmin,nii.shape[1]-boxmax):
                for k in range(boxmin,nii.shape[2]-boxmax):
                    if (nii[i,j,k]==gap) & (np.sum(nii[i-boxmin:i+boxmax+1,j-boxmin:j+boxmax+1,k-boxmin:k+boxmax+1]==gap) <= maxgaps):
                        hg,loc=np.histogram(nii[i-boxmin:i+boxmax+1,j-boxmin:j+boxmax+1,k-boxmin:k+boxmax+1],bins=int(np.max(nii[i-boxmin:i+boxmax+1,j-boxmin:j+boxmax+1,k-boxmin:k+boxmax+1])))
                        maxloc=loc[np.where(hg == np.max(hg))]+1
                        maxloc=maxloc[0]
                        if (maxloc >= minv) & (maxloc <= maxv) & (np.max(hg) >= fillthres):
                            nii3[i,j,k]=maxloc
                        
    hdr['datatype']=16
    hdr['bitpix']=32
    hdr['scl_slope']=1
    hdr['scl_inter']=0
    hdr['vox_offset']=352
    savenii(nii3,hdr,nii_dir+'fill_'+nii_file)

def fillgapsavg(nii_dir, nii_file, gap=0, fillthres_ratio=1/3, boxsize=3, filltype='mean', prefix='fill'):
    nii,hdr=readnii(nii_dir+nii_file,scaling=False)
    
    boxmin=int((boxsize-1)//2)
    boxmax=int(np.ceil((boxsize-1)/2))
    
    boxtotal=boxsize**3
    fillthres=int(boxtotal*fillthres_ratio)
    
    niif=copy.deepcopy(nii).astype(np.float32)
    
    for i in range(boxmin,nii.shape[0]-boxmax):
        for j in range(boxmin,nii.shape[1]-boxmax):
            for k in range(boxmin,nii.shape[2]-boxmax):
                if (nii[i,j,k]==gap) & (np.sum(nii[i-boxmin:i+boxmax+1,j-boxmin:j+boxmax+1,k-boxmin:k+boxmax+1]!=gap) > fillthres):
                    tmp=nii[i-boxmin:i+boxmax+1,j-boxmin:j+boxmax+1,k-boxmin:k+boxmax+1]
                    if filltype=='mean':
                        niif[i,j,k]=np.mean(tmp[(tmp!=gap)])
                    elif filltype=='max':
                        niif[i,j,k]=np.max(tmp[(tmp!=gap)])
                    elif filltype=='min':
                        niif[i,j,k]=np.min(tmp[(tmp!=gap)])
                    else:
                        niif[i,j,k]=np.median(tmp[(tmp!=gap)])
                          
    hdr['datatype']=16
    hdr['bitpix']=32
    hdr['scl_slope']=1
    hdr['scl_inter']=0
    hdr['vox_offset']=352
    savenii(niif,hdr,nii_dir+prefix+'_'+nii_file)


def atropos_seg(an4_dir,gmin=4,gmax=6,reversed_contrast=False,weirdMP2R_contrast=False,gmid=5):
    
    an4files=[]
    for f in os.listdir(an4_dir):
        if f.endswith("nii.gz"):
            an4files.append(f)
    nr_files=len(an4files)-2
    
    t1,hdr=readnii(an4_dir+'AN4Segmentation0N4.nii.gz')
    an4pos=np.zeros([t1.shape[0],t1.shape[1],t1.shape[2],nr_files],dtype=np.float32)
    for i in range(nr_files):
        an4pos[:,:,:,i],hdr=readnii(an4_dir+'AN4SegmentationPosteriors'+str(i+1)+'.nii.gz',scaling=False)
    
    an4pos[an4pos >= 0.1]=1
    an4pos[an4pos < 0.1]=0
    
    if reversed_contrast:
        csf=np.sum(an4pos[:,:,:,gmax:],axis=3)
        gm=np.sum(an4pos[:,:,:,gmin-1:gmax],axis=3)
        wm=np.sum(an4pos[:,:,:,0:gmin-1],axis=3)
    elif weirdMP2R_contrast:
        csf=np.sum(an4pos[:,:,:,gmax-1:],axis=3)
        gm=np.sum(an4pos[:,:,:,gmin-1:gmid],axis=3)
        wm=np.sum(an4pos[:,:,:,gmid:gmax-1],axis=3)
    else:    
        csf=np.sum(an4pos[:,:,:,0:gmin],axis=3)
        gm=np.sum(an4pos[:,:,:,gmin:gmax],axis=3)
        wm=np.sum(an4pos[:,:,:,gmax:],axis=3)
        
    # csf[csf >= 1]=1
    # gm[gm >= 1]=1
    # wm[wm >= 1]=1
    
    fwhm=2*np.sqrt(2*np.log(2))*hdr['pixdim'][1]
    smm=0.75
    sigma=smm/fwhm
    
    csf = ndimage.gaussian_filter(csf, sigma, mode='mirror')
    gm = ndimage.gaussian_filter(gm, sigma, mode='mirror')
    wm = ndimage.gaussian_filter(wm, sigma, mode='mirror')
    
    if weirdMP2R_contrast:
        csf[csf>=0.05]=1
        wm[wm>=0.05]=1
        gm[gm>=0.05]=1
    else:
        csf[csf>=0.7]=1
        wm[wm>=0.7]=1
        gm[gm>=0.3]=1
    
    csfgm=np.zeros(t1.shape)
    gmwm=np.zeros(t1.shape)
    
    for i in range(t1.shape[0]):
        for j in range(t1.shape[1]):
            for k in range(t1.shape[2]):
                minx=i-1
                maxx=i+2
                miny=j-1
                maxy=j+2
                minz=k-1
                maxz=k+2
                if minx < 0: minx=0
                if miny < 0: miny=0
                if minz < 0: minz=0
                if maxx > t1.shape[0]: maxx=t1.shape[0]
                if maxy > t1.shape[1]: maxy=t1.shape[1]
                if maxz > t1.shape[2]: maxz=t1.shape[2]
                if (csf[i,j,k]==1) & (np.max(gm[minx:maxx,miny:maxy,minz:maxz])==1):
                    csfgm[i,j,k]=1
                if (wm[i,j,k]==1) & (np.max(gm[minx:maxx,miny:maxy,minz:maxz])==1):
                    gmwm[i,j,k]=1
        
    segt1=np.zeros(t1.shape,dtype=np.int32)
    if weirdMP2R_contrast:
        segt1[gmwm==1]=2
        segt1[csfgm==1]=1
        segt1[gm==1]=3
    else:
        segt1[csfgm==1]=1
        segt1[gmwm==1]=2
        segt1[gm==1]=3
    
    for i in range(t1.shape[0]):
        for j in range(t1.shape[1]):
            for k in range(t1.shape[2]):
                minx=i-1
                maxx=i+2
                miny=j-1
                maxy=j+2
                minz=k-1
                maxz=k+2
                if minx < 0: minx=0
                if miny < 0: miny=0
                if minz < 0: minz=0
                if maxx > t1.shape[0]: maxx=t1.shape[0]
                if maxy > t1.shape[1]: maxy=t1.shape[1]
                if maxz > t1.shape[2]: maxz=t1.shape[2]
                if (segt1[i,j,k]==3) & (np.sum(segt1[minx:maxx,miny:maxy,minz:maxz]==0)>0):
                    for ii in range(minx,maxx):
                        for jj in range(miny,maxy):
                            for kk in range(minz,maxz):
                                if (segt1[ii,jj,kk]==0)&(t1[ii,jj,kk]<=t1[i,j,k]):
                                    segt1[ii,jj,kk]=1
                                if (segt1[ii,jj,kk]==0)&(t1[ii,jj,kk]>t1[i,j,k]):
                                    segt1[ii,jj,kk]=2
                    
    hdr['datatype']=8
    hdr['bitpix']=32
    hdr['scl_slope']=1
    hdr['scl_inter']=0
    savenii(segt1,hdr,an4_dir+'segt1.nii')

def MP2Rclass_seg(gm_file,wm_file,csf_file,out_file):
    
    gm,hdrg=readnii(gm_file)
    wm,hdrw=readnii(wm_file)
    csf,hdrc=readnii(csf_file)
            
    segt1=np.zeros(gm.shape,dtype=np.int16)
    segt1[gm >= 0.05]=3
    
    for i in range(gm.shape[0]):
        for j in range(gm.shape[1]):
            for k in range(gm.shape[2]):
                minx=i-1
                maxx=i+2
                miny=j-1
                maxy=j+2
                minz=k-1
                maxz=k+2
                if minx < 0: minx=0
                if miny < 0: miny=0
                if minz < 0: minz=0
                if maxx > gm.shape[0]: maxx=gm.shape[0]
                if maxy > gm.shape[1]: maxy=gm.shape[1]
                if maxz > gm.shape[2]: maxz=gm.shape[2]
                if (segt1[i,j,k]==0) & (np.sum(segt1[minx:maxx,miny:maxy,minz:maxz]==3)>0):
                    if wm[i,j,k] > csf[i,j,k]:
                        segt1[i,j,k]=2
                    elif csf[i,j,k] > wm[i,j,k]:
                        segt1[i,j,k]=1
                    elif np.sum(wm[minx:maxx,miny:maxy,minz:maxz]) > np.sum(csf[minx:maxx,miny:maxy,minz:maxz]):
                        segt1[i,j,k]=2
                    else:
                        segt1[i,j,k]=1

    hdrg['datatype']=4
    hdrg['bitpix']=16
    hdrg['scl_slope']=1
    hdrg['scl_inter']=0
    hdrg['vox_offset']=352
    savenii(segt1,hdrg,out_file)

def fsribbon_seg(ribbon_file,t1_file,out_file):
    
    gm,hdr=readnii(ribbon_file)
    t1,hdrt1=readnii(t1_file)
      
    segt1=np.zeros(gm.shape,dtype=np.int16)
    segt1[gm > 0]=3
    
    for i in range(gm.shape[0]):
        for j in range(gm.shape[1]):
            for k in range(gm.shape[2]):
                if (segt1[i,j,k]==0):
                    minx=i-1
                    maxx=i+2
                    miny=j-1
                    maxy=j+2
                    minz=k-1
                    maxz=k+2
                    if minx < 0: minx=0
                    if miny < 0: miny=0
                    if minz < 0: minz=0
                    if maxx > gm.shape[0]: maxx=gm.shape[0]
                    if maxy > gm.shape[1]: maxy=gm.shape[1]
                    if maxz > gm.shape[2]: maxz=gm.shape[2]
                    if (np.sum(segt1[minx:maxx,miny:maxy,minz:maxz]==3)>0):
                        tmp1=copy.deepcopy(t1[minx:maxx,miny:maxy,minz:maxz])
                        tmp2=copy.deepcopy(segt1[minx:maxx,miny:maxy,minz:maxz])
                        tmp=np.mean(tmp1[tmp2==3])
                        if t1[i,j,k] <= tmp:
                            segt1[i,j,k]=1
                        elif t1[i,j,k] > tmp:
                            segt1[i,j,k]=2
                            
    for i in range(gm.shape[0]):
        for j in range(gm.shape[1]):
            for k in range(gm.shape[2]):
                if segt1[i,j,k] == 2:
                    minx=i-1
                    maxx=i+2
                    miny=j-1
                    maxy=j+2
                    minz=k-1
                    maxz=k+2
                    if minx < 0: minx=0
                    if miny < 0: miny=0
                    if minz < 0: minz=0
                    if maxx > gm.shape[0]: maxx=gm.shape[0]
                    if maxy > gm.shape[1]: maxy=gm.shape[1]
                    if maxz > gm.shape[2]: maxz=gm.shape[2]
                    #if np.sum(segt1[minx:maxx,miny:maxy,minz:maxz] == 1) > np.sum(segt1[minx:maxx,miny:maxy,minz:maxz] == 3):
                    #if np.sum(segt1[minx:maxx,miny:maxy,minz:maxz] == 1) > 3:
                    if np.sum(segt1[minx:maxx,miny:maxy,minz:maxz] == 1) > 5:
                        segt1[i,j,k]=1

    hdr['datatype']=4
    hdr['bitpix']=16
    hdr['scl_slope']=1
    hdr['scl_inter']=0
    hdr['vox_offset']=352
    savenii(segt1,hdr,out_file)

def fsl_topup_params(json_file,phasedim="y"):
    with open(json_file) as f:
        data = json.load(f)
    te=data['EchoTime']
    acq_params=np.zeros([2,4],dtype=np.float16)
    if phasedim=="y":
        acq_params[0,1]=1
        acq_params[1,1]=-1
    elif phasedim=="z":
        acq_params[0,2]=1
        acq_params[1,2]=-1
    else:
        acq_params[0,0]=1
        acq_params[1,0]=-1
    acq_params[:,3]=te
    jdir=json_file.split('.json')[0]
    np.savetxt(jdir+'_acq_param.txt', acq_params, fmt='%8.6f')

def layersmooth(niifile,layfile,fwhm=1,kernelsize=7,nlay=6,parcfile='',lowcut=10,suffix='ls'):
    nii,hdr=readnii(niifile)
    lay,hdrlay=readnii(layfile,scaling=False)
    
    nvox=hdr['dim'][1]*hdr['dim'][2]*hdr['dim'][3]
    nscans=hdr['dim'][4]
    mask=np.zeros([hdr['dim'][1],hdr['dim'][2],hdr['dim'][3]],dtype=np.int16)
    mask[np.nanmean(nii,axis=3) > lowcut]=1
    nii=np.reshape(nii,[nvox,nscans])
    lay=np.reshape(lay,nvox)
    
    if parcfile != '':
        parc,hdrparc=readnii(parcfile,scaling=False)
        mask*=parc.astype(np.int16)
    mask=np.reshape(mask,nvox)
    del parc

    layz=np.max(lay)
    lay=np.ceil(lay/(layz/float(nlay)))
    
    ks=kernelsize
    voxfwhm=fwhm/np.array(hdr['pixdim'][1:4])
    gsigma=voxfwhm/(2*np.sqrt(2*np.log(2)))
    gf=gaussian_filter_3D(gsigma,ks)
    
    niipad=np.empty([hdr['dim'][1]+(2*ks),hdr['dim'][2]+(2*ks),hdr['dim'][3]+(2*ks),nscans])
    
    niic=np.zeros([hdr['dim'][1]+(2*ks),hdr['dim'][2]+(2*ks),hdr['dim'][3]+(2*ks),nscans],dtype=np.float32)
    niis=np.zeros([nvox,nscans],dtype=np.float32)
    nvar=np.max(mask)
    
    print('Start layer smoothing of '+niifile)
    for ii in range(1,nvar+1):
        for jj in range(1,nlay+1):
            niitmp=copy.deepcopy(nii)
            niipad[:]=np.nan
            niic*=0
            niitmp[(mask!=ii) | (lay!=jj),:]=np.nan
            niitmp=np.reshape(niitmp,[hdr['dim'][1],hdr['dim'][2],hdr['dim'][3],nscans])
            niipad[ks:-ks,ks:-ks,ks:-ks,:]=copy.deepcopy(niitmp)
            for kk in range(nscans):
                niic[:,:,:,kk]=convol_nan_3D(np.squeeze(niipad[:,:,:,kk]),gf)
            niicunpad=np.reshape(niic[ks:-ks,ks:-ks,ks:-ks,:],[nvox,nscans])
            niis[(mask==ii) & (lay==jj),:]=copy.deepcopy(niicunpad[(mask==ii) & (lay==jj),:])
            print('smoothed '+str(jj) +' of '+str(nlay)+' layers in '+str(ii)+' of '+str(nvar)+' areas')
    
    hdr['datatype']=16
    hdr['bitpix']=32
    hdr['scl_slope']=1
    hdr['scl_inter']=0
    niis=np.reshape(niis,[hdr['dim'][1],hdr['dim'][2],hdr['dim'][3],nscans]).astype(np.float32)
    niistring=niifile.split('.')
    if niistring[-1]=='gz':
        newnii=niistring[0]+'-'+suffix+'.'+niistring[1]+'.'+niistring[2]
    else:
        newnii=niistring[0]+'-'+suffix+'.'+niistring[1]       
    savenii(niis,hdr,newnii)

def laysmo(niifile,layfile,fwhm=1,kernelsize=7,nlay=6,layedge=0,parcfile='',lowcut=10,minparc=None,maxparc=None,suffix='ls'):
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
    
    nvox=hdr['dim'][1]*hdr['dim'][2]*hdr['dim'][3]
    nscans=hdr['dim'][4]
    mask=np.zeros([hdr['dim'][1],hdr['dim'][2],hdr['dim'][3]],dtype=np.int16)
    mask[np.nanmean(nii,axis=3) > lowcut]=1
    nii=np.reshape(nii,[nvox,nscans])
    lay=np.reshape(lay,nvox)
    
    if parcfile:
        parc,hdrparc=readnii(parcfile,scaling=False)
        if not minparc:
            minparc=1
        if not maxparc:
            maxparc=parc.max()
        parc[(parc<minparc) | (parc>maxparc)]=0
        mask*=parc.astype(np.int16)
    mask=np.reshape(mask,nvox)
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
            niitmp=copy.deepcopy(nii)
            niipad[:]=np.nan
            niitmp[(mask!=ii) | (lay!=jj),:]=np.nan
            niitmp=np.reshape(niitmp,[hdr['dim'][1],hdr['dim'][2],hdr['dim'][3],nscans])
            niipad[ks:-ks,ks:-ks,ks:-ks,:]=copy.deepcopy(niitmp)
            for kk in tqdm(range(ks,hdr['dim'][1]+ks)):
                for ll in range(ks,hdr['dim'][2]+ks):
                    for mm in range(ks,hdr['dim'][3]+ks):
                        if np.sum(np.isnan(niipad[kk,ll,mm,:]))==0:
                            tmpdat=niipad[kk-ksa:kk+ksb,ll-ksa:ll+ksb,mm-ksa:mm+ksb,:]
                            mult = tmpdat * gf
                            mult = np.nansum(mult,axis=(0,1,2))
                            norm = np.sum(gf[~np.isnan(tmpdat)])/nscans
                            niic[kk-ks,ll-ks,mm-ks,:]= mult / norm
            print('smoothed '+str(jj) +' of '+str(nlay)+' layers in '+str(ii)+' of '+str(nvar)+' areas')
    
    del gf,niitmp,niipad,tmpdat,mult,norm,lay,mask,nii
    
    hdr['datatype']=16
    hdr['bitpix']=32
    hdr['scl_slope']=1
    hdr['scl_inter']=0
    hdr['vox_offset']=352
    niistring=niifile.split('.nii')
    newnii=niistring[0]+'_'+suffix+'.nii'+niistring[1]
    savenii(niic.astype(np.float32),hdr,newnii)


def frangi_vessel(niifile,outfile):
    nii,hdr=readnii(niifile)
    
    ves=np.zeros(nii.shape,dtype=np.float32)
    for i in range(nii.shape[2]):
        if np.std(nii[:,:,i]>0):
            ves[:,:,i]=FrangiFilter2D(np.squeeze(nii[:,:,i]))
    
    hdr['bitpix']=32
    hdr['datatype']=16
    hdr['vox_offset']=352
    hdr['scl_slope']=1
    savenii(ves,hdr,outfile)
    


  
