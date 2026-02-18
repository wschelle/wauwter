#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:44:41 2026

@author: WauWter
"""
import numpy as np

def unique(array):
    # Initialize an empty numpy array
    unique_items = np.empty(shape=(0,), dtype=array.dtype)

    # Traverse for all elements
    for x in array:
        # Check if exists in unique_items or not
        if x not in unique_items:
            # Use numpy.append to append the element
            unique_items = np.append(unique_items, x)

    return unique_items

def butterworth_filter_3D(xdim, ydim, zdim, cutoff, order):
    distarr=np.zeros([xdim,ydim,zdim])
    for k in range(zdim):
        for j in range(ydim):
            for i in range(xdim):
                distarr[i,j,k] = np.sqrt(((i-xdim/2))**2 + ((j-ydim/2))**2 + ((k-zdim/2))**2)
    bFilter = 1.0 / np.sqrt((1 + (distarr/cutoff)**(2*order)))
    bFilter = np.roll(bFilter, (int(np.ceil(xdim/2)), int(np.ceil(ydim/2)), int(np.ceil(zdim/2))),axis=(0,1,2))
    return(bFilter)

def gaussian_filter_3D(sigmas,kernel_size):
    kernel=np.zeros([kernel_size,kernel_size,kernel_size])
    center=kernel_size//2
    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                kernel[i,j,k]=np.exp(-(((center-i)**2/(2*sigmas[0]**2)) + ((center-j)**2/(2*sigmas[1]**2)) + ((center-k)**2/(2*sigmas[2]**2))))
    kernel/=np.max(kernel)
    return(kernel)

def convol_nan_3D(data,kernel,fillnan=False):
    result=np.zeros(data.shape)
    ksx=kernel.shape[0]
    ksy=kernel.shape[1]
    ksz=kernel.shape[2]
    data2=np.empty([data.shape[0]+2*ksx,data.shape[1]+2*ksy,data.shape[2]+2*ksz])
    data2[:]=np.nan
    data2[ksx:-ksx,ksy:-ksy,ksz:-ksz]=data
    for i in range(ksx,data.shape[0]+ksx):
        for j in range(ksy,data.shape[1]+ksy):
            for k in range(ksz,data.shape[2]+ksz):
                if (~np.isnan(data2[i,j,k])) | fillnan:
                    tmp=data2[i-ksx//2:i+round(ksx/2),j-ksy//2:j+round(ksy/2),k-ksz//2:k+round(ksz/2)]*kernel
                    result[i-ksx,j-ksy,k-ksz]=np.nansum(tmp)/np.sum(kernel[~np.isnan(tmp)])
    return(result)
    
def satterthwaite_dof(data,group_column=None,data_column=None,group_column_idx=0,data_column_idx=0):
    if group_column==None:
        groups=unique(data[:,group_column_idx])
        groupvar=np.zeros([len(groups),2])
        for i in range(len(groups)):
            groupvar[i,0]=np.std(data[data[:,group_column_idx]==groups[i],data_column_idx])**2
            groupvar[i,1]=np.sum(data[:,group_column_idx]==groups[i])
    else:
        groups=unique(data[group_column])
        groupvar=np.zeros([len(groups),2])
        groupdat=data[data_column]
        for i in range(len(groups)):
            groupvar[i,0]=np.std(groupdat[data[group_column]==groups[i]])**2
            groupvar[i,1]=np.sum(data[group_column]==groups[i])
    #print('groupvar: ',groupvar)
    term1=0
    term2=0
    for i in range(len(groups)):
        term1+=groupvar[i,0]/groupvar[i,1]
        term2+=(groupvar[i,0]/groupvar[i,1])**2 / (groupvar[i,1]-1)
    term1=term1**2
    dof=term1/term2
    #print('term1: ',term1)
    #print('term2: ',term2)
    return(dof)

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

    