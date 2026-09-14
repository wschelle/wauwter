#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:44:41 2023

@author: wousch
"""
import numpy as np
from scipy import stats

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

def gaussian_kernel_3D(sigmas, kernel_size):
    if np.isscalar(sigmas):
        sigmas = [sigmas, sigmas, sigmas]
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing='ij')
    kernel = np.exp(-0.5 * (xx**2 / sigmas[0]**2 + yy**2 / sigmas[1]**2 + zz**2 / sigmas[2]**2))
    kernel /= np.sum(kernel)  # Normalize
    return kernel

def gaussian_filter_3D(sigmas,kernel_size):
    kernel=np.zeros([kernel_size,kernel_size,kernel_size])
    center=kernel_size//2
    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                kernel[i,j,k]=np.exp(-(((center-i)**2/(2*sigmas[0]**2)) + ((center-j)**2/(2*sigmas[1]**2)) + ((center-k)**2/(2*sigmas[2]**2))))
    kernel/=np.max(kernel)
    return(kernel)

def convolve_nan_3D(data, kernel, fillnan=False):
    result = np.zeros_like(data)
    ksx, ksy, ksz = kernel.shape
    data_padded = np.pad(data, ((ksx//2, ksx//2), (ksy//2, ksy//2), (ksz//2, ksz//2)), mode='constant', constant_values=np.nan)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                patch = data_padded[i:i+ksx, j:j+ksy, k:k+ksz]
                if (~np.isnan(patch)).any() or fillnan:
                    weights = kernel * (~np.isnan(patch))
                    result[i, j, k] = np.nansum(patch * kernel) / np.sum(weights)
    return result

    
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

def multiple_correlation(X,y):
    # X = np array [nfactors,ntime]
    # y = np array [ntime]
    c=np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        c[i]=np.corrcoef(X[i,:],y)[0][1]
    R = np.corrcoef(X)
    Rsq = c.T @ np.linalg.inv(R) @ c
    return (Rsq)

def pearson_cc(x,y):
    numerator=np.sum((x-np.mean(x))*(y-np.mean(y)))
    denominator=np.sqrt(np.sum((x-np.mean(x))**2) * np.sum((y-np.mean(y))**2))
    r=numerator/denominator
    return r

def linreg(x,y):
    X = np.asarray([np.ones(len(x)), x]).T
    beta_0, beta_1 = np.linalg.inv(X.T @ X) @ X.T @ y
    return np.array([beta_0, beta_1])
    
def quick_pca(data,k=2):
    data -= data.mean(axis=0) # data needs to be in shape [n_observations,n_features]
    U, S, Vt = np.linalg.svd(data, full_matrices=False)
    pc = U[:, :k] * S[:k]
    return pc

def convert_t_to_z(t_data, dof):
    # Convert t-values to p-values, then to Z-scores
    z_scores = stats.norm.ppf(stats.t.cdf(t_data, dof))
    z_scores[np.isnan(z_scores)] = 0
    z_scores[np.isinf(z_scores)] = 0
    return z_scores

def cosine_dist(x,y):
    cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return 1-cos_sim
    
