#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 09:48:22 2026

@author: WauWter
"""
import os
import sys
from .wauwternifti import readnii
from .wauwterfmri import loadmp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

orig_nii=sys.argv[1]
proc_nii=sys.argv[2]
mpfile1=sys.argv[3]
mpfile2=sys.argv[4]

data0,hdr0 = readnii(orig_nii,scaling=False)
data1,hdr1 = readnii(proc_nii,scaling=False)
data0=data0.astype(np.float32)
data1=data1.astype(np.float32)

mp1=loadmp(mpfile1,csv=1)
mp2=loadmp(mpfile2,csv=1)

def get_random_brain_voxels(data, num_voxels=10, intensity_threshold=1000):
    """
    Pick random voxels likely within the brain.

    Parameters:
    - data_4d: 4D numpy array (X, Y, Z, Time)
    - num_voxels: Number of random voxels to pick (default: 10)
    - intensity_threshold: Minimum intensity to consider as "brain" (adjust as needed)

    Returns:
    - List of (x, y, z) coordinates for the selected voxels
    """
    mean_intensity = np.mean(data, axis=3)
    brain_mask = mean_intensity > intensity_threshold
    brain_coords = np.argwhere(brain_mask)

    # Randomly select 'num_voxels' coordinates
    if len(brain_coords) < num_voxels:
        raise ValueError(f"Only {len(brain_coords)} voxels meet the intensity threshold. Try lowering the threshold.")
    random_indices = np.random.choice(len(brain_coords), size=num_voxels, replace=False)
    random_voxels = brain_coords[random_indices]

    return random_voxels

def plot_voxel_timeseries(data, voxels=None, start=0, stop=None, title="Voxel Time Series"):
    """Plot time series for selected voxels."""
    if voxels is None:
        # Pick a few random voxels
        # voxels = [(int(data.shape[0]//1.5), int(data.shape[1]//1.5), int(data.shape[2]//1.5)),
        #           (data.shape[0]//2, data.shape[1]//2, data.shape[2]//2),
        #           (data.shape[0]//3, data.shape[1]//3, data.shape[2]//3)]
        voxels = get_random_brain_voxels(data)
    plt.figure(figsize=(12, 6))
    if stop is None:
        stop=data.shape[3]
    for i, (x, y, z) in enumerate(voxels):
        ts = data[x, y, z, start:stop]
        plt.plot(ts, label=f'Voxel {i+1} ({x}, {y}, {z})')
    plt.title(title)
    plt.xlabel('Time (TR)')
    plt.ylabel('Signal Intensity')
    plt.legend()
    plt.grid()
    # plt.show()

def plot_mean_std(data, zx=1.35, title="Mean, Std and tSNR"):
    """Plot mean and std across time for each voxel (axial slice)."""
    mean_data = np.mean(data, axis=3)
    std_data = np.std(data, axis=3)
    tsnr = np.divide(mean_data, std_data, out=np.zeros_like(mean_data), where=std_data!=0)
    fig, axes = plt.subplots(3, 3, figsize=(18, 24))
    im_mean = axes[0,0].imshow(mean_data[:, :, int(data.shape[2]//zx)], cmap='gray')
    axes[0,0].set_title("Mean")
    axes[0,0].axis('off')
    plt.colorbar(im_mean, ax=axes[0,0], fraction=0.046, pad=0.04)
    im_std = axes[0,1].imshow(std_data[:, :, int(data.shape[2]//zx)], cmap='inferno')
    axes[0,1].set_title("Std")
    axes[0,1].axis('off')
    plt.colorbar(im_std, ax=axes[0,1], fraction=0.046, pad=0.04)
    im_tsnr = axes[0,2].imshow(tsnr[:, :, int(data.shape[2]//zx)], cmap='viridis')
    axes[0,2].set_title("tSNR")
    axes[0,2].axis('off')
    plt.colorbar(im_tsnr, ax=axes[0,2], fraction=0.046, pad=0.04)
    im_mean = axes[1,0].imshow(mean_data[:, int(data.shape[1]//zx),:], cmap='gray')
    axes[1,0].set_title("Mean")
    axes[1,0].axis('off')
    plt.colorbar(im_mean, ax=axes[1,0], fraction=0.046, pad=0.04)
    im_std = axes[1,1].imshow(std_data[:, int(data.shape[1]//zx),:], cmap='inferno')
    axes[1,1].set_title("Std")
    axes[1,1].axis('off')
    plt.colorbar(im_std, ax=axes[1,1], fraction=0.046, pad=0.04)
    im_tsnr = axes[1,2].imshow(tsnr[:, int(data.shape[1]//zx),:], cmap='viridis')
    axes[1,2].set_title("tSNR")
    axes[1,2].axis('off')
    plt.colorbar(im_tsnr, ax=axes[1,2], fraction=0.046, pad=0.04)
    im_mean = axes[2,0].imshow(mean_data[int(data.shape[0]//zx),:,:], cmap='gray')
    axes[2,0].set_title("Mean")
    axes[2,0].axis('off')
    plt.colorbar(im_mean, ax=axes[2,0], fraction=0.046, pad=0.04)
    im_std = axes[2,1].imshow(std_data[int(data.shape[0]//zx),:,:], cmap='inferno')
    axes[2,1].set_title("Std")
    axes[2,1].axis('off')
    plt.colorbar(im_std, ax=axes[2,1], fraction=0.046, pad=0.04)
    im_tsnr = axes[2,2].imshow(tsnr[int(data.shape[0]//zx),:,:], cmap='viridis')
    axes[2,2].set_title("tSNR")
    axes[2,2].axis('off')
    plt.colorbar(im_tsnr, ax=axes[2,2], fraction=0.046, pad=0.04)
    plt.suptitle(title)
    # plt.show()


def carpet_plot(data, cut=30, zscore=False, title="Carpet Plot"):
    """Generate a carpet plot."""
    # Reshape data for carpet plot
    mean_data = np.mean(data, axis=3)
    cutoff = np.percentile(mean_data, cut)
    included = np.where(mean_data>=cutoff)
    carpet_data = data[included[0],included[1],included[2],:]
    if zscore:
        for i in range(carpet_data.shape[1]):
            carpet_data[:,i]=(carpet_data[:,i]-np.mean(carpet_data[:,i]))/np.std(carpet_data[:,i])
    plt.figure(figsize=(12, 6))
    plt.imshow(carpet_data, aspect='auto', cmap='viridis')
    plt.title(title)
    plt.xlabel('Time (TR)')
    plt.ylabel('Voxels')
    if zscore:
        plt.colorbar(label='Z-score Signal Intensity')
    else:
        plt.colorbar(label='Signal Intensity')
    # plt.show()

def dispmat(matr,title='2D-matrix'):
    msize=matr.shape
    fig, axes = plt.subplots(msize[0], 1, figsize=(msize[0],msize[0]*2))
    for i in range(msize[0]):
        axes[i].plot(matr[i,:])
    axes[0].set_title(title)
    
# Create a PDF file
niistring=orig_nii.split('.nii')
with PdfPages(niistring[0]+'_MRIQC.pdf') as pdf:
    # Plot voxel time series (original)
    plt.figure(figsize=(12, 6))
    plot_voxel_timeseries(data0, title="Original: Voxel Time Series")
    pdf.savefig()
    plt.close()

    # Plot voxel time series (preprocessed)
    plt.figure(figsize=(12, 6))
    plot_voxel_timeseries(data1, title="Preprocessed: Voxel Time Series")
    pdf.savefig()
    plt.close()
    
    # motion parameters
    plt.figure(figsize=(12, 6))
    dispmat(mp1[:,0:265],title='motionpar:3trans+3rot')
    pdf.savefig()
    plt.close()

    # motion parameters disp
    plt.figure(figsize=(12, 6))
    dispmat(mp2[:,0:265],title='motionpar:GlobFWD,TDxyz,RDx,RDy,RDz')
    pdf.savefig()
    plt.close()

    # Plot mean and std (original)
    plt.figure(figsize=(12, 6))
    plot_mean_std(data0, title="Original: Mean,Std & tSNR")
    pdf.savefig()
    plt.close()

    # Plot mean and std (preprocessed)
    plt.figure(figsize=(12, 6))
    plot_mean_std(data1, title="Preprocessed: Mean, Std & tSNR")
    pdf.savefig()
    plt.close()

    # Carpet plot (original)
    plt.figure(figsize=(12, 6))
    carpet_plot(data0, title="Original: Carpet Plot")
    pdf.savefig()
    plt.close()

    # Carpet plot (preprocessed)
    plt.figure(figsize=(12, 6))
    carpet_plot(data1, title="Preprocessed: Carpet Plot")
    pdf.savefig()
    plt.close()


    
