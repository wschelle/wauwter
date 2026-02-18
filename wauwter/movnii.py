#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 01:47:03 2023

@author: WauWter
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .wauwternifti import readnii

def nii_movie(niifile,slicex=None,slicey=None,slicez=None,windowheight=6,colmap='gray',speed=8,scale=75, save_gif=False,gif_path=None):
    if isinstance(niifile,str):
        nii,hdr=readnii(niifile)
    else:
        nii=niifile
        
    if scale !=100:
        nii[nii>np.max(nii)*(scale/100)]=np.max(nii)*(scale/100)
    
    niisize=nii.shape
    
    # Determine slice positions if not provided
    sx = niisize[0] // 2.5 if slicex is None else slicex
    sy = niisize[1] // 2.5 if slicey is None else slicey
    sz = niisize[2] // 2.5 if slicez is None else slicez
    frl = niisize[3]
    
    frame = 0
    
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(2*windowheight, windowheight))
    imx=ax1.imshow(nii[:,:,int(sz),int(frame)],cmap=colmap)
    imy=ax2.imshow(nii[:,int(sy),:,int(frame)],cmap=colmap)
    imz=ax3.imshow(nii[int(sx),:,:,int(frame)],cmap=colmap)
    time_text = ax3.text(0.96, 0.98, '', transform=ax3.transAxes, color='white', fontsize=12, weight='bold',
                         ha='right', va='top')

    def update(*args):
        nonlocal frame, sx, sy, sz, frl
     
        imx.set_array(nii[:,:,int(sz),int(frame)])
        imy.set_array(nii[:,int(sy),:,int(frame)])
        imz.set_array(nii[int(sx),:,:,int(frame)])
        
        time_text.set_text(f'Frame: {frame}')

        frame += 1
        frame %= frl

        return imx,imy,imz,time_text

    ani = animation.FuncAnimation(fig, update, interval=speed,blit=True, cache_frame_data=False)
    if save_gif:
        ani.save(gif_path, writer='imagemagick', fps=30)
    plt.show()
    return ani
