#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:15:12 2026

@author: WauWter
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math


def imgview_3d(
    underlay,
    overlay=None,
    ul_cmap='Greys_r',
    ol_cmap='RdBu_r',
    save_img_path=None,
    nslices=5,
    slice_axis='z',
    interpol='nearest',
    ol_alpha=0.9,
    ol_thresh=None,
    fontsize=18,
    ul_vmin=None,
    ul_vmax=None,
    ol_vmin=None,
    ol_vmax=None,
):

    plt.rc('font', size=fontsize)

    # ----------------------------
    # determine slice positions
    # ----------------------------
    if slice_axis == 'x':
        dim = underlay.shape[0]
    elif slice_axis == 'y':
        dim = underlay.shape[1]
    else:
        dim = underlay.shape[2]

    slice_range = np.round(
        np.linspace(0, dim - 1, nslices + 2)
    )[1:-1].astype(int)

    # ----------------------------
    # dynamic subplot layout
    # ----------------------------
    nplots = nslices + 1  # + overview
    ncols = math.ceil(math.sqrt(nplots))
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5*ncols, 5*nrows)
    )

    axes = np.array(axes).flatten()

    # ----------------------------
    # underlay scaling
    # ----------------------------
    ul_vmin = underlay.min() if ul_vmin is None else ul_vmin
    ul_vmax = underlay.max() if ul_vmax is None else ul_vmax

    # ----------------------------
    # overlay processing
    # ----------------------------
    if overlay is not None:

        if ol_thresh is not None:
            overlay = np.ma.masked_where(
                np.abs(overlay) < ol_thresh,
                overlay
            )

        if ol_vmax is None:
            ol_vmax = np.nanmax(np.abs(overlay))

        if ol_vmin is None:
            ol_vmin = -ol_vmax

        norm = mpl.colors.TwoSlopeNorm(
            vmin=ol_vmin,
            vcenter=0,
            vmax=ol_vmax
        )

        cmap = plt.get_cmap(ol_cmap).copy()
        cmap.set_bad(alpha=0)

    # ----------------------------
    # plot slices
    # ----------------------------
    for i, idx in enumerate(slice_range):

        ax = axes[i]

        if slice_axis == 'x':
            ul_slice = underlay[idx, :, :]
            ol_slice = overlay[idx, :, :] if overlay is not None else None

        elif slice_axis == 'y':
            ul_slice = underlay[:, idx, :]
            ol_slice = overlay[:, idx, :] if overlay is not None else None

        else:
            ul_slice = underlay[:, :, idx]
            ol_slice = overlay[:, :, idx] if overlay is not None else None

        ax.imshow(
            ul_slice,
            cmap=ul_cmap,
            vmin=ul_vmin,
            vmax=ul_vmax,
            interpolation='nearest'
        )

        if overlay is not None:

            ax.imshow(
                ol_slice,
                cmap=cmap,
                norm=norm,
                interpolation=interpol,
                alpha=ol_alpha
            )

        ax.set_title(f"Slice {idx}")
        ax.axis("off")

    # ----------------------------
    # overview slice
    # ----------------------------
    ax = axes[nslices]

    if slice_axis == 'x':
        overview = underlay[:, :, underlay.shape[2] // 2]
        ax.imshow(overview, cmap=ul_cmap)
        for idx in slice_range:
            ax.axvline(idx, color='red', linestyle='--')

    elif slice_axis == 'y':
        overview = underlay[:, underlay.shape[1] // 2, :]
        ax.imshow(overview, cmap=ul_cmap)
        for idx in slice_range:
            ax.axhline(idx, color='red', linestyle='--')

    else:
        overview = underlay[underlay.shape[0] // 2, :, :]
        ax.imshow(overview, cmap=ul_cmap)
        for idx in slice_range:
            ax.axvline(idx, color='red', linestyle='--')

    # ax.set_title("Overview")
    ax.axis("off")

    # ----------------------------
    # hide unused axes
    # ----------------------------
    for j in range(nplots, len(axes)):
        axes[j].axis('off')

    # ----------------------------
    # colorbar
    # ----------------------------
    if overlay is not None:

        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=norm
        )

        sm.set_array([])

        fig.colorbar(
            sm,
            ax=axes[:nplots],
            fraction=0.02,
            pad=0.02
        )

    plt.tight_layout()

    if save_img_path:
        plt.savefig(save_img_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()


def imgview_scroll(
    underlay,
    overlay=None,
    slice_axis="z",
    ul_cmap="Greys_r",
    ol_cmap="RdBu_r",
    ol_thresh=None,
    ol_vmax=None,
    alpha=0.9,
):

    axis_dict = {"x":0,"y":1,"z":2}
    axis = axis_dict[slice_axis]

    dim = underlay.shape[axis]
    slice_idx = dim // 2

    # underlay contrast
    ul_vmin = underlay.min()
    ul_vmax = underlay.max()

    brightness = 0
    contrast = 1
    rotation = 0

    # overlay prep
    if overlay is not None:

        if ol_thresh is not None:
            overlay = np.ma.masked_where(np.abs(overlay) < ol_thresh, overlay)

        if ol_vmax is None:
            vmax = np.nanmax(np.abs(overlay))
        else:
            vmax=ol_vmax

        norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        cmap = plt.get_cmap(ol_cmap).copy()
        cmap.set_bad(alpha=0)

    def get_slice(vol, idx):

        if axis == 0:
            sl = vol[idx,:,:]
        elif axis == 1:
            sl = vol[:,idx,:]
        else:
            sl = vol[:,:,idx]

        return np.rot90(sl, rotation)

    # crosshair position
    cross_x = underlay.shape[1]//2
    cross_y = underlay.shape[0]//2

    fig, ax = plt.subplots(figsize=(6,6))

    ul_im = ax.imshow(get_slice(underlay, slice_idx),
                      cmap=ul_cmap,
                      vmin=ul_vmin,
                      vmax=ul_vmax)

    if overlay is not None:

        ol_im = ax.imshow(get_slice(overlay, slice_idx),
                          cmap=cmap,
                          norm=norm,
                          alpha=alpha)

        sm = plt.cm.ScalarMappable(norm=norm,cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm,ax=ax)

    hline = ax.axhline(cross_y,color="yellow",lw=1)
    vline = ax.axvline(cross_x,color="yellow",lw=1)

    ax.axis("off")

    dragging = False
    start_x = None
    start_y = None


    def redraw():

        ul_im.set_data(get_slice(underlay,slice_idx))

        if overlay is not None:
            ol_im.set_data(get_slice(overlay,slice_idx))

        ul_im.set_clim(ul_vmin*contrast+brightness,
                       ul_vmax*contrast+brightness)

        ax.set_title(f"axis={['x','y','z'][axis]}  slice={slice_idx}")
        fig.canvas.draw_idle()


    # scroll slices
    def on_scroll(event):

        nonlocal slice_idx

        if event.button == "up":
            slice_idx = min(slice_idx+1,dim-1)
        else:
            slice_idx = max(slice_idx-1,0)

        redraw()


    # keyboard navigation
    def on_key(event):

        nonlocal slice_idx, axis, dim, rotation
    
        if event.key == "j":
            slice_idx = max(slice_idx-1,0)
    
        elif event.key == "k":
            slice_idx = min(slice_idx+1,dim-1)
    
        elif event.key in ["x","y","z"]:
            axis = axis_dict[event.key]
            dim = underlay.shape[axis]
            slice_idx = dim//2
    
        elif event.key == "r":
            rotation = (rotation + 1) % 4
    
        elif event.key == "R":
            rotation = (rotation - 1) % 4
    
        redraw()


    # mouse click (move crosshair)
    def on_click(event):

        nonlocal cross_x, cross_y, dragging, start_x, start_y
    
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
    
            cross_x = int(event.xdata)
            cross_y = int(event.ydata)
    
            vline.set_xdata([cross_x, cross_x])
            hline.set_ydata([cross_y, cross_y])
    
            fig.canvas.draw_idle()
    
        elif event.button == 3:
    
            dragging = True
            start_x = event.x
            start_y = event.y

    def on_release(event):

        nonlocal dragging
        dragging = False

    # contrast adjustment
    def on_motion(event):

        nonlocal brightness,contrast,start_x,start_y

        if not dragging:
            return

        dx = event.x-start_x
        dy = event.y-start_y

        contrast += dx*0.005
        brightness += dy*0.5

        start_x = event.x
        start_y = event.y

        redraw()


    fig.canvas.mpl_connect("scroll_event",on_scroll)
    fig.canvas.mpl_connect("key_press_event",on_key)
    fig.canvas.mpl_connect("button_press_event",on_click)
    fig.canvas.mpl_connect("button_release_event",on_release)
    fig.canvas.mpl_connect("motion_notify_event",on_motion)

    redraw()

    plt.show()