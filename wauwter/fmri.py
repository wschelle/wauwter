#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 14:00:08 2022

@author: wousch
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import datetime
import nibabel as nib
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz
import copy
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from Python.python_scripts.hrf import spm_hrf_compat,lambdify_t,glover,T

def gloverhrf(hrflen,timestep):
    #from nipy.modalities.fmri import hrf, utils
    #hrf_func = utils.lambdify_t(hrf.glover(utils.T))
    hrf_func = lambdify_t(glover(T))
    t = np.arange(0,hrflen,timestep)
    hrf=hrf_func(t)
    hrf/=np.max(hrf)
    return(hrf)

def gammahrf(hrflen,timestep):
    #from nipy.modalities.fmri.hrf import spm_hrf_compat
    t = np.arange(0,hrflen,timestep)
    hrf=spm_hrf_compat(t, peak_delay=5.5, under_delay=12, peak_disp=1, under_disp=1, p_u_ratio=2.5, normalize=True)
    #hrf=spm_hrf_compat(t, peak_delay=6, under_delay=16, peak_disp=1, under_disp=1, p_u_ratio=6, normalize=True)
    hrf/=np.max(hrf)
    return(hrf)

def doublegammahrf(p0,hrflen,timestep):
    #p0=np.array([6,16,1,1,6,0,1])
    t = np.arange(0,hrflen,timestep)
    hrf=spm_hrf_compat(t, peak_delay=p0[0], under_delay=p0[1], peak_disp=p0[2], under_disp=p0[3], p_u_ratio=p0[4], normalize=True)
    hrf/=np.max(hrf)
    hrf*=p0[6]
    hrf+=p0[5]
    return(hrf)

def hrf_convolve(onsets,maxtime,TR=1,upsample_factor=10,glover_hrf=False,normalize_f=True):
    nr_factors=int(np.max(onsets[:,0])+1)
    nr_events=int(onsets.shape[0])
    fmat=np.zeros([nr_factors,int(maxtime*upsample_factor)])
    if glover_hrf:
        hrf=gloverhrf(25,1/upsample_factor)
    else:
        hrf=gammahrf(25,1/upsample_factor)
    fmat_conv=np.zeros([nr_factors,int(maxtime*upsample_factor+len(hrf)-1)])
    fmat_conv_intp=np.zeros([nr_factors,int(np.round(maxtime/TR))])
    if len(onsets[0,:]==3):
        onsets=np.c_[ onsets, np.ones(nr_events) ]
    for i in range(nr_events):
        fmat[int(onsets[i,0]),int(np.round(onsets[i,1]*upsample_factor)):int(np.round((onsets[i,1]+onsets[i,2])*upsample_factor))]=onsets[i,3]
    realtime=np.arange(0,maxtime,1/upsample_factor)
    scantime=np.arange(0,maxtime,TR)
    if scantime[-1]>=maxtime: scantime=scantime[:-1]
    for i in range(nr_factors):
        fmat_conv[i,:]=np.convolve(fmat[i,:], hrf)
        if (normalize_f)&(np.max(fmat_conv[i,:])!=0):fmat_conv[i,:]/=np.max(fmat_conv[i,:])
        fcon = interp1d(realtime[0:int(maxtime*upsample_factor)],fmat_conv[i,0:int(maxtime*upsample_factor)],kind='cubic')
        fmat_conv_intp[i,:] = fcon(scantime)
    if not normalize_f: fmat_conv_intp/=np.max(fmat_conv_intp)
    return fmat_conv_intp

def hrf_convolve_pm(onsets,maxtime,TR=1,upsample_factor=10,glover_hrf=True,normalize_f=True):
    nr_events=int(onsets.shape[0])
    nr_pm=int(onsets.shape[1]-3)
    nr_factors=int(np.max(onsets[:,0])+1+nr_pm)
    fmat=np.zeros([nr_factors,int(maxtime*upsample_factor)])
    if glover_hrf:
        hrf=gloverhrf(25,1/upsample_factor)
    else:
        hrf=gammahrf(25,1/upsample_factor)
    fmat_conv=np.zeros([nr_factors,int(maxtime*upsample_factor+len(hrf)-1)])
    fmat_conv_intp=np.zeros([nr_factors,int(np.round(maxtime/TR))])
    if len(onsets[0,:]==3):
        onsets=np.c_[ onsets, np.ones(nr_events) ]
    for i in range(nr_events):
        fmat[int(onsets[i,0]),int(np.round(onsets[i,1]*upsample_factor)):int(np.round((onsets[i,1]+onsets[i,2])*upsample_factor))]=onsets[i,-1]
    for i in range(nr_pm):
        for j in range(nr_events):
            fmat[int(np.max(onsets[:,0])+1+i),int(np.round(onsets[j,1]*upsample_factor)):int(np.round((onsets[j,1]+onsets[j,2])*upsample_factor))]=onsets[j,3+i]
    realtime=np.arange(0,maxtime,1/upsample_factor)
    scantime=np.arange(0,maxtime,TR)
    if scantime[-1]>=maxtime: scantime=scantime[:-1]
    for i in range(nr_factors):
        fmat_conv[i,:]=np.convolve(fmat[i,:], hrf)
        if (normalize_f)&(np.max(fmat_conv[i,:])!=0):fmat_conv[i,:]/=np.max(fmat_conv[i,:])
        fcon = interp1d(realtime[0:int(maxtime*upsample_factor)],fmat_conv[i,0:int(maxtime*upsample_factor)],kind='cubic')
        fmat_conv_intp[i,:] = fcon(scantime)
    if not normalize_f: fmat_conv_intp/=np.max(fmat_conv_intp)
    return fmat_conv_intp

def cosfilt(nf,time):
    fm=np.zeros([nf,time])
    for i in range(nf):
        fm[i,:]=np.cos(np.arange(time)*(i+1)*math.pi/time)
    return(fm)

def dispmat(matr):
    msize=matr.shape
    fig, axes = plt.subplots(msize[0], 1, figsize=(msize[0],msize[0]*2))
    for i in range(msize[0]):
        axes[i].plot(matr[i,:])
    plt.show()

def loadmp(mpfile,csv=None):
    if csv:
        mp=np.genfromtxt(mpfile,delimiter=',')
    else:
        mp=np.loadtxt(mpfile)
        
    mp=mp.T
    return(mp)

def hpfilt(dat,tr,cut,addfilt=0,mask=0,convperc=0,showfiltmat=True):
    #reg = LinearRegression()
    datsize=dat.shape
    nfac=datsize[0]
    ntime=datsize[1]
    dat2=np.zeros([int(nfac),int(ntime)],dtype=np.float32)
    nfilt=int(np.floor(2*(ntime*(tr/1.)*cut)))
    fm=cosfilt(nfilt,int(ntime))
    yhat=np.zeros(dat.shape,dtype=np.float32)
    if np.sum(addfilt) != 0:
        fm=np.concatenate([fm,addfilt])
        nfilt+=addfilt.shape[0]
    if np.sum(mask) == 0:
        mask=np.ones(nfac)
    for i in range(nfilt):
        fm-=np.mean(fm[i,:])
    fm=np.vstack((fm,np.ones(ntime)))
    nfilt+=1
    if showfiltmat:
        dispmat(fm)
    beta=np.zeros([nfac,nfilt],dtype=np.float32)
    fm=fm.T
    for i in tqdm(range(nfac)):
        if mask[i] != 0:
            #reg.fit(fm.T, dat[i,:])
            beta[i,:] = np.linalg.inv(fm.T @ fm) @ fm.T @ dat[i,:]
            yhat[i,:] = fm @ beta[i,:]
            if convperc != 0:
                dat2[i,:]=(dat[i,:]-yhat[i,:])/beta[i,-1]*100
            else:
                dat2[i,:]=dat[i,:]-yhat[i,:]
    return(dat2)

def rval2tval(R,N):
    numerator = R * np.sqrt(N - 2)
    denominator = np.sqrt(1 - R**2)
    t = numerator / denominator
    return(t)

def mult_R2(designmatrix):
    mcc=np.zeros(designmatrix.shape[0],dtype=np.float32)
    clist=np.arange(designmatrix.shape[0],dtype=np.int16)
    reg = LinearRegression()
    for i in range(designmatrix.shape[0]):
        reg.fit(designmatrix[clist[clist != i],:].T, designmatrix[i,:])
        tmp = reg.predict(designmatrix[clist[clist != i],:].T)
        mcc[i] = r2_score(designmatrix[i,:],tmp)
    return mcc

def mreg(dm,dat,mask):
    dmsize=dm.shape
    datsize=dat.shape
    betas=np.zeros([datsize[0],dmsize[0]+1],dtype=np.float32)
    msres=np.zeros([datsize[0]],dtype=np.float32)
    reg = LinearRegression()
    for i in range(dmsize[0]):
        dm[i,:]-=np.mean(dm[i,:])
    for i in tqdm(range(datsize[0])):
        if mask[i] != 0:
            reg.fit(dm.T,dat[i,:])
            betas[i,0:dmsize[0]]=reg.coef_
            betas[i,dmsize[0]]=reg.intercept_
            msres[i]=(np.sum((dat[i,:]-reg.predict(dm.T))**2))/(datsize[1]-dmsize[0]-1)
    return(betas,msres)

def GLM(X, Y, mask, norm_X=True, add_const=True, beta_only=False, betaresid_only=True):
    if norm_X:
        for i in range(X.shape[0]):
            X-=np.mean(X[i,:])
    if add_const:
        X=np.vstack((X,np.ones(X.shape[1])))
        
    beta=np.zeros([Y.shape[0],X.shape[0]],dtype=np.float32)
    yhat=np.zeros(Y.shape,dtype=np.float32)
    
    X=X.T #because my X is normally of shape [factor,time], which is transposed here

    for i in tqdm(range(Y.shape[0])):
        if mask[i] != 0:
            beta[i,:] = np.linalg.inv(X.T @ X) @ X.T @ Y[i,:]
            yhat[i,:] = X @ beta[i,:]
        
    residuals=Y-yhat
    msres=np.sum(residuals**2,axis=1)/(X.shape[0]-X.shape[1])
    
    if beta_only:
        return beta
    elif betaresid_only:
        return beta,msres
    else:
        return beta,msres,yhat
    
def GLM_ridge(
    X,
    Y,
    mask=None,
    frac=None,
    norm_X=True,
    add_const=True,
    beta_only=False,
    betaresid_only=True
):
    """
    Ridge GLM with voxel-specific fractional ridge.

    X: [factors,time]
    Y: [voxels,time]
    mask: [voxels]
    frac:
        None -> OLS
        scalar -> same ridge for all voxels
        array [voxels] -> voxel-specific ridge fraction
    """

    X = X.copy()

    if norm_X:
        for i in range(X.shape[0]):
            X[i, :] -= np.mean(X[i, :])

    if add_const:
        X = np.vstack((X, np.ones(X.shape[1])))

    # transpose to time x regressors
    X = X.T

    ntime, nreg = X.shape

    XtX = X.T @ X
    XtY = X.T @ Y.T   # regress all voxels simultaneously

    # OLS
    if frac is None:
        beta = np.linalg.solve(
            XtX,
            XtY
        ).T

    else:
        
        if mask is None:
            mask = np.ones(Y.shape[0],dtype=np.int16)

        # ridge scaling
        lambda_max = np.trace(XtX) / nreg

        if np.isscalar(frac):
            lambdas = np.ones(Y.shape[0]) * frac * lambda_max
        else:
            lambdas = frac * lambda_max

        beta = np.zeros((Y.shape[0], nreg), dtype=np.float32)

        I = np.eye(nreg)

        for v in tqdm(range(Y.shape[0])):

            if mask[v] != 0:

                lam = lambdas[v]

                beta[v, :] = np.linalg.solve(
                    XtX + lam * I,
                    XtY[:, v]
                )

    # predictions
    yhat = X @ beta.T

    residuals = Y - yhat.T

    dof = ntime - nreg

    msres = np.sum(residuals**2, axis=1) / dof

    if beta_only:
        return beta

    elif betaresid_only:
        return beta, msres

    else:
        return beta, msres, yhat.T
    
def GLS(X, Y, mask, yhat_0, norm_X=True, add_const=True, beta_only=False, betaresid_only=True):
    if norm_X:
        for i in range(X.shape[0]):
            X-=np.mean(X[i,:])
            
    if add_const:
        X=np.vstack((X,np.ones(X.shape[1])))
    
    beta=np.zeros([Y.shape[0],X.shape[0]],dtype=np.float32)
    yhat=np.zeros(Y.shape,dtype=np.float32)
    phi=np.zeros(Y.shape[0],dtype=np.float32)
    
    X=X.T
    
    tpl=toeplitz(np.arange(Y.shape[1]))
    tpl=tpl.astype(np.int32)
    
    for i in tqdm(range(Y.shape[0])):
        if mask[i] != 0:
            res0 = Y[i,:] - yhat[i,:]
            res1 = np.roll(res0,1)
            phi[i] = (res0 - res0.mean()) @ (res1 - res1.mean()) / np.sqrt(np.sum((res0 - res0.mean()) ** 2) * np.sum((res1 - res1.mean()) ** 2))
    
    for i in tqdm(range(Y.shape[0])):
        if mask[i] != 0:
            V = phi[i] ** tpl
            V = np.linalg.inv(V)
            beta[i,:] = np.linalg.inv(X.T @ V @ X) @ X.T @ V @ Y[i,:]
            yhat[i,:] = X @ beta[i,:]
    
    residuals = Y - yhat
    msres=np.sum(residuals**2,axis=1)/(X.shape[0]-X.shape[1])
    
    if beta_only:
        return beta
    elif betaresid_only:
        return beta,msres
    else:
        return beta,msres,yhat

def tcon(contrast,dm,betas,msres,mask):
    bsize=betas.shape
    dmsize=dm.shape
    if len(dmsize)==2:
        ntime=dmsize[1]
        nfac=dmsize[0]
    else:
        ntime=dmsize[0]
        nfac=1
    c=np.zeros([nfac+1,1])
    c[:,0]=contrast
    tval=np.zeros(bsize[0])
    t1=np.zeros([nfac+1,1])
    #term2=transpose(c)#invert(x#transpose(x))#c
    t2=c[0:nfac,0].T @ np.linalg.inv(dm @ dm.T) @ c[0:nfac,0]
    for i in tqdm(range(bsize[0])):
        if mask[i] != 0:
            t1[:,0]=betas[i,:]
            tval[i]=c.T @ t1 / (np.sqrt(msres[i]*t2))
    return(tval)

def write_metric(filename,dat,metrictags=['','']):
    datsize=dat.shape
    nv=datsize[0]
    nc=datsize[1]
    intval=10
    r=intval.to_bytes(1,'little')
    r=r.decode("utf-8") 
    header='BeginHeader'+r+'Caret-Version 5.65'+r+'comment '+r+'date '+str(datetime.datetime.now())+r+'encoding BINARY'+r+'pubmed_id '+r+'EndHeader'+r+'tag-version 2'+r
    header=header+'tag-number-of-nodes '+str(nv)+r
    header=header+'tag-number-of-columns '+str(nc)+r
    header=header+'tag-title '+r
    if metrictags is ['','']:
        for i in range(nc):
            header=header+'tag-column-name '+str(i)+' Volume '+str(i)+r
    else:
        for i in range(nc):
            header=header+'tag-column-name '+str(i)+' '+metrictags[i]+r
    for i in range(nc):
        header=header+'tag-column-comment '+str(i)+' Made by Wauwter with Python'+r
    for i in range(nc):
        header=header+'tag-column-study-meta-data '+str(i)+r
    for i in range(nc):
        header=header+'tag-column-color-mapping '+str(i)+' -1.000000 1.000000'+r
    for i in range(nc):
        header=header+'tag-column-threshold '+str(i)+' 0.000000 0.000000'+r
    for i in range(nc):
        header=header+'tag-column-average-threshold '+str(i)+' 0.000000 0.000000'+r
    header=header+'tag-BEGIN-DATA'+r
    dat=dat.astype('float32')
    #dat=dat.T
    #dat.byteswap(inplace=True)
    dat=dat.newbyteorder("<")
    #dat=bytearray(dat)
    binary_file = open(filename, "wb")
    binary_file.write(bytes(header, 'utf-8'))
    #dat.tofile(filename)
    #binary_file.write(dat)
    binary_file.write(dat.tobytes())
    binary_file.close()
    
def write_mgh(filename,dat):
    datsize=dat.shape
    if len(datsize)==1:
        wdat=np.zeros([datsize[0],1,1],dtype='float32')
        wdat[:,0,0]=dat
    elif len(datsize)==2:
        wdat=np.zeros([datsize[0],1,datsize[1]],dtype='float32')
        wdat[:,0,:]=dat
    wdat=nib.freesurfer.mghformat.MGHImage(wdat,None)
    nib.save(wdat,filename)

def write_nii(filename,dat,aff):
    wdat=nib.nifti1.Nifti1Image(dat, aff)
    nib.save(wdat,filename)

def prf_est(dm,dat,mask):
    nfac=dm.shape[0]
    nv=dat.shape[0]
    reg = LinearRegression()
    prfc=np.zeros(nv)
    prfs=np.zeros(nv)
    prfa=np.zeros(nv)
    prfi=np.zeros(nv)
    prfr=np.zeros(nv)
    fwhm=2*np.sqrt(2*np.log(2))
    for i in range(nfac):
        dm[i,:]-=np.mean(dm[i,:])
    
    be,re,yh=GLM(dm,dat,mask,betaresid_only=False)
    co=np.squeeze(be[:,-1])
    be=be[:,:-1]
    
    for i in tqdm(range(nv)):
        if mask[i] != 0:
            prfa[i]=np.max(be[i,:])
            prfi[i]=co[i]
            prfc[i]=np.where(be[i,:] == np.max(be[i,:]))[0][0]
            prfs[i]=np.sum(be[i,:] > prfa[i]/2)/fwhm
            prfr[i]=np.corrcoef(dat[i,:],yh[i,:])[0][1]**2
    return(prfc,prfs,prfa,prfi,prfr)

def lmfit_prf(params, dm, ydata):
    cons = params['cons'].value
    ampl = params['ampl'].value
    center = params['center'].value
    sigma = params['sigma'].value
    
    ymodel=np.zeros(dm.shape)
    for i in range(dm.shape[0]):
        ymodel[i,:]=dm[i,:] * np.exp(-(i-center)**2 / (2*sigma**2))
    ymodel=np.sum(ymodel,axis=0)
    ymodel*=ampl
    ymodel+=cons
    return (ymodel - ydata)

def prf2d(designmatrix,p0,xgrid,ygrid):
    xs=designmatrix.shape
    x2=np.zeros(xs,dtype=np.float32)
    # xdiff = np.arange(xs[0],dtype=np.float32)
    # ydiff = np.arange(xs[1],dtype=np.float32)
    # xdiff, ydiff = np.meshgrid(xdiff, ydiff)
    xdiff=xgrid-p0[2]
    ydiff=ygrid-p0[3]
    xdiff**=2
    ydiff**=2
    xdiff/=(p0[4]**2)
    ydiff/=((p0[4]/p0[5])**2)
    num=xdiff+ydiff
    expo=np.exp(-num/2)
    x2 = np.tensordot(designmatrix,expo, axes=([0,1],[0,1]))
    x2/=np.nanmax(x2)
    x2[np.isnan(x2)]=0
    x2*=p0[1]
    x2+=p0[0]
    return(x2)

def lmfit_prf2d(params,designmatrix,ydata,xgrid,ygrid):
    p0=np.zeros(6,dtype=np.float32)
    p0[0]=params['cons'].value
    p0[1]=params['amp'].value
    p0[2]=params['centerX'].value
    p0[3]=params['centerY'].value
    p0[4]=params['XYsigma'].value
    p0[5]=params['XYsigmaRatio'].value
    ymodel=prf2d(designmatrix,p0,xgrid,ygrid)
    return(ymodel-ydata)
    
def make_fir(fironset,firlength,maxtime):
    firmatrix=np.zeros([int(np.max(fironset[:,0])+1)*firlength,maxtime])
    for i in range(len(fironset[:,0])):
        for j in range(firlength):
            if fironset[i,1]+j < maxtime:
                firmatrix[fironset[i,0]*firlength+j,fironset[i,1]+j]=1
    firmatrix=firmatrix[np.sum(firmatrix,axis=1)>1,:]
    #dispmat(firmat4[0::4,:])
    return firmatrix

def calc_fir(firmatrix,datamatrix,mask):
    b=GLM(firmatrix,datamatrix,mask,norm_X=False,beta_only=True)
    return b[:,:-1]

def gaussian1D(xrange,const,amplitude,center,sigma):
    numerator=(xrange-center)**2
    denominator=2*sigma**2
    gauss=amplitude*np.exp(-numerator/denominator)+const
    return gauss

def nrprf_est(beta_matrix,nrprf_center=3.5,nrprf_sigma=1,stepsize=100):
    gauss_pos=gaussian1D(np.arange(-nrprf_center,nrprf_center,1/stepsize),0,1,nrprf_center,nrprf_sigma)
    gauss_neg=gaussian1D(np.arange(-nrprf_center,nrprf_center,1/stepsize),0,-1,-nrprf_center,nrprf_sigma)
    gauss=gauss_pos+gauss_neg
    gauss/=np.max(gauss)
    nr_factors=beta_matrix.shape[1]-1
    nr_datapoints=beta_matrix.shape[0]
    ordinal_regress=np.zeros([nr_datapoints,nr_factors])
    nrprf_matrix=np.zeros([nr_datapoints,nr_factors+4])
    for i in tqdm(range(nr_datapoints)):
        if np.std(beta_matrix[i,:])!=0:
            ordinal_regress[i,:]=copy.deepcopy(beta_matrix[i,:-1])
            ordinal_regress[i,:]/=np.max(ordinal_regress[i,:])
            ordinal_regress[i,ordinal_regress[i,:]<gauss[0]]=gauss[0]
            nrprf_matrix[i,0]=nrprf_center
            nrprf_matrix[i,1]=nrprf_sigma
            nrprf_matrix[i,2]=np.max(beta_matrix[i,:-1])
            nrprf_matrix[i,3]=beta_matrix[i,-1]
            for j in range(nr_factors):
                nrprf_matrix[i,j+4]=np.max(np.where(ordinal_regress[i,j] >= gauss))
    nrprf_matrix[:,4:]/=stepsize
    return nrprf_matrix

def nrprf(params,dm):
    ymodelpos=np.zeros(dm.shape)
    ymodelneg=np.zeros(dm.shape)
    for i in range(dm.shape[0]):
        ymodelpos[i,:]=dm[i,:] * np.exp(-(params['p'+str(i)].value - (params['center'].value*2))**2 / (2*params['sigma'].value**2))
        ymodelneg[i,:]=dm[i,:] * np.exp(-(params['p'+str(i)].value - 0)**2 / (2*params['sigma'].value**2))
    ymodelpos=np.sum(ymodelpos,axis=0)    
    ymodelneg=np.sum(ymodelneg,axis=0)
    ymodel=ymodelpos+(-ymodelneg)
    ymodel*=params['ampl'].value
    ymodel+=params['cons'].value
    return ymodel
    
def lmfit_nrprf(params, dm, ydata):
    ymodel=nrprf(params,dm)
    return (ymodel - ydata)

def nrprf_center_sigma(fitparams,paramstart=4,paramstop=8):
    fp=fitparams[:,paramstart:paramstop]
    centercurve=np.max(fitparams[:,0])
    sigmacurve=np.max(fitparams[:,1])
    hwhm=np.sqrt(2*np.log(2))*sigmacurve
    fp2=copy.deepcopy(fp)
    fp2[fp < centercurve*2-hwhm]=0
    fp2/=centercurve*2
    centerpos=np.zeros(fp.shape[0],dtype=np.float32)
    sigmapos=np.zeros(fp.shape[0],dtype=np.float32)
    com=np.zeros(fp.shape[1],dtype=np.float32)
    for i in tqdm(range(fp.shape[0])):
        if np.std(fp2[i,:]) != 0:
            maxfit=np.where(fp2[i,:] == fp2[i,:].max())[0]
            if len(maxfit) == 1:
                centerpos[i] = maxfit + 1
            else:
                com*=0
                for j in range(fp.shape[1]):
                    com[j]+= (j+1)*fp2[i,j]
                    centerpos[i] = np.sum(com) / np.sum(fp2[i,:])
            sigmapos[i] = np.sum(fp2[i,:])*hwhm
    return centerpos,sigmapos

def lmfit_1DGauss(params,xrange,ydata):
    gauss=gaussian1D(xrange,params['cons'],params['ampl'],params['center'],params['sigma'])
    return(gauss-ydata)

def multiple_comparison(statistic,dfn=None,dfd=None,alpha=0.05,method='fdr_bh',test='t',tail=2,cutoff_only=True):
    stat0=statistic[statistic!=0]
    if test=='t':
        uncorr_pval = stats.t.sf(np.abs(stat0), dfn-1)*tail
    if test=='f':
        uncorr_pval = 1-stats.f.cdf(stat0, dfn, dfd)
    corr_pval=multipletests(uncorr_pval,alpha,method=method)[1]
    if ((cutoff_only) & (np.min(corr_pval[stat0>0])<alpha)):
        return np.min(stat0[(corr_pval<alpha) & (stat0>0)])
    else:
        return uncorr_pval,corr_pval

def goodness_of_fit_F(modelfit,data,mask,dof,chisq=None):
    #modelfit & data should be of the same size (datapoints,time)
    #mask is of shape datapoints
    #dof is 2-element list: [model DoF,ErrorDoF]
    #optional: provide chisq values of size datapoints
    #returns array size datapoints with F-statistics
    fval=np.zeros(mask.shape)
    msm=np.zeros(mask.shape)
    for i in tqdm(range(mask.shape[0])):
        if mask[i]!=0:
            msm[i]=np.sum((np.mean(data[i,:])-modelfit[i,:])**2)/dof[0]
    # if chisq==None:
    #     chisq=np.sum((data-modelfit)**2,axis=1)
    mse=chisq/dof[1]
    fval[mse!=0]=msm[mse!=0]/mse[mse!=0]
    return(fval)
    
def inverselog_hrf(timerange,hrfparams):
    # HRF params should be an array with following structure:
    # Amplitudes(A), temporal derivative (T), dispersion derivative (D) & constant (C):
    # [A1,T1,D1,A2,T2,D2,A3,T3,D3,C], example:
    # np.array([2.18, 3.26, 0.98, -2.35, 6.23, 2.27, 0.17, 18.26, 2.57, 0])
    # or:
    # [A1,T1,D1,A2,T2,D2,A3,T3,D3,A4,T4,D4,C], example:
    # np.array([-0.1, 0.0, 0.3, 1.4, 3., 0.8, -2.1, 10.0, 1.8, 0.8, 15.0, 1., 0.0])
    f1=(timerange-hrfparams[1])/hrfparams[2]
    f2=(timerange-hrfparams[4])/hrfparams[5]
    l1=1/(1+np.exp(-f1))
    l2=1/(1+np.exp(-f2))
    hrf = hrfparams[0]*l1 + hrfparams[3]*l2 
    if len(hrfparams)==7:
        hrf += hrfparams[6]
    elif len(hrfparams)==10:
        f3=(timerange-hrfparams[7])/hrfparams[8]
        l3=1/(1+np.exp(-f3))
        hrf += (hrfparams[6]*l3 + hrfparams[9])
    else:
        f4=(timerange-hrfparams[10])/hrfparams[11]
        l4=1/(1+np.exp(-f4))
        hrf += (hrfparams[9]*l4 + hrfparams[12])
    return(hrf)

def lmfit_ilhrf(params, timerange, ydata):
    if len(params) == 7:
        hrfparams=np.array([params['A1'].value,params['T1'].value,params['D1'].value,
                            params['A2'].value,params['T2'].value,params['D2'].value,params['C'].value])
    elif len(params) == 10:
        hrfparams=np.array([params['A1'].value,params['T1'].value,params['D1'].value,
                            params['A2'].value,params['T2'].value,params['D2'].value,
                            params['A3'].value,params['T3'].value,params['D3'].value,params['C'].value])
    else:
        hrfparams=np.array([params['A1'].value,params['T1'].value,params['D1'].value,
                            params['A2'].value,params['T2'].value,params['D2'].value,
                            params['A3'].value,params['T3'].value,params['D3'].value,
                            params['A4'].value,params['T4'].value,params['D4'].value,params['C'].value])
    hrf=inverselog_hrf(timerange,hrfparams)
    return (hrf - ydata)

# def gppi(designmatrix,seedtimeseries,contrasts=None):
#     if contrasts==None:
#         contrasts=np.ones(designmatrix.shape[0])
        
#     ppi_mat=copy.deepcopy(designmatrix)
#     for i in range(designmatrix.shape[0]):
#         #ppi_mat[i,:]-=np.min(ppi_mat[i,:])
#         ppi_mat[i,:]/=np.max(ppi_mat[i,:])
#         ppi_mat[i,:]-=0.5
#         ppi_mat[i,:]*=contrasts[i]
#         ppi_mat[i,:]*=seedtimeseries
        
#     ppi_mat=np.vstack([designmatrix,seedtimeseries,ppi_mat])
#     return(ppi_mat)
    
    
    
    
    