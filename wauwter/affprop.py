#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:34:37 2023

@author: WauWter
"""
import numpy as np
import copy

def aff_prop_a(A,R,lambda_damp):
    # 'Affinitity Propagation clustering'
    # 'Makes availability matrix'
    # 'Supply availability (A) and responsibility (R) matrix'
    # 'Keyword: lambda (default=0.5)'
    N=R.shape[1]
    A_update = copy.deepcopy(R)
    A_update[A_update < 0] = 0
    A_update[range(N),range(N)] = 0
    A_update = np.sum(A_update,axis=0)
    A_update += np.diagonal(R)
    A_update = np.tile(A_update,(N,1))
    
    R2 = copy.deepcopy(R)
    R2[R < 0] = 0
    A_update -= R2
    A_update[A_update > 0] = 0
    
    R2 = copy.deepcopy(R)
    R2[range(N),range(N)] = 0
    R2[R2 < 0] = 0
    A_update[range(N),range(N)] = np.sum(R2,axis=0)
    
    A = (1-lambda_damp)*A_update + lambda_damp*A
    return(A)


def aff_prop_r(S,A,R,lambda_damp,bit16):
    # 'Affinitity Propagation clustering'
    # 'Makes responsibility matrix'
    # 'Supply similarity (S), availability (A), and responsibility (R) matrix'
    # 'Keyword: lambda (default=0.5)'
    
    N=S.shape[1]
    SA=S+A
    SA[range(N),range(N)]=-np.inf
    
    #first_max=np.max(SA,axis=1)
    idx_max=np.argmax(SA,axis=1)
    SA[range(N),idx_max]=-np.inf
    second_max=np.max(SA,axis=1)
    del SA
    if bit16==1:
        maxmat=np.ones([N,N],dtype=np.float16)
    else:
        maxmat=np.ones([N,N],dtype=np.float32)
    maxmat[range(N),idx_max]=second_max
    R_update = S - maxmat
    R = (1 - lambda_damp) * R_update + lambda_damp * R
    return(R)
  
def aff_prop_s(M,preference,prefmultiply,bit16):
    # 'Affinitity Propagation clustering'
    # 'Makes similaritiy matrix'
    # 'supply matrix: [points,observations]'
    # 'Example: M=[100,3] for 100 points with 3 values each (e.g. x,y,z)'
    # 'Returns similarity matrix S (e.g. [100,100])'
    # 'Keyword: preference (default=median(S)), prefmultiply (default=1)'
    print('Making similarity matrix...')
    sizeM=M.shape
    if bit16==1:
        S=np.zeros([sizeM[0],sizeM[0]],dtype=np.float16)
    else:
        S=np.zeros([sizeM[0],sizeM[0]],dtype=np.float32)
        
    if len(sizeM)==1:
        tmp=np.tile(M,(sizeM[0],1))
        S+=(-(tmp-tmp.T)**2)
    else:
        for i in range(sizeM[1]):
            tmp=np.tile(M[:,i],(sizeM[0],1)).T
            S+=(-(tmp-tmp.T)**2)
    if preference == None:
        preference=np.median(S)
    preference*=prefmultiply
    S[np.arange(0,sizeM[0]),np.arange(0,sizeM[0])]=preference
    print('Done.')
    return(S)

def aff_prop(M,preference=None,prefmultiply=1,lambda_damp=0.5,maxiter=10,maxtries=300,verbose=False,bit16=0,simple_numbering=True):
    # 'Affinitity Propagation clustering for IDL, and for Python :)'
    # 'Supply matrix: [points,observations]'
    # 'Example: M=[100,3] for 100 points with 3 values each (e.g. x,y,z)'
    # 'Clustering occurs on the basis of the observations. Any number of observations allowed'
    # 'Returns an array of labels with the same size as number of points'
    # 'Each entry of the returned label array refers to the index of its exemplar'
    # 'Keyword: preference (default=median similarity). How likely points choose themself as exemplar. Ranges from [-inf,0]'
    # 'Keyword: prefmultiply (default=1). Scales the preference relative to the median'
    # 'Keyword: lambda_damp (default=0.5). Damping of the iterative updating routine (higher=slower)'
    # 'Keyword: maxiter (default=10). Max iterations of the same label outcome'
    # 'Keyword: maxtries (default=10000). Max nr of tries. Useful if stuck in loop'
    # 'Keyword: verb. Print label iteration and nr tries'
    # 'Keyword: bit16 (default=0). 16-bit instead of 32-bit precision'
    # 'Keyword: simple_numbering (default=True). Renames clusters from 1 to max cluster nr.'
    
    if bit16==1:
        M=M.astype(np.float16)
        
    S=aff_prop_s(M,preference,prefmultiply,bit16)
    
    if verbose:
        print('Median preference = '+str(S[0,0]/prefmultiply))
        print('Adjusted preference = '+str(S[0,0]))
    
    SM=M.shape
    R=copy.deepcopy(S)*0
    A=copy.deepcopy(S)*0
    labels=np.arange(0,SM[0])
    labels2=copy.deepcopy(labels)
    now=0
    ntries=0
    
    print('Starting convergence loop.')
    while ((now < maxiter) & (ntries < maxtries)):
        R=aff_prop_r(S,A,R,lambda_damp,bit16)
        A=aff_prop_a(A,R,lambda_damp)
        solut=A+R
        maxsolut=np.max(solut,axis=1)
        for i in range(SM[0]):
            tmp=np.where(solut[i,:]==maxsolut[i])
            labels[i]=np.median(tmp)
        if np.array_equal(labels,labels2):
            now+=1
        else:
            labels2=copy.deepcopy(labels)
            now=0
        ntries+=1
        if verbose:
            print('ITER: ',now)
            print('TRY: ',ntries)
    
    if simple_numbering:
        final_labels=np.zeros(labels.size,dtype=np.int64)
        ul=np.unique(labels)
        for idx,key in enumerate(ul):
            final_labels[labels==key]=idx+1
    else:
        final_labels=labels
    
    unique_labels=np.unique(final_labels)
    return final_labels, unique_labels
