import numpy as np
import cupy as cp
from parameters import *
from carray import CachedArray
def get_spin_spin_factors(pauli_spin):
    ##The following static code gets the possible spin exchange (C2) collisions and multiplicative factors in Spin 1 gas directly from the representation and lists their derivatives for use in CL
    #it was a mistake. i should have just hardcoded it, only advantage is quicker generalization to spin 2 maybe

    #this is the full collsion tensor in H_int=psi_i*psi_j*F_ijkl*psi_conj_k*psi_conj_l. 
    F_ijkl=cp.einsum("ajk,ail->ijkl",pauli_spin,pauli_spin)
    F_ijkl[cp.abs(F_ijkl)<1e-8]=0.0
    #since most elements are 0, as full tensor contraction to get derivatives would be very inefficient. 
    #Instead, we extract the nonzero indices and values in a "sparse tensor" style

    #Get all nonzero ijkl pairs
    inds,values=cp.array(cp.nonzero(F_ijkl)),F_ijkl[cp.nonzero(F_ijkl)]
    #Derivatives can be taken wrt the psi only (exchange between psi and psi_conj is symmetric)
    #we remove each of the first two elements, and by that get two lists of "derivative by" and "resulting term"
    #if we resum the terms with the same "derivative by", we will get the full derivative of H_int by this term
    d1=inds[0]
    d2=inds[1]
    ds=cp.concatenate((d1,d2))
    ms1=inds[(1,2,3),].T
    ms2=inds[(0,2,3),].T
    ms=np.concatenate((ms1,ms2))

    sum_inds=[[],[],[]]
    sum_vals=[[],[],[]]
    for d,m,v in zip(ds,ms,values):
        m[-2:].sort()#psi_conj order doesnt matter
        sum_inds[d.get()].append(m.get())
        sum_vals[d.get()].append(v.get())

    #get all the unique index groups  and sum their multiplicators up, then remove the 0 ones
    inds_unique=[]
    factors_unique=[]

    for s,v in zip(sum_inds,sum_vals):
        unique,unique_inv=np.unique(s,return_inverse=True,axis=0)
        f=np.zeros(len(unique))+0.0j
        #print(len(unique_inv),len(v))
        for u,mult in zip(unique_inv,v):
            f[u]+=mult    
        f[np.abs(f)<1e-8]=0.0    
        unique_nonzero=unique[np.abs(f)>1e-8]
        f_nonzero=f[np.abs(f)>1e-8]
        factors_unique.append(cp.array(f_nonzero))
        inds_unique.append(unique_nonzero)
    return factors_unique, inds_unique





    
def k_to_minus_k(psi_k, s_axes=space_axes):
    return CachedArray.roll(cp.flip(CachedArray.roll(psi_k,-1-d_space//2,axis=s_axes),axis=s_axes),d_space//2,axis=s_axes)
def anom_product(t,s_axes):
    return t*k_to_minus_k(t, s_axes)