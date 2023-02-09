from parameters import *
from helpers import get_spin_spin_factors
from jits import make_Hint, rho_t
from carray import CachedArray
import cupy as cp
import cupyx.scipy.fft as cufft
import scipy.fft
scipy.fft.set_global_backend(cufft)

#time_derivative
def d_dt(psi):
    psi_advanced=cp.roll(psi,-1,axis=time_axis)
    return (psi_advanced-psi)


#directly implementing Laplacian using fft is a lot faster (see benchmarks)
def laplacian_k(fields):
    return -cp.fft.ifftn(fields.get_psi_k()*laplacian_grid,axes=space_axes)
def laplacian_k_conj(fields):
    return -cp.fft.fftn(fields.get_psi_k_conj()*laplacian_grid,axes=space_axes)





factors_unique,inds_unique=get_spin_spin_factors(pauli_spin1)
#get individual arrays for each component because JIT demands it
in1,in2,in3=inds_unique
in1=cp.array(in1).astype(cp.uint32)
in2=cp.array(in2).astype(cp.uint32)
in3=cp.array(in3).astype(cp.uint32)
f1,f2,f3=factors_unique
f2=cp.array(f2)
f1=cp.array(f1)
f3=cp.array(f3)



 
                        
def calculate_drift(fields):
    psi=fields.get_psi()
    psi_conj=fields.get_psi_conj()
    psi_conj_adv=CachedArray.roll(psi_conj,-1,axis=time_axis)
    psi_ret=CachedArray.roll(psi,1,axis=time_axis)
    rho_p=rho_t(psi_conj,psi_ret)
    rho_m=rho_t(psi_conj_adv,psi)
    #intra-state-collisions are easy       
    #print(rho_p.shape)
    H_int=C0*psi_ret*rho_p
    H_int_conj=C0*psi_conj_adv*rho_m
    #cross-scattering and spin mixing
    ##jit compiled is around 30% faster but horrible to debug and also d=3 only (I will NOT write implementation for every dim)
    if comp==3:
        if dim==3:
            #for dim=3 we have JIT
            make_Hint[(8,8,8),(8,8,8)](psi_ret,psi_conj,H_int,in1,in2,in3,f1,f2,f3,C2)
            make_Hint[(8,8,8),(8,8,8)](psi_conj_adv,psi,H_int_conj,in1,in2,in3,f1,f2,f3,C2)
        else:
            ##python method is fast enough for lower dims
            for i in range(comp):
                for inds,f in zip(inds_unique[i],factors_unique[i]):
                    H_int[i]     +=f*C2*psi_conj[inds[0]]*(psi_ret[inds[1]]*psi_ret[inds[2]])
                    H_int_conj[i]+=f*C2*psi[inds[0]]*(psi_conj_adv[inds[1]]*psi_conj_adv[inds[2]])
    drift=(-d_dt(psi_ret)+EPS*(laplacian_k(fields)+MUBAR*psi_ret-H_int))
    drift_conj=(d_dt(psi_conj)+EPS*(laplacian_k_conj(fields)+MUBAR*psi_conj_adv-H_int_conj))
    return drift,drift_conj
k_eps=1e-8

ADAPT_STEP=False
def time_step(fields):
    global k_max_avg
    drift,drift_conj=calculate_drift(fields)
    #implementation of adaptive step size from https://arxiv.org/pdf/0912.0617.pdf
    #modified for moving average (idk why)
    if ADAPT_STEP:
        k_max=max([cp.amax(cp.abs(drift)),cp.amax(cp.abs(drift_conj))])
        if k_max_avg==0:
            k_max_avg=k_max
        else:
            k_max_avg=BETA*k_max_avg+(1-BETA)*k_max
        dt_n=min(k_max_avg/(k_max+k_eps)*DT+k_eps,3*DT)
    else:
        dt_n=DT
    #noise term
    noise=cp.random.normal(0.0,1.0,size=dims+[2],dtype=cp.float32).view(cp.complex64).squeeze()
    noise_conj=cp.conj(noise)
    delta=dt_n*drift+cp.sqrt(dt_n)*noise
    delta_conj=dt_n*drift_conj+cp.sqrt(dt_n)*noise_conj
    return fields.get_psi()+delta,fields.get_psi_conj()+delta_conj,dt_n


class Fields:
    def __init__(self, load_file=None, init_mean_field=True):
        if load_file!=None:
            self.psi_conj=cp.load("psi_conj"+load_file+".npy")
            self.psi=cp.load("psi"+load_file+".npy")
        else:
            self.psi=cp.zeros(dims)+0.0j+cp.random.random(dims)*1e-8
            self.psi_conj=cp.zeros(dims)+0.0j+cp.random.random(dims)*1e-8
            if init_mean_field and comp==3 and 0<qbar<1:
                self.psi[1]+=rho0
                self.psi_conj[1]+=rho0
                self.psi[0]+=rho1
                self.psi_conj[0]+=rho1+1e-6
                self.psi[2]+=rho1
                self.psi_conj[2]+=rho1-1e-6
        self.psi=self.psi.view(CachedArray)
        self.psi_conj=self.psi_conj.view(CachedArray)
        self.psi_k=cp.zeros(dims)+0.0j+cp.random.random(dims)*1e-8
        self.psi_k_conj=cp.zeros(dims)+0.0j+cp.random.random(dims)*1e-8
        self.k_is_known=False
    def get_psi_k(self):
        if not self.k_is_known:
            self.psi_k=cp.fft.fftn(self.psi, axes=space_axes).view(CachedArray)
            self.psi_k_conj=cp.fft.ifftn(self.psi_conj,axes=space_axes).view(CachedArray)
            self.k_is_known=True
        return self.psi_k
    def get_psi_k_conj(self):
        if not self.k_is_known:
            self.psi_k=cp.fft.fftn(psi, axes=space_axes).view(CachedArray)
            self.psi_k_conj=cp.fft.ifftn(psi_conj,axes=space_axes).view(CachedArray)
            self.k_is_known=True
        return self.psi_k_conj
    def get_psi(self):
        return self.psi
    def get_psi_conj(self):
        return self.psi_conj
    def evolve(self):
        self.k_is_known=False
        self.psi,self.psi_conj,dt_n=time_step(self)
        if cp.any(cp.isnan(self.psi)):
            raise ValueError("Psi became NaN, please adjust Parameters")
        return dt_n
    def save(self):
        cp.save("psi.npy",self.psi)
        cp.save("psi_conj.npy",self.psi_conj)