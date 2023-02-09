from parameters import *
from jits import *
from carray import CachedArray
from helpers import k_to_minus_k, anom_product
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy import optimize
import cupyx.scipy.fft as cufft
import scipy.fft
scipy.fft.set_global_backend(cufft)
##transformations

#transform psi and psi_k into the quasiparticle operators that diagonalize the hamiltonian
#this mode is the easiest to implement, and is also gapless
#see https://arxiv.org/pdf/0912.0355.pdf for all transformations
#define some variables first
eps_k=laplacian_grid
E_k_fz=cp.sqrt(eps_k*(eps_k+ZE_Q))+1e-8
fz_pre_1=cp.sqrt((eps_k+ZE_Q/2+E_k_fz)/(2*E_k_fz))
fz_pre_2=cp.sqrt((eps_k+ZE_Q/2-E_k_fz)/(2*E_k_fz))

def BA_fz_mode(psi_k,psi_k_conj):
    #MAYBE TODO: i am not doing any time shifting of the conj here but instead i think u should just be able to time shift the quasiparticle operators?
    #if i did the time shift here then we get non time-ordered products so think this should be right, have to calculate to be sure tho
    psi_minus_k=k_to_minus_k(psi_k)
    psi_minus_k_conj=k_to_minus_k(psi_k_conj)
    #print(psi_minus_k)
    #print(psi_k)
    b_k_fz=     (fz_pre_1*(psi_k[2]     -     psi_k[0])+fz_pre_2*(psi_minus_k_conj[2]-psi_minus_k_conj[0]))/cp.sqrt(2)
    b_k_conj_fz=(fz_pre_1*(psi_k_conj[2]-psi_k_conj[0])+fz_pre_2*(     psi_minus_k[2]-     psi_minus_k[0]))/cp.sqrt(2)
    return b_k_fz,b_k_conj_fz


def fz_prod(psi_k,psi_k_conj):
    psi_f=1/cp.sqrt(2)*(psi_k[2]-psi_k[0])
    psi_f_conj=1/cp.sqrt(2)*(psi_k_conj[2]-psi_k_conj[0])
    psi_minus_f     =k_to_minus_k(psi_f,     s_axes=tuple(range(1,dim+1)))
    psi_minus_f_conj=k_to_minus_k(psi_f_conj,s_axes=tuple(range(1,dim+1)))
    
    bdagb=fz_pre_1**2*psi_f_conj*psi_f\
        +fz_pre_2**2*(psi_minus_f_conj*psi_minus_f+1)\
        +fz_pre_1*fz_pre_2*(psi_f_conj*psi_minus_f_conj+psi_f*psi_minus_f)
    return bdagb
##observables





#mean density
def mean_density(psi_k, psi_k_conj):
    psi_k_conj_adv=CachedArray.roll(psi_k_conj,-1,axis=time_axis)
    rho=cp.tensordot(psi_k_conj_adv,psi_k,axes=[range(1,dim+2),range(1,dim+2)])#always use tensordot instead of einsum or manually multiply and sum bc it is super fast for some reason #cp.einsum("ctijk,ctijk->c",psi_k_conj_adv,psi_k)/d_time/d_space**dim
    #print(rho)
    rho=cp.diag(rho)/d_time/d_space**dim
    #rho2=cp.einsum("btijk,ctijk->bc",psi_k_conj_adv,psi_k)
    return rho

#create masks for each momentum bin
#this way, the binning involves only array multiplications and summations == fast
BIN_W=1/d_space
#N_BINS=d_space//(4-dim)
momentum_grid=cp.sqrt(laplacian_grid)
max_momentum=1.5# the lower this is the faster the program, and high momenta are usually inaccurate anyways, so not worth it to get possible bins
N_BINS=int(max_momentum/BIN_W)
bin_grid=cp.floor(momentum_grid/max_momentum*N_BINS).astype(cp.int32)
number_masks=cp.array([cp.where(bin_grid==i,1,0) for i in range(N_BINS)])
number_masks=number_masks[cp.any(number_masks,axis=tuple(range(1,1+dim)))]
bins_per_mom=cp.count_nonzero(number_masks,axis=tuple(range(1,1+dim)))
momentum_axis=cp.sum(number_masks*momentum_grid,axis=tuple(range(1,1+dim)))/bins_per_mom
corr_factor=1/bins_per_mom
N_BINS=len(number_masks)
def occ_numbers(psi_k,psi_k_conj):
    psi_k_conj_adv=CachedArray.roll(psi_k_conj,-1,axis=time_axis)
    rho_k=cp.sum(cp.real(psi_k*psi_k_conj_adv),axis=time_axis)
    occ_bins=cp.tensordot(rho_k,number_masks,axes=(range(1,1+dim),range(1,1+dim)))
    return occ_bins/d_space**dim/d_time*corr_factor



#mainly needed to get the dispersion, but could also be generally interesting
def occ_numbers_quasiparticles(psi_k,psi_k_conj):
    #only have one component rn but for future extra modes i already add the dimension
    psi_k_conj_adv=CachedArray.roll(psi_k_conj,-1,axis=time_axis)
    
    b=fz_prod(psi_k,psi_k_conj_adv)
    
    
    #b_k,b_k_conj_adv=cp.swapaxes(cp.array([BA_fz_mode(psi_k,psi_k_conj_adv)]),0,1)
    b=b.view(cp.ndarray)
    rho_k=cp.sum(cp.real(cp.array([b])),axis=time_axis)
    occ_bins=cp.tensordot(rho_k,number_masks,axes=(range(1,1+dim),range(1,1+dim)))
    return -occ_bins/d_space**dim/d_time*corr_factor

#calculates the squared dispersion w^2*N_k, still has to be divided by the respecive momentum bin and taken sqrt
#if hamiltonian not diagonal in modes have to do bogoliubov trafo first
def dispersion(psi_k,psi_k_conj):
    psi_k_conj_p1=cp.roll(psi_k_conj,-1,axis=time_axis)
    psi_k_conj_p2=cp.roll(psi_k_conj,-2,axis=time_axis)
    psi_k_m1=cp.roll(psi_k,1,axis=time_axis)
    w_k=cp.sum(cp.real((psi_k_conj_p2-psi_k_conj_p1)*(psi_k-psi_k_m1)),axis=time_axis)
    disp_bins=cp.tensordot(w_k,number_masks,axes=(range(1,1+dim),range(1,1+dim)))/d_space**dim/d_time
    return disp_bins*corr_factor


def dispersion_quasiparticles(psi_k,psi_k_conj):
    #only have one component rn but for future extra modes i already add the dimensiono do bogoliubov trafo first
    psi_k_conj_p1=CachedArray.roll(psi_k_conj,-1,axis=time_axis)
    psi_k_conj_p2=CachedArray.roll(psi_k_conj,-2,axis=time_axis)
    psi_k_m1=CachedArray.roll(psi_k,1,axis=time_axis)
    
    
    psi_f=1/cp.sqrt(2)*(psi_k[2]-psi_k[0]).view(cp.ndarray)
    psi_f_conj_p1=1/cp.sqrt(2)*(psi_k_conj_p1[2]-psi_k_conj_p1[0]).view(cp.ndarray)
    psi_f_conj_p2=1/cp.sqrt(2)*(psi_k_conj_p2[2]-psi_k_conj_p2[0]).view(cp.ndarray)
    psi_f_m1=1/cp.sqrt(2)*(psi_k_m1[2]-psi_k_m1[0]).view(cp.ndarray)
    #D=(fz_pre_1**2+fz_pre_2**2)*((psi_f_conj_p2-psi_f_conj_p1)*(psi_f-psi_f_m1))\
    #    +fz_pre_1*fz_pre_2*(anom_product(psi_f_conj_p2-psi_f_conj_p1, s_axes=tuple(range(1,dim+1)))\
    #    +anom_product(psi_f-psi_f_m1, s_axes=tuple(range(1,dim+1))))
    #D=D.view(cp.ndarray)
    D=cp.zeros_like(psi_f)
    make_quasi_disp[(8,8,8),(8,8,8)](D,fz_pre_1,fz_pre_2,psi_f,psi_f_conj_p1,psi_f_conj_p2,psi_f_m1)
    
    w_k=cp.sum(cp.real(cp.array([D])),axis=time_axis)
    disp_bins=cp.tensordot(w_k,number_masks,axes=(range(1,1+dim),range(1,1+dim)))/d_space**dim/d_time
    return disp_bins*corr_factor

    
def eval_dispersion(disp_bins, spec_bins):
    return cp.where(-disp_bins/spec_bins>0,cp.sqrt(-disp_bins/spec_bins/d_time)/EPS,0)

def polarization(psi,psi_conj):
    if comp==3:
        psi_conj_adv=CachedArray.roll(psi_conj,-1,axis=time_axis)
        p=cp.tensordot(pauli_spin1,psi, axes=([2],[0]))
        p=cp.tensordot(psi_conj_adv,p,axes=(range(dim+2),range(1,dim+3)))
        return p/d_space**dim/d_time
    else:
        return cp.array([0,0,0])


def get_observables(psi,psi_conj):
    psi_k=cp.fft.fftn(psi, axes=space_axes).view(CachedArray)
    psi_k_conj=cp.fft.ifftn(psi_conj,axes=space_axes).view(CachedArray)
    
    occ=mean_density(psi_k,psi_k_conj)
    disp_quasi_bins=dispersion_quasiparticles(psi_k,psi_k_conj)

    mom_bins=occ_numbers(psi_k,psi_k_conj)
    mom_quasi_bins=occ_numbers_quasiparticles(psi_k,psi_k_conj)


    p=polarization(psi,psi_conj)
    
    return occ, disp_quasi_bins, mom_bins, mom_quasi_bins, p

class ObservablesTracker:
    def __init__(self):
        self.reset()
    def reset(self):
        self.occ=cp.zeros((comp))+0.0j

        self.mom_bins=cp.zeros((comp,N_BINS))
        self.mom_quasi_bins=cp.zeros((1,N_BINS))

        self.disp_bins=cp.zeros((comp,N_BINS))
        self.disp_quasi_bins=cp.zeros((1,N_BINS))

        self.pol=cp.zeros((3))+0.0j
        self.pol2=cp.zeros((3))+0.0j
        self.poltrans=cp.zeros((1))+0.0j
        self.l_time=0
        

    def print_debug(self, plot=True):
        print("Langevin Time passed:",self.l_time)
        print("Density:", self.occ/self.l_time)
        print("Total Density:", sum(self.occ/self.l_time))
        print("Condensate Fraction:",self.mom_bins[:,0]/self.occ)
        print("X/Y/Z Polarization:",self.pol/cp.sum(self.occ))
        if plot:
            plt.plot(momentum_axis.get(), (self.mom_bins/self.l_time).get().T)
            plt.yscale("log")
            plt.legend(["+1","0","-1"])
            plt.title("Distribution")

            plt.show()
            disp_evaled=eval_dispersion(self.disp_quasi_bins,self.mom_quasi_bins).get()[0]
            plt.plot(momentum_axis.get(),disp_evaled)
            try:
                def bog_spectrum(k,a,b):
                    return a*np.sqrt(k**2*(k**2+b*ZE_Q))
                popt,pcov=optimize.curve_fit(bog_spectrum, momentum_axis.get(),disp_evaled,p0=[1,1],maxfev=10000,bounds=(0,np.inf))
                plt.plot(momentum_axis.get(),bog_spectrum(momentum_axis.get(), 1,1))
                print(popt)
            except Exception as e: 
                print(e)

            plt.title("Dispersion of the fz mode")
            plt.show()
            plt.plot(momentum_axis.get()[1:],(self.mom_quasi_bins/self.l_time).get()[0,1:])
            plt.title("Occupation of the fz mode")
            plt.show()
    def save(self):
        if self.l_time>300:
            cp.savez(out_filename+".npz", density=self.occ, momentum_bin=self.mom_bins, momentum_quasi_bin=self.mom_quasi_bins, disp_quasi_bins=self.disp_quasi_bins, polarization=self.pol, pol2=self.pol2, poltrans=self.poltrans, l_time=self.l_time)
            with open(out_filename+".info","w") as out:
                out.write(out_filedescriptor)
    def load(self, filename):
        out=cp.load(filename)
        self.occ=out["density"]
        self.mom_bins=out["momentum_bin"]
        self.mom_quasi_bins=out["momentum_quasi_bin"]
        self.disp_quasi_bins=out["disp_quasi_bins"]
        self.pol=out["polarization"]
        self.l_time=out["l_time"]
    def load_add(self,filename):
        out=cp.load(filename)
        self.occ+=out["density"]
        self.mom_bins+=out["momentum_bin"]
        self.mom_quasi_bins+=out["momentum_quasi_bin"]
        self.disp_quasi_bins+=out["disp_quasi_bins"]
        self.pol=+out["polarization"]
        self.l_time=+out["l_time"]
    def observe(self,fields,dt_n):
        psi=fields.get_psi()
        psi_conj=fields.get_psi_conj()
        psi_k=fields.get_psi_k()
        psi_k_conj=fields.get_psi_k_conj()
        
        psi_k=cp.fft.fftn(psi, axes=space_axes).view(CachedArray)
        psi_k_conj=cp.fft.ifftn(psi_conj,axes=space_axes).view(CachedArray)

        self.occ+=mean_density(psi_k,psi_k_conj)*dt_n
        self.disp_quasi_bins+=dispersion_quasiparticles(psi_k,psi_k_conj)*dt_n

        self.mom_bins+=occ_numbers(psi_k,psi_k_conj)*dt_n
        self.mom_quasi_bins+=occ_numbers_quasiparticles(psi_k,psi_k_conj)*dt_n
        p=polarization(psi,psi_conj)
        self.pol+=p*dt_n
        self.pol2+=p**2*dt_n
        self.poltrans+=cp.sqrt(p[0]**2+p[1]**2)*dt_n
        self.l_time+=dt_n
        