import cupy as cp
import cupyx as cpx
from cupyx import scipy, jit
from cupyx.scipy import signal
from cupyx.profiler import benchmark
import cupyx.scipy.fft
import numpy as np
ALPHA=0.0

#Spin 1 interaction constants
C0=0.1
C2=0.05

#dimensionality
comp=3
dim=3
d_space=16
d_time=16

#Gas constants
DT=3e-3
EPS=0.05#T=1/(EPS*d_time)
MU=0.5
G=0.1

print(f'T={1/(EPS*d_time)}')
print(f'BKT Temp: {cp.pi*cp.sqrt(MU/G)}')
##create a lattice with d spatial and 1 time dimension (time is 0 axis)
dims=[comp]+[d_time]+[d_space]*dim
psi=cp.zeros(dims)+0.0j
#psi+=cp.random.rand(d_time,d_space,d_space,d_space)*
psi_conj=cp.zeros(dims)+0.0j
#psi_conj+=cp.random.rand(d_time,d_space)
#psi+=cp.sqrt(MU/G)*1
#psi_conj+=cp.sqrt(MU/G)*1
#S=psi d psibar +psi dx2 psi + u psi2 + g psi4
time_axis=1
space_axes=tuple(range(2,dim+2))
comp_axis=0
print(dims)
##create a integral measure grid for quick computation
m_axes=cp.array(cp.meshgrid(*([cp.arange(d_space)]*dim)))
print(m_axes.shape)
#the measure ist abs(prod(cos(pi*j/N)))
measure_grid=cp.abs(cp.prod(cp.cos(cp.pi*m_axes/d_space),axis=0))
print(measure_grid.shape)
laplacian_grid=4*cp.sum(cp.sin(cp.pi*m_axes/d_space)**2,axis=0)
print(laplacian_grid.shape)
#central time derivative, hope the rolling on GPU is efficient
def d_dt(psi):
	psi_advanced=cp.roll(psi,-1,axis=time_axis)
	#psi_retarded=cp.roll(psi,1,axis=time_axis)
	return (psi_advanced-psi)

"""
laplacian_kernel=cp.zeros(([1]+[3]*dim))
laplacian_kernel[tuple([0]+[1]*dim)]=1
for d in range(dim):
	middle=cp.array([0]+[1]*dim,dtype=int)
	m_up=middle+cp.eye(dim+1,dtype=int)[d+1]
	m_down=middle-cp.eye(dim+1,dtype=int)[d+1]
	laplacian_kernel[tuple(m_up)]=1
	laplacian_kernel[tuple(m_down)]=1

laplacian_kernel[tuple([0]+[1]*dim)]=-2*dim
#laplacian using kernel convolution
def laplacian(psi):
	psib=cp.pad(psi, ([[0,0]]+[[1,1]]*dim),mode="wrap")
	return signal.fftconvolve(psib,laplacian_kernel,mode="valid")
"""

#directly implementing using fft is a lot faster (see benchmarks)
def laplacian_k(psi):
	psi_k=cp.fft.fftn(psi, axes=space_axes)
	psi_k*=laplacian_grid
	psi=cp.fft.ifftn(psi_k, axes=space_axes)
	return -psi

##this gets list and factors for multiplication in U(1) gas directly from the representation
#it was a mistake. i should have just hardcoded it, only advantage is quicker generalization to spin 2 maybe
pauli_spin1=cp.array([1/cp.sqrt(2)*cp.array([[0,1,0], [1,0,1], [0,1,0]]),\
			 		 1j/cp.sqrt(2)*cp.array([[0,-1,0],[1,0,-1],[0,1,0]]),\
			 					   cp.array([[1,0,0], [0,0,0], [0,0,-1]])])
F_ijkl=cp.einsum("ajk,ail->ijkl",pauli_spin1,pauli_spin1)

F_sum_path=np.einsum_path("btijk,abcd,ctijk,dtijk->atijk",psi.get(),F_ijkl.get(),psi_conj.get(),psi_conj.get(), optimize="optimal")
F_ijkl[cp.abs(F_ijkl)<1e-8]=0.0
inds,values=cp.array(cp.nonzero(F_ijkl)),F_ijkl[cp.nonzero(F_ijkl)]
d1=inds[0]
d2=inds[1]
ds=cp.concatenate((d1,d2))
ms1=inds[(1,2,3),].T
ms2=inds[(0,2,3),].T
ms=np.concatenate((ms1,ms2))

sum_inds=[[],[],[]]
sum_vals=[[],[],[]]
for d,m,v in zip(ds,ms,values):
	m[-2:].sort()
	sum_inds[d.get()].append(m.get())
	sum_vals[d.get()].append(v.get())

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
	inds_unique.append(unique_nonzero-1)


#print(inds.T,values)
#print(inds_unique,factors_unique)
def calculate_drift(psi, psi_conj):
	psi_conj_adv=cp.roll(psi_conj,-1,axis=time_axis)
	psi_ret=cp.roll(psi,1,axis=time_axis)
	rhop=cp.sum(psi_conj*psi_ret,axis=0)
	rhom=cp.sum(psi_conj_adv*psi,axis=0)
	H_int=C0*psi_ret*rhop
	H_int_conj=C0*psi_conj_adv*rhom
	for i in range(comp):
		for inds,f in zip(inds_unique[i],factors_unique[i]):
			H_int[i]+=f*C2*psi_conj[inds[0]]*psi_ret[inds[1]]*psi_ret[inds[2]]
			H_int_conj[i]+=f*C2*psi[inds[0]]*psi_conj_adv[inds[1]]*psi_conj_adv[inds[2]]

	drift=DT*(-d_dt(psi_ret)+EPS*(laplacian_k(psi_ret)+MU*psi_ret-H_int))
	drift_conj=DT*(d_dt(psi_conj)+EPS*(laplacian_k(psi_conj_adv)+MU*psi_conj_adv-H_int))
	return drift,drift_conj

def time_step(psi,psi_conj,momentum,momentum_conj):
	drift,drift_conj=calculate_drift(psi,psi_conj)
	momentum=ALPHA*momentum+(1-ALPHA)*drift
	momentum_conj=ALPHA*momentum_conj+(1-ALPHA)*drift_conj
	noise=cp.random.normal(0,cp.sqrt(DT),size=dims+[2],dtype=cp.float32).view(cp.complex64).squeeze()
	noise_conj=cp.conj(noise)
	delta=momentum+noise
	delta_conj=momentum_conj+noise_conj
	return psi+delta,psi_conj+delta_conj,momentum,momentum_conj

##observables
#mean density
def mean_density(psi_k, psi_k_conj):
	psi_k_conj_adv=cp.roll(psi_k_conj,-1,axis=time_axis)
	rho=cp.sum(psi_k_conj_adv*psi_k*measure_grid,axis=range(1,dim+2))/d_time/d_space**dim
	return rho

#we create masks for each momentum bin
#this way, the binning involves only array multiplications and summations == fast
N_BINS=d_space
momentum_grid=cp.sqrt(laplacian_grid)
max_momentum=cp.max(momentum_grid)
bin_grid=cp.floor(momentum_grid/max_momentum*N_BINS).astype(cp.int32)
number_masks=cp.array([cp.where(bin_grid==i,1,0) for i in range(N_BINS)])
def occ_numbers(psi_k,psi_k_conj):
	psi_k_conj_adv=cp.roll(psi_k_conj,-1,axis=time_axis)
	rho_k=cp.sum(cp.real(psi_k*psi_k_conj_adv),axis=time_axis)
	occ_bins=cp.tensordot(rho_k,number_masks,axes=(range(1,1+dim),range(1,1+dim)))
	return occ_bins

#calculates the squared dispersion w^2*N_k, still has to be divided by the respecive momentum bin and taken sqrt
def dispersion(psi_k,psi_k_conj):
	psi_k_conj_p1=cp.roll(psi_k_conj,-1,axis=time_axis)
	psi_k_conj_p2=cp.roll(psi_k_conj,-2,axis=time_axis)
	psi_k_m1=cp.roll(psi_k,1,axis=time_axis)
	w_k=cp.sum(cp.real((psi_k_conj_p2-psi_k_conj_p1)*(psi_k-psi_k_m1)),axis=time_axis)
	disp_bins=cp.tensordot(w_k,number_masks,axes=(range(1,1+dim),range(1,1+dim)))
	return disp_bins
	
def eval_dispersion(disp_bins, spec_bins):
	return cp.where(-disp_bins/spec_bins>0,cp.sqrt(-disp_bins/spec_bins)/EPS,0)

occ=[]
mom_bins=[]
cache=cp.fft.config.get_plan_cache()


print(benchmark(d_dt, (psi,),n_repeat=5))
print(benchmark(laplacian_k, (psi,),n_repeat=5))
print(benchmark(mean_density, (psi,psi_conj),n_repeat=5))
print(benchmark(occ_numbers,(psi,psi_conj),n_repeat=5))
print(benchmark(calculate_drift,(psi,psi_conj),n_repeat=5))
input()
momentum=0
momentum_conj=0
for i in range(10000000):
	if i%1000==0:
		print(i)
	psi,psi_conj,momentum,momentum_conj=time_step(psi,psi_conj,momentum,momentum_conj)
	if i>=150000:
		ALPHA=0
		psi_k=cp.fft.fftn(psi, axes=space_axes)
		psi_k_conj=cp.fft.fftn(psi_conj,axes=space_axes)
		occ.append(mean_density(psi_k,psi_k_conj))
		mom_bins.append(occ_numbers(psi_k,psi_k_conj))
		if i%1000==0:
			print("Particle Number:",cp.average(occ,axis=0))
			print("Condensate Fraction:",cp.average(mom_bins,axis=0)[:,0]/cp.sum(cp.average(mom_bins,axis=0),axis=1))
			print(cp.average(mom_bins,axis=0))

