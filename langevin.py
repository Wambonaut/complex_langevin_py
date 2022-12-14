import cupy as cp
import cupyx as cpx
from cupyx import scipy, jit
from cupyx.scipy import signal
from cupyx.profiler import benchmark
import cupyx.scipy.fft
import numpy as np
dim=3
d_space=16
d_time=16
DT=3e-3
DX=1
EPS=0.01
MU=0.2
G=0.1
##create a lattice with d spatial and 1 time dimension (time is 0 axis)
dims=[d_time]+[d_space]*dim
psi=cp.zeros(dims)+0.0j
#psi+=cp.random.rand(d_time,d_space,d_space,d_space)*
psi_dag=cp.zeros(dims)+0.0j
#psi_dag+=cp.random.rand(d_time,d_space)
#psi+=cp.sqrt(MU/G)*1.1
#psi_dag+=cp.sqrt(MU/G)*1.1
#S=psi d psibar +psi dx2 psi + u psi2 + g psi4



##create a integral measure grid for quick computation
m_axes=cp.array(cp.meshgrid(*([cp.arange(d_space)]*dim)))
#the measure ist abs(prod(cos(pi*j/N)))
measure_grid=cp.abs(cp.prod(cp.cos(cp.pi*m_axes/d_space),axis=0))
laplacian_grid=4*cp.sum(cp.sin(cp.pi*m_axes/d_space)**2,axis=0)
#central time derivative, hope the rolling on GPU is efficient
def d_dt(psi):
	psi_advanced=cp.roll(psi,-1,axis=0)
	#psi_retarded=cp.roll(psi,1,axis=0)
	return (psi_advanced-psi)

#best laplacian implementation probably a kernel convolution, this way we do index crazy shit here and then never again, and also take advantage of fast cuda fft
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

#directly implementing using fft is a lot faster (see benchmarks)
def laplacian_k(psi):
	psi_k=cp.fft.fftn(psi, axes=tuple(range(1,dim+1)))
	psi_k*=laplacian_grid	
	psi=cp.fft.ifftn(psi_k, axes=tuple(range(1,dim+1)))
	return -psi


def calculate_drift(psi, psi_dag):
	psi_dag_adv=cp.roll(psi_dag,-1,axis=0)
	psi_ret=cp.roll(psi,1,axis=0)
	#print(laplacian_k(psi)[0,0,0])
	#print(laplacian(psi)[0,0,0])
	drift=DT*(-d_dt(psi_ret)+EPS*(laplacian_k(psi_ret)+MU*psi_ret-G*psi_ret*(psi_dag*psi_ret)))
	drift_dag=DT*(d_dt(psi_dag)+EPS*(laplacian_k(psi_dag_adv)+MU*psi_dag_adv-G*psi_dag_adv*(psi_dag_adv*psi)))
	noise=cp.random.normal(0,cp.sqrt(DT),size=dims+[2],dtype=cp.float32).view(cp.complex64).squeeze()
	noise_dag=cp.conj(noise)
	delta=drift+noise
	delta_dag=drift_dag+noise_dag
	return delta,delta_dag

def time_step(psi,psi_dag):
	drift,drift_dag=calculate_drift(psi,psi_dag)
	return psi+drift,psi_dag+drift_dag

##observables
#mean density
def mean_density(psi_k, psi_k_dag):
	psi_k_dag_adv=cp.roll(psi_k_dag,-1,axis=0)
	rho=cp.sum(psi_k_dag_adv*psi_k*measure_grid)/d_time/d_space**dim
	return rho

N_BINS=d_space
momentum_grid=cp.sqrt(laplacian_grid)
max_momentum=cp.max(momentum_grid)
bin_grid=cp.floor(momentum_grid/max_momentum*N_BINS).astype(cp.int32)
print(bin_grid)
number_grids=cp.array([cp.where(bin_grid==i,1,0) for i in range(N_BINS)])
def occ_numbers(psi_k,psi_k_dag):
	psi_k_dag_adv=cp.roll(psi_k_dag,-1,axis=0)
	rho_k=cp.sum(cp.real(psi_k*psi_k_dag_adv),axis=0)
	occ_bins=cp.sum(rho_k*number_grids,axis=tuple(range(1,dim+1)))
	return occ_bins

occ=[]
mom_bins=[]
cache=cp.fft.config.get_plan_cache()


print(benchmark(d_dt, (psi,),n_repeat=100))
print(benchmark(laplacian_k, (psi,),n_repeat=100))
print(benchmark(mean_density, (psi,psi_dag),n_repeat=100))
print(benchmark(occ_numbers,(psi,psi_dag),n_repeat=100))
for i in range(1000000):
	if i%1000==0:
		print(i)
	psi,psi_dag=time_step(psi,psi_dag)
	if i>=100000:
		psi_k=cp.fft.fftn(psi, axes=tuple(range(1,dim+1)))
		psi_k_dag=cp.fft.fftn(psi_dag,axes=tuple(range(1,dim+1)))
		occ.append(mean_density(psi_k,psi_k_dag))
		mom_bins.append(occ_numbers(psi_k,psi_k_dag))
		if i%1000==0:
			print(cp.average(occ))
			print("Condensate Fraction:",cp.average(mom_bins,axis=0)[0]/cp.sum(cp.average(mom_bins,axis=0)))
			print(cp.average(mom_bins,axis=0))

##todo: (maybe) rewrite laplacian using fft since it is fast
##understand dispersion (its because its on a lattice like CM)
##write observables code using fft
##
