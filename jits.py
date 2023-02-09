import cupy as cp
from cupyx import scipy, jit
from parameters import *
##JIT implementation of the spin-spin C2 interaction
@jit.rawkernel()
def make_Hint(psi,psi_conj,H_int,in1,in2,in3,f1,f2,f3,C):
    tidx=jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    tidy=jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    tidz=jit.blockIdx.z * jit.blockDim.z + jit.threadIdx.z
    n_tidx=jit.gridDim.x * jit.blockDim.x
    n_tidy=jit.gridDim.y * jit.blockDim.y
    n_tidz=jit.gridDim.z * jit.blockDim.z
    for t in range(d_time):
        for x in range(tidx, d_space,n_tidx):
            for y in range(tidy, d_space,n_tidy):
                for z in range(tidz, d_space,n_tidz):
                    for in_ind in range(len(in1)):
                        #idk why it doesnt work to multiply all three terms together directly but it doesnt
                        #hope compiler is smart enough to make it efficient anyway
                        H_intermediate=psi_conj[in1[in_ind,0],t,x,y,z]
                        H_intermediate*=psi[in1[in_ind,1],t,x,y,z]
                        H_intermediate*=psi[in1[in_ind,2],t,x,y,z]
                        H_int[0,t,x,y,z]+=f1[in_ind]*C*H_intermediate                    
                    for in_ind in range(len(in2)):
                        H_intermediate=psi_conj[in2[in_ind,0],t,x,y,z]
                        H_intermediate*=psi[in2[in_ind,1],t,x,y,z]
                        H_intermediate*=psi[in2[in_ind,2],t,x,y,z]
                        H_int[1,t,x,y,z]+=f2[in_ind]*C*H_intermediate                
                    for in_ind in range(len(in3)):
                        H_intermediate=psi_conj[in3[in_ind,0],t,x,y,z]
                        H_intermediate*=psi[in3[in_ind,1],t,x,y,z]
                        H_intermediate*=psi[in3[in_ind,2],t,x,y,z]
                        H_int[2,t,x,y,z]+=f3[in_ind]*C*H_intermediate

#quick density contraction, should be bit faster since cupy multiply and sum kinda slow
@cp.fuse()
def rho_t(psi,psi_conj):
    return cp.sum(psi * psi_conj, axis = 0)
                       
#annoying calculation of the quasiparticle dispersion, not a lot faster but w.e
@jit.rawkernel()
def make_quasi_disp(D,fz_pre_1,fz_pre_2,psi_f,psi_f_conj_p1,psi_f_conj_p2,psi_f_m1):
    tidx=jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    tidy=jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    tidz=jit.blockIdx.z * jit.blockDim.z + jit.threadIdx.z
    n_tidx=jit.gridDim.x * jit.blockDim.x
    n_tidy=jit.gridDim.y * jit.blockDim.y
    n_tidz=jit.gridDim.z * jit.blockDim.z
    for x in range(tidx, d_space,n_tidx):
        for y in range(tidy, d_space,n_tidy):
            for z in range(tidz, d_space,n_tidz):
                for t in range(d_time):
                    D[t,x,y,z]+=(fz_pre_1[x,y,z]**2+fz_pre_2[x,y,z]**2)*((psi_f_conj_p2[t,x,y,z]-psi_f_conj_p1[t,x,y,z])*(psi_f[t,x,y,z]-psi_f_m1[t,x,y,z]))\
                    +fz_pre_1[x,y,z]*fz_pre_2[x,y,z]*((psi_f_conj_p2[t,x,y,z]-psi_f_conj_p1[t,x,y,z])*(psi_f_conj_p2[t,(d_space-x)%d_space,(d_space-y)%d_space,(d_space-z)%d_space]-psi_f_conj_p1[t,(d_space-x)%d_space,(d_space-y)%d_space,(d_space-z)%d_space]) \
                    +(psi_f[t,x,y,z]-psi_f_m1[t,x,y,z])*(psi_f[t,(d_space-x)%d_space,(d_space-y)%d_space,(d_space-z)%d_space]-psi_f_m1[t,(d_space-x)%d_space,(d_space-y)%d_space,(d_space-z)%d_space]))
