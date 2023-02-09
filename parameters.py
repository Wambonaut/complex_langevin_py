import cupy as cp
from datetime import datetime
import numpy as np
import time

time.sleep(np.random.rand()*10)#this is a hacky solution to make multiple processes read the parameter file one after the other (in 99% of cases), but cba doing it better


now=datetime.now()#num constants
BETA=0.9
DT=3e-2
time_steps=300000
eq_steps=50000

#dimensionality

comp=3
dim=3
d_space=32
d_time=32
dims=[comp]+[d_time]+[d_space]*dim

#Gas constants
C0=0.5
C2=-0.25
ZE_P=0.0
ZE_Q=0.9
EPS=0.05
T=1/(EPS*d_time)
MU=1


##load variables from exploration file. This way, everytime the program is called, it laods a new set of variables from the param file, making it easy to explore a paramter area of interest by calling a lot of slurm jobs

names_file="var_names.npy"
param_file="exploration_params.npy"
out_list="out_list.txt"
timestr=now.strftime("%d-%m-%Y_%H:%M")
out_filename=f"out/Sim_{timestr}{hash(np.random.rand())}"
out_filedescriptor=f"dim={d_space}^{dim}x{d_time}\nC0={C0}\nC1={C2}\nEPS={EPS}\nMU={MU}\n"

var_names=np.load(names_file)
param_list=np.load(param_file)
if True:
    if len(param_list)>0:
        #load the parameters
        for n,p in zip(var_names,param_list[0]):
            exec(f"{n}={p}")##THIS IS INCREDIBLY INSECURE, ONE SHOULD JUST DIRECTLY ASSIGN, BUT I JUST HOPE NOBODY HAS THE PRIVILEGES TO PUT AN EVIL FILE IN MY DIRECTORY
        #note the future outfile in a list
        with open(out_list, "ab") as of:
            np.savetxt(of, np.array([param_list[0]]),delimiter=" ",header=out_filename)
        #remove the parameters used from the list
        np.save("exploration_params.npy",param_list[1:])
    
    
    

if comp==3:
    ZE=cp.array([ZE_Q+ZE_P,0,ZE_Q-ZE_P])
else:
    ZE=cp.array([0]*comp)

MUBAR=cp.expand_dims(MU-ZE,axis=tuple(range(1,dim+2)))

#mean field calculations
rho_exp=(MU-ZE_Q/2)/(C0+C2)
qbar=-ZE_Q/C2/rho_exp/2
rho1=cp.sqrt(rho_exp)*cp.sqrt(1-qbar)/2
rho0=cp.sqrt(rho_exp)*cp.sqrt((1+qbar)/2)

#####helper constants
time_axis=1
space_axes=tuple(range(2,dim+2))
comp_axis=0

#####make grids

##create a integral measure grid for quick computation
m_axes=cp.array(cp.meshgrid(*([cp.arange(d_space)]*dim)))
#the measure ist abs(prod(cos(pi*j/N)))
#measure_grid=cp.abs(cp.prod(cp.cos(cp.pi*m_axes/d_space),axis=0))#LOCAL VERSION
measure_grid=cp.zeros(dims)+1
#we keep the laplacian grid so it is equivalent to a local kernel, just k**2 might also work tho
#laplacian_grid=4*cp.sum(cp.sin(cp.pi*m_axes/d_space)**2,axis=0)#LOCAL VERSION
laplacian_grid=cp.fft.fftshift((2*cp.pi/d_space)**2*cp.sum((m_axes-d_space/2)**2,axis=0))



pauli_spin1=cp.array([1/cp.sqrt(2)*cp.array([[0,1.0,0], [1.0,0,1.0], [0,1.0,0]]),\
                      1j/cp.sqrt(2)*cp.array([[0,-1,0],[1,0,-1],[0,1,0]]),\
                                    cp.array([[1,0,0], [0,0,0], [0,0,-1]])])


def describe_parameters():
    print(out_filedescriptor)