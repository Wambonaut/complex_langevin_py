import cupy as cp
import cupyx as cpx
from cupyx import scipy, jit
from cupyx.profiler import benchmark
import numpy as np

import carray
import observables
import propagator
import helpers
from carray import CachedArray
from parameters import *
from propagator import Fields
from observables import ObservablesTracker
print("Temp:",T)
print("Mean field density:",rho_exp)
print("qbar expected:",qbar)
print("polarization expected:",cp.sqrt(1-qbar**2))



#some benchmarks
def benchmarks(fields):
    psi=fields.get_psi()
    psi_conj=fields.get_psi_conj()
    print(benchmark(carray.arrid,(psi,),n_repeat=1))
    print(benchmark(propagator.d_dt, (psi,),n_repeat=1))
    print(benchmark(propagator.laplacian_k, (fields,),n_repeat=1))
    print(benchmark(propagator.calculate_drift,(fields,),n_repeat=1))
    print(benchmark(propagator.time_step,(fields,),n_repeat=1))
    
    print(benchmark(observables.mean_density, (psi,psi_conj),n_repeat=1))
    print(benchmark(observables.occ_numbers,(psi,psi_conj),n_repeat=1))
    print(benchmark(observables.occ_numbers_quasiparticles,(psi,psi_conj),n_repeat=1))
    print(benchmark(observables.polarization,(psi,psi_conj),n_repeat=1))
    print(benchmark(observables.dispersion_quasiparticles,(psi,psi_conj),n_repeat=1))
    print(benchmark(CachedArray.roll,(psi_conj,1,time_axis),n_repeat=1))
    
cache=cp.fft.config.get_plan_cache()

out_steps=50000

tracker=ObservablesTracker()
fields=Fields()

describe_parameters()
for i in range(0,time_steps):
    if i%2500==0:
        print(i)
        fields.save()
    dt_n=fields.evolve()
    if i>=eq_steps or i%out_steps==0:
        if i<eq_steps:
            tracker.reset()
        tracker.observe(fields,dt_n)
        
        if i%out_steps==0:
            tracker.print_debug(plot=False)
            tracker.save()