# complex_langevin_py
Implementation of the Complex Langevin Method in Python (GPU accelerated)   
Run with CUPY_ACCELERATORS=cutensornet,cutensor,cub for best performance
   
## Currently working:   
Perform complex langevin on multi-component complex nonrelativistic scalar field      
Arbitrary spatial dimension   
Arbitrary lattice sizes   
Observables: particle number, momentum spectrum, dispersion    
CUDA acceleration   
Spin-1 Spin-Spin and quadratic Zeeman Interaction
   
## TODO:  
Add more Observables, especially arbitrary n-point functions   
(Maybe) refactor to OOP   
Improve GPU acceleration with costum CUDA kernels (already pretty good tho)   
