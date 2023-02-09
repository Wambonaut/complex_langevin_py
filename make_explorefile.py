#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cupy as cp
import numpy as np
from itertools import product
data=np.genfromtxt("explore.txt",dtype=None,encoding="utf-8")
var_names=[d[0] for d in data]
spaces=[np.linspace(float(d[1]),float(d[2]),int(d[3])) for d in data]
params=product(*spaces)
p=np.array(list(params))
v=np.array(list(var_names))
np.save("var_names.npy",v)
np.save("exploration_params.npy", p)




