import numpy as np
from random import Random

np_rand_gen = np.random.default_rng(0)
py_rand_gen = Random(0) 

def set_random_seed(seed):
    global np_rand_gen
    global py_rand_gen

    np_rand_gen = np.random.default_rng(seed)
    py_rand_gen = Random(seed) 
