import numpy as np
from numba import jit

@jit(nopython=True) 
def sqrt_err(coo_tensor, vals, shape, a, b, c):
    result = 0.0
    for item in range(coo_tensor.shape[0]):
        coord = coo_tensor[item]
        result += (vals[item] - np.sum(
            a[coord[0], :] * b[coord[1], :] * c[coord[2], :]))**2        
    return np.sqrt(result)


@jit(nopython=True) 
def sqrt_err_relative(coo_tensor, vals, shape, a, b, c):
    result = sqrt_err(coo_tensor, vals, shape, a, b, c)        
    return result / np.sqrt((vals**2).sum())