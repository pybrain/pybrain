from scipy import zeros, where

def one_to_n(val, maxval):
    a = zeros(maxval, float)
    a[val] = 1.
    return a

def n_to_one(arr):
    return where(arr == 1)[0][0]
    