from pyrex_header cimport exp, tanh, ndarray, import_array
import_array()

# general purpose functions

# safe sigmoid
def sigmoid(double x):
    if x > 30.0:
       return 1.0
    elif x < -30.0:
       return 0.0
    else:   	
       return 1.0/(1.0+exp(-x))
              
def sigmoidprime(double x):
       cdef double a
       a = sigmoid(x)
       return a*(1.0 - a)
       
def tanhprime(double x):
       cdef double a
       a = tanh(x)
       return 1.0 - a*a
