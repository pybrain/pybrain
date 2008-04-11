__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.tools.shortcuts import buildNetwork
from pybrain import LinearLayer
from pybrain.tools.pyrex._linearlayer import LinearLayer_forwardImplementation

N = buildNetwork(2,3,4)
 
ll = N['out']
ll2 = N['in']

def f(*a):
    print '--',
    LinearLayer_forwardImplementation(*a)
    print '!!'

def f2(*a):
    print '**',    
    LinearLayer._forwardImplementation(ll2, *a)
    print '!!'

ll._forwardImplementation = f
ll2._forwardImplementation = f2
 

print ll.__class__
print ll2.__class__

N.activate([0,0])
