from scipy import zeros, rand
import time

from pybrain.structure.module import Module
from pybrain import MDLSTMLayer, LSTMLayer, Network, LinearLayer, IdentityConnection
from pybrain.tests import gradientCheck


class LSTMLayer2(Module):
    sequential = True
    
    def __init__(self, size, peepholes = False, name = None):
        Module.__init__(self, size*4, size, name = name)
        self.base = MDLSTMLayer(size, peepholes = peepholes, dimensions = 1)
        
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf2 = zeros(self.outdim*2)
        self.base._forwardImplementation(inbuf, outbuf2)
        outbuf += outbuf2[:self.outdim]
        
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        outbuf2 = zeros(self.outdim*2)
        outbuf2[:self.outdim] += outbuf
        outerr2 = zeros(self.outdim*2)
        outerr2[:self.outdim] += outerr
        self.base._backwardImplementation(outerr2, inerr, outbuf2, inbuf)
        
    def _setParameters(self, p):
        self.base._setParameters(p)
    
        
class LSTMLayer3(Network):
    def __init__(self, dim, peepholes = False, name = None):
        Network.__init__(self, name = name)
        self.addInputModule(LinearLayer(4*dim, name = 'i'))
        self.addOutputModule(LinearLayer(dim, name = 'o'))
        self.addModule(MDLSTMLayer(dim, peepholes = peepholes, name = 'h'))
        self.addConnection(IdentityConnection(self['i'], self['h'], outSliceTo = 4*dim))
        self.addConnection(IdentityConnection(self['h'], self['o'], inSliceTo = dim))
        self.addRecurrentConnection(IdentityConnection(self['h'], self['h'], inSliceFrom = dim, outSliceFrom = 4*dim))
        self.sortModules()
        
        
def compareImplementations():
    dim = 4
    peep = True
    l1 = LSTMLayer(dim, peep)
    # incomplete: l2 = LSTMLayer2(dim, peep)
    l3 = LSTMLayer3(dim, peep)
    if peep:
        l1._setParameters(l3.getParameters())
        #l2._setParameters(l3.getParameters())
    steps = 3
    for dummy in range(steps):
        r = rand(4*dim)
        print l1.activate(r)
        #print l2.activate(r)
        print l3.activate(r)
    
    for dummy in range(steps):
        r = rand(1*dim)
        print l1.backActivate(r)
        #print l2.backActivate(r)
        print l3.backActivate(r)
    gradientCheck(l1)
    #gradientCheck(l2)
    gradientCheck(l3)
    
    
def compareTimings(dim = 10, steps = 100):
    peep = True
    l1 = LSTMLayer(dim, peep)
    l3 = LSTMLayer3(dim, peep)
    r1 = rand(4*dim)*0.001
    r2 = rand(dim)*0.001
    start1 = time.time()
    for dummy in range(steps):
        l1.activate(r1)
    for dummy in range(steps):
        l1.backActivate(r2)
    time1 = time.time() - start1
    start2 = time.time()
    for dummy in range(steps):
        l3.activate(r1)
    for dummy in range(steps):
        l3.backActivate(r2)
    time2 = time.time() - start2
    print 'size', dim, 'steps', steps
    print ' 1:', time1
    print ' 2:', time2
    
    
if __name__ == '__main__':
    compareImplementations()
    compareTimings(1, 2500)
    compareTimings(10, 2000)
    compareTimings(100, 800)
    compareTimings(1000, 150)