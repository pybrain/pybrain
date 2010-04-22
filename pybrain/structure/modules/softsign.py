__author__ = 'Thomas Rückstieß, ruecksti@in.tum.de'

from neuronlayer import NeuronLayer

class SoftSignLayer(NeuronLayer):
    """ A layer implementing the tanh squashing function. """
    
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = inbuf / (1 + abs(inbuf))
        
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = (1 - abs(outbuf))**2 * outerr
