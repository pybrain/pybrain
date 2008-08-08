from neuronlayer import NeuronLayer
from pybrain.tools.functions import safeExp


class SoftmaxLayer(NeuronLayer):
    """ A layer implementing a softmax distribution over the input."""
    
    # TODO: collapsing option?
    # CHECKME: temperature parameter?
    
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = safeExp(inbuf)
        outbuf /= sum(outbuf)
        
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outerr
        