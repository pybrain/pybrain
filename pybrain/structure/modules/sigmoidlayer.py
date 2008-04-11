from neuronlayer import NeuronLayer
from pybrain.tools.functions import sigmoid
from pybrain.utilities import substitute


class SigmoidLayer(NeuronLayer):
    """ A layer implementing the sigmoid squashing function. """

    @substitute('pybrain.tools.pyrex._sigmoidlayer.SigmoidLayer_forwardImplementation')
    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = sigmoid(inbuf)
        
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outbuf*(1-outbuf)*outerr