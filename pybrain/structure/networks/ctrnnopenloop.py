
__author__ = 'Daniel L Elliott, dane@cs.colostate.edu'

from pybrain.structure.networks import RecurrentNetwork
from pybrain.structure.modules import CTRNNLayer, LinearLayer, SigmoidLayer, SoftmaxLayer
from pybrain.structure.connections import FullConnection, IdentityConnection

import numpy as np

class CTRNNOpenLoop(RecurrentNetwork):
    """ Class that implements the CTRNN when operating in open-loop mode (receiving inputs)

    Requires a description of how the units should be connected and their tau values.  This is done using two NxN matrix where N is the total number of nodes in the network"""

    def __init__(self, numIO, numContextInputs, numContextNodes, IOtau, contextTau, *args, **kwargs):
        """ do it! """

        # call parent contructor
        RecurrentNetwork.__init__(self, *args, **kwargs)

        # create CTRNN layers
        contextPotentials = CTRNNLayer(numContextNodes+numContextInputs,name="contu")
        IOpotentials = CTRNNLayer(numIO,name="IOu")

        # set CTRNN layer parameters
        contextPotentials.setTau(np.ones(numContextNodes+numContextInputs) * contextTau)
        IOpotentials.setTau(np.ones(numIO) * IOtau)

        # add modules to network
        self.addInputModule(LinearLayer(numIO,name="IOx"))
        self.addModule(LinearLayer(numContextNodes+numContextInputs,name="contx"))

        self.addModule(contextPotentials)
        self.addModule(IOpotentials)

        self.addModule(SigmoidLayer(numContextNodes+numContextInputs,name="conty"))
        self.addOutputModule(SoftmaxLayer(numIO,name="IOy"))

        # add connections to network
        self.addConnection(FullConnection(self["IOx"],
                                          self["IOu"],
                                          name = "IOc1"))
        self.addConnection(FullConnection(self["contx"],
                                          self["contu"],
                                          name = "contc1"))
        self.addConnection(FullConnection(self["IOx"],
                                          self["contu"],
                                          name = "IOx2contu"))
        self.addConnection(FullConnection(self["contx"],
                                          self["IOu"],
                                          name = "contx2IOu"))

        self.addConnection(IdentityConnection(self["IOu"],
                                              self["IOy"],
                                              name = "IOc2"))
        self.addConnection(IdentityConnection(self["contu"],
                                              self["conty"],
                                              name = "contc2"))

        self.addRecurrentConnection(IdentityConnection(self["conty"],
                                                       self["contx"],
                                                       name = "contr1"))
        
