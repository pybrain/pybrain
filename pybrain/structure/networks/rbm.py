# -*- coding: utf-8 -*-

__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'
__version__ = '$Id$'


from pybrain.structure import (LinearLayer, SigmoidLayer, FullConnection,
                               BiasUnit, FeedForwardNetwork)


class Rbm(object):
    """Class that holds a network and offers some shortcuts."""

    @property
    def params(self):
        return self.con.params
        pass

    @property
    def biasParams(self):
        return self.biascon.params

    @property
    def visibleDim(self):
        return self.net.indim

    @property
    def hiddenDim(self):
        return self.net.outdim

    def __init__(self, net):
        self.net = net
        self.net.sortModules()
        self.bias = [i for i in self.net.modules if isinstance(i, BiasUnit)][0]
        self.biascon = self.net.connections[self.bias][0]
        self.visible = net['visible']
        self.hidden = net['hidden']
        self.con = self.net.connections[self.visible][0]

    @classmethod
    def fromDims(cls, visibledim, hiddendim, params=None, biasParams=None):
        """Return a restricted Boltzmann machine of the given dimensions with the
        given distributions."""
        net = FeedForwardNetwork()
        bias = BiasUnit('bias')
        visible = LinearLayer(visibledim, 'visible')
        hidden = SigmoidLayer(hiddendim, 'hidden')
        con1 = FullConnection(visible, hidden)
        con2 = FullConnection(bias, hidden)
        if params is not None:
            con1.params[:] = params
        if biasParams is not None:
            con2.params[:] = biasParams

        net.addInputModule(visible)
        net.addModule(bias)
        net.addOutputModule(hidden)
        net.addConnection(con1)
        net.addConnection(con2)
        net.sortModules()
        return cls(net)

    @classmethod
    def fromModules(cls, visible, hidden, bias, con, biascon):
        net = FeedForwardNetwork()
        net.addInputModule(visible)
        net.addModule(bias)
        net.addOutputModule(hidden)
        net.addConnection(con)
        net.addConnection(biascon)
        net.sortModules()
        return cls(net)

    def invert(self):
        """Return the inverse rbm."""
        # TODO: check if shape is correct
        return self.__class__.fromDims(self.hiddenDim, self.visibleDim,
                                       params=self.params)

    def activate(self, inpt):
        return self.net.activate(inpt)
