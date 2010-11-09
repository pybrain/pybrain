# -*- coding: utf-8 -*-


"""Module that contains the RecurrentNetwork class."""


__author__ = 'Justin Bayer, bayer.justin@googlemail.com'


from pybrain.structure.networks.network import Network
from pybrain.structure.connections.shared import SharedConnection


class RecurrentNetworkComponent(object):

    sequential = True

    def __init__(self, forget=None, name=None, *args, **kwargs):
        self.recurrentConns = []
        self.maxoffset = 0
        self.forget = forget

    def __str__(self):
        s = super(RecurrentNetworkComponent, self).__str__()
        s += "   Recurrent Connections:\n    %s" % (
                sorted(self.recurrentConns, key=lambda c: c.name))
        return s

    def _containerIterator(self):
        for c in super(RecurrentNetworkComponent, self)._containerIterator():
            yield c
        for c in self.recurrentConns:
            if c.paramdim and not isinstance(c, SharedConnection):
                yield c

    def addRecurrentConnection(self, c):
        """Add a connection to the network and mark it as a recurrent one."""
        if isinstance(c, SharedConnection):
            if c.mother not in self.motherconnections:
                self.motherconnections.append(c.mother)
                c.mother.owner = self
        elif c.paramdim > 0:
            c.owner = self
        self.recurrentConns.append(c)
        self.sorted = False

    def activate(self, inpt):
        """Do one transformation of an input and return the result."""
        self.inputbuffer[self.offset] = inpt
        self.forward()
        if self.forget:
            return self.outputbuffer[self.offset].copy()
        else:
            return self.outputbuffer[self.offset - 1].copy()

    def backActivate(self, outerr):
        """Do one transformation of an output error outerr backward and return
        the error on the input."""
        self.outputerror[self.offset - 1] = outerr
        self.backward()
        return self.inputerror[self.offset].copy()

    def forward(self):
        """Produce the output from the input."""
        if not (self.offset + 1 < self.inputbuffer.shape[0]):
            self._growBuffers()
        super(RecurrentNetworkComponent, self).forward()
        self.offset += 1
        self.maxoffset = max(self.offset, self.maxoffset)

    def backward(self):
        """Produce the input error from the output error."""
        self.offset -= 1
        super(RecurrentNetworkComponent, self).backward()

    def _isLastTimestep(self):
        return self.offset == self.maxoffset

    def _forwardImplementation(self, inbuf, outbuf):
        assert self.sorted, ".sortModules() has not been called"

        if self.forget:
            self.offset += 1

        index = 0
        offset = self.offset
        for m in self.inmodules:
            m.inputbuffer[offset] = inbuf[index:index + m.indim]
            index += m.indim

        if offset > 0:
            for c in self.recurrentConns:
                c.forward(offset - 1, offset)

        for m in self.modulesSorted:
            m.forward()
            for c in self.connections[m]:
                c.forward(offset, offset)

        if self.forget:
            for m in self.modules:
                m.shift(-1)
            offset -= 1
            self.offset -= 2

        index = 0
        for m in self.outmodules:
            outbuf[index:index + m.outdim] = m.outputbuffer[offset]
            index += m.outdim

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        assert not self.forget, "Cannot back propagate a forgetful network"
        assert self.sorted, ".sortModules() has not been called"
        index = 0
        offset = self.offset
        for m in self.outmodules:
            m.outputerror[offset] = outerr[index:index + m.outdim]
            index += m.outdim

        if not self._isLastTimestep():
            for c in self.recurrentConns:
                c.backward(offset, offset + 1)

        for m in reversed(self.modulesSorted):
            for c in self.connections[m]:
                c.backward(offset, offset)
            m.offset = offset
            m.backward()

        index = 0
        for m in self.inmodules:
            inerr[index:index + m.indim] = m.inputerror[offset]
            index += m.indim

    def sortModules(self):
        self.recurrentConns.sort(key=lambda x: x.name)
        super(RecurrentNetworkComponent, self).sortModules()


class RecurrentNetwork(RecurrentNetworkComponent, Network):
    """Class that implements networks which can work with sequential data.

    Until .reset() is called, the network keeps track of all previous inputs and
    thus allows the use of recurrent connections and layers that look back in
    time, unless forget is set to True."""

    bufferlist = Network.bufferlist

    def __init__(self, *args, **kwargs):
        Network.__init__(self, *args, **kwargs)
        if 'forget' in kwargs:
            forget = kwargs['forget']
        else:
            forget = False
        RecurrentNetworkComponent.__init__(self, forget, *args, **kwargs)
