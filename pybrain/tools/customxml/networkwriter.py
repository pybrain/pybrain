__author__ = 'Tom Schaul, tom@idsia.ch'

from inspect import isclass

from .handling import XMLHandling
from pybrain.structure.connections.shared import SharedConnection
from pybrain.structure.networks.network import Network
from pybrain.structure.networks.recurrent import RecurrentNetwork
from pybrain.utilities import canonicClassString

# TODO: higher precision on writing parameters


class NetworkWriter(XMLHandling):
    """ A class that can take a network and write it to an XML file """

    @staticmethod
    def appendToFile(net, filename):
        """ append the network to an existing xml file """
        w = NetworkWriter(filename, newfile = False)
        netroot = w.newRootNode('Network')
        w.writeNetwork(net, netroot)
        w.save()

    @staticmethod
    def writeToFile(net, filename):
        """ write the network as a new xml file """
        w = NetworkWriter(filename, newfile = True)
        netroot = w.newRootNode('Network')
        w.writeNetwork(net, netroot)
        w.save()

    def writeNetwork(self, net, netroot):
        """ write a Network into a new XML node """
        netroot.setAttribute('name', net.name)
        netroot.setAttribute('class', canonicClassString(net))
        if net.argdict:
            self.writeArgs(netroot, net.argdict)

        # the modules
        mods = self.newChild(netroot, 'Modules')
        # first write the input modules (in order)
        for im in net.inmodules:
            self.writeModule(mods, im, True, im in net.outmodules)
        # now the output modules (in order)
        for om in net.outmodules:
            if om not in net.inmodules:
                self.writeModule(mods, om, False, True)
        # now the rest
        for m in net.modulesSorted:
            if m not in net.inmodules and m not in net.outmodules:
                self.writeModule(mods, m, False, False)

        # the motherconnections
        if len(net.motherconnections) > 0:
            mothers = self.newChild(netroot, 'MotherConnections')
            for m in net.motherconnections:
                self.writeBuildable(mothers, m)

        # the connections
        conns = self.newChild(netroot, 'Connections')
        for m in net.modulesSorted:
            for c in net.connections[m]:
                self.writeConnection(conns, c, False)
        if hasattr(net, "recurrentConns"):
            for c in net.recurrentConns:
                self.writeConnection(conns, c, True)

    def writeModule(self, rootnode, m, inmodule, outmodule):
        if isinstance(m, Network):
            mnode = self.newChild(rootnode, 'Network')
            self.writeNetwork(m, mnode)
        else:
            mnode = self.writeBuildable(rootnode, m)
        if inmodule:
            mnode.setAttribute('inmodule', 'True')
        elif outmodule:
            mnode.setAttribute('outmodule', 'True')

    def writeConnection(self, rootnode, c, recurrent):
        mnode = self.writeBuildable(rootnode, c)
        if recurrent:
            mnode.setAttribute('recurrent', 'True')

    def writeBuildable(self, rootnode, m):
        """ store the class (with path) and name in a new child. """
        mname = m.__class__.__name__
        mnode = self.newChild(rootnode, mname)
        mnode.setAttribute('name', m.name)
        mnode.setAttribute('class', canonicClassString(m))
        if m.argdict:
            self.writeArgs(mnode, m.argdict)
        if m.paramdim > 0 and not isinstance(m, SharedConnection):
            self.writeParams(mnode, m.params)
        return mnode

    def writeArgs(self, node, argdict):
        """ write a dictionnary of arguments """
        for name, val in list(argdict.items()):
            if val != None:
                tmp = self.newChild(node, name)
                if isclass(val):
                    s = canonicClassString(val)
                else:
                    s = getattr(val, 'name', repr(val))
                tmp.setAttribute('val', s)

    def writeParams(self, node, params):
        # TODO: might be insufficient precision
        pnode = self.newChild(node, 'Parameters')
        self.addTextNode(pnode, str(list(params)))
