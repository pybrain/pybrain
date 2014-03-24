from __future__ import with_statement


__author__ = 'Daan Wierstra and Tom Schaul'

import scipy

import logging
from itertools import chain

from pybrain.structure.moduleslice import ModuleSlice
from pybrain.structure.modules.module import Module
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.structure.connections.shared import SharedConnection
from pybrain.structure.evolvables.evolvable import Evolvable


class NetworkConstructionException(Exception):
    """Exception that indicates that the structure of the network is invalid."""


class Network(Module, ParameterContainer):
    """Abstract class for linking different modules with connections."""

    __offset = 0

    def __getOffset(self):
        return self.__offset

    def __setOffset(self, x):
        self.__offset = x
        for m in self.modules:
            m.offset = x

    offset = property(__getOffset, __setOffset)


    def __init__(self, name=None, **args):
        ParameterContainer.__init__(self, **args)
        self.name = name
        # Due to the necessity of regular testing for membership, modules are
        # stored in a set.
        self.modules = set()
        self.modulesSorted = []
        # The connections are stored in a dictionary: the key is the module
        # where the connection leaves from, the value is a list of the
        # corresponding connections.
        self.connections = {}
        self.inmodules = []
        self.outmodules = []
        # Special treatment of weight-shared connections.
        self.motherconnections = []
        # This flag is used to make sure that the modules are reordered when
        # new connections are added.
        self.sorted = False

    def __str__(self):
        sortedByName = lambda itr: sorted(itr, key=lambda i: i.name)

        params = {
            'name': self.name,
            'modules': self.modulesSorted,
            'connections':
                sortedByName(chain(*(sortedByName(self.connections[m])
                                     for m in self.modulesSorted))),
        }

        s = ("%(name)s\n" +
             "   Modules:\n    %(modules)s\n" +
             "   Connections:\n    %(connections)s\n") % params

        return s

    def __getitem__(self, name):
        """Return the module with the given name."""
        for m in self.modules:
            if m.name == name:
                return m
        return None

    def _containerIterator(self):
        """Return an iterator over the non-empty ParameterContainers of the
        network.

        The order IS deterministic."""
        for m in self.modulesSorted:
            if m.paramdim:
                yield m
            for c in self.connections[m]:
                if c.paramdim and not isinstance(c, SharedConnection):
                    yield c
        for mc in self.motherconnections:
            if mc.paramdim:
                yield mc

    def addModule(self, m):
        """Add the given module to the network."""
        if isinstance(m, ModuleSlice):
            m = m.base
        if m not in self.modules:
            self.modules.add(m)
        if not m in self.connections:
            self.connections[m] = []
        if m.paramdim > 0:
            m.owner = self
        if m.sequential and not self.sequential:
            logging.warning(
                ("Module %s is sequential, and added to a FFN. Are you sure " +
                "you know what you're doing?") % m)
        self.sorted = False

    def addInputModule(self, m):
        """Add the given module to the network and mark it as an input module.
        """
        if isinstance(m, ModuleSlice): m = m.base
        if m not in self.inmodules:
            self.inmodules.append(m)
        self.addModule(m)

    def addOutputModule(self, m):
        """Add the given module to the network and mark it as an output module.
        """
        if isinstance(m, ModuleSlice):
            m = m.base
        if m not in self.outmodules:
            self.outmodules.append(m)
        self.addModule(m)

    def addConnection(self, c):
        """Add the given connection to the network."""
        if not c.inmod in self.connections:
            self.connections[c.inmod] = []
        self.connections[c.inmod].append(c)
        if isinstance(c, SharedConnection):
            if c.mother not in self.motherconnections:
                self.motherconnections.append(c.mother)
                c.mother.owner = self
        elif c.paramdim > 0:
            c.owner = self
        self.sorted = False

    def _growBuffers(self):
        for m in self.modules:
            m._growBuffers()
        super(Network, self)._growBuffers()

    def reset(self):
        """Reset all component modules and the network."""
        Module.reset(self)
        for m in self.modules:
            m.reset()

    def _setParameters(self, p, owner=None):
        """ put slices of this array back into the modules """
        ParameterContainer._setParameters(self, p, owner)
        index = 0
        for x in self._containerIterator():
            x._setParameters(self.params[index:index + x.paramdim], self)
            index += x.paramdim

    def _setDerivatives(self, d, owner=None):
        """ put slices of this array back into the modules """
        ParameterContainer._setDerivatives(self, d, owner)
        index = 0
        for x in self._containerIterator():
            x._setDerivatives(self.derivs[index:index + x.paramdim], self)
            index += x.paramdim

    def _forwardImplementation(self, inbuf, outbuf):
        raise NotImplemented("Must be implemented by subclass.")

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        raise NotImplemented("Must be implemented by subclass.")

    def _topologicalSort(self):
        """Update the network structure and make .modulesSorted a topologically
        sorted list of the modules."""
        # Algorithm: R. E. Tarjan (1972), stolen from:
        #     http://www.bitformation.com/art/python_toposort.html

        # Create a directed graph, including a counter of incoming connections.
        graph = {}
        for node in self.modules:
            if node not in graph:
                # Zero incoming connections.
                graph[node] = [0]
        for c in chain(*self.connections.values()):
            graph[c.inmod].append(c.outmod)
            # Update the count of incoming arcs in outnode.
            graph[c.outmod][0] += 1

        # Find all roots (nodes with zero incoming arcs).
        roots = [node for (node, nodeinfo) in graph.items() if nodeinfo[0] == 0]

        # Make sure the ordering on all runs is the same.
        roots.sort(key=lambda x: x.name)

        # Repeatedly emit a root and remove it from the graph. Removing
        # a node may convert some of the node's direct children into roots.
        # Whenever that happens, we append the new roots to the list of
        # current roots.
        self.modulesSorted = []
        while len(roots) != 0:
            root = roots[0]
            roots = roots[1:]
            self.modulesSorted.append(root)
            for child in graph[root][1:]:
                graph[child][0] -= 1
                if graph[child][0] == 0:
                    roots.append(child)
            del graph[root]

        if graph:
            raise NetworkConstructionException("Loop in network graph.")

    def sortModules(self):
        """Prepare the network for activation by sorting the internal
        datastructure.

        Needs to be called before activation."""
        if self.sorted:
            return
        # Sort the modules.
        self._topologicalSort()
        # Sort the connections by name.
        for m in self.modules:
            self.connections[m].sort(key=lambda x: x.name)
        self.motherconnections.sort(key=lambda x: x.name)

        # Create a single array with all parameters.
        tmp = [pc.params for pc in self._containerIterator()]
        total_size = sum(scipy.size(i) for i in tmp)
        ParameterContainer.__init__(self, total_size)
        if total_size > 0:
            self.params[:] = scipy.concatenate(tmp)
            self._setParameters(self.params)

            # Create a single array with all derivatives.
            tmp = [pc.derivs for pc in self._containerIterator()]
            self.resetDerivatives()
            self.derivs[:] = scipy.concatenate(tmp)
            self._setDerivatives(self.derivs)

        # TODO: make this a property; indim and outdim are invalid before
        # .sortModules is called!
        # Determine the input and output dimensions of the network.
        self.indim = sum(m.indim for m in self.inmodules)
        self.outdim = sum(m.outdim for m in self.outmodules)

        self.indim = 0
        for m in self.inmodules:
            self.indim += m.indim
        self.outdim = 0
        for m in self.outmodules:
            self.outdim += m.outdim

        # Initialize the network buffers.
        self.bufferlist = []
        Module.__init__(self, self.indim, self.outdim, name=self.name)
        self.sorted = True

    def _resetBuffers(self, length=1):
        super(Network, self)._resetBuffers(length)
        for m in self.modules:
            m._resetBuffers(length)

    def copy(self, keepBuffers=False):
        if not keepBuffers:
            self._resetBuffers()
        cp = Evolvable.copy(self)
        if self.paramdim > 0:
            cp._setParameters(self.params.copy())
        return cp

    def convertToFastNetwork(self):
        """ Attempt to transform the network into a fast network. If fast networks are not available,
        or the network cannot be converted, it returns None. """

        from pybrain.structure.networks import FeedForwardNetwork, RecurrentNetwork
        try:
            from arac.pybrainbridge import _RecurrentNetwork, _FeedForwardNetwork #@UnresolvedImport
        except:
            print("No fast networks available.")
            return None

        net = self.copy()
        if isinstance(net, FeedForwardNetwork):
            cnet = _FeedForwardNetwork()
        elif isinstance(net, RecurrentNetwork):
            cnet = _RecurrentNetwork()

        for m in net.inmodules:
            cnet.addInputModule(m)
        for m in net.outmodules:
            cnet.addOutputModule(m)
        for m in net.modules:
            cnet.addModule(m)

        for clist in net.connections.values():
            for c in clist:
                cnet.addConnection(c)
        if isinstance(net, RecurrentNetwork):
            for c in net.recurrentConns:
                cnet.addRecurrentConnection(c)

        try:
            cnet.sortModules()
        except ValueError:
            print("Network cannot be converted.")
            return None

        cnet.owner = cnet
        return cnet
