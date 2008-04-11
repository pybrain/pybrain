from __future__ import with_statement

__author__ = 'Daan Wierstra and Tom Schaul'

import cPickle
from scipy import zeros, size

from pybrain.structure.moduleslice import ModuleSlice
from pybrain.structure.modules.module import Module
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.utilities import combineLists, substitute
from pybrain.structure.connections.shared import SharedConnection


class Network(Module):
    """ A network is linking different modules with connections. """

    def __init__(self, name = None, **args):
        self.name = name
        self.setArgs(**args)
        # a sorted list of modules
        self.modules = []
        # the connections are stored in a dictionnary:  the key is the module where the
        # connection leaves from, the value is a list of the corresponding connections
        self.connections = {}
        self.recurrentConns = []
        self.inmodules = []
        self.outmodules = []
        # special treatment of weight-shared connections
        self.motherconnections = []
        # this flag is used to make sure that the modules are reordered when new connections are added
        self.sorted = False
        
    def __str__(self):
        s = self.name
        s += '\n  Modules:\n    '
        s += repr(self.modules)
        s += '\n  Connections:\n    '
        s += str(combineLists(map(lambda m: self.connections[m], self.modules)))

        if len(self.recurrentConns) > 0:
            s += '\n  Recurrent Connections:\n    '
            s += str(sorted(self.recurrentConns, key=lambda x: x.name))
        return s
        
    def __getitem__(self, name):
        """return the module with that name """
        for m in self.modules:
            if m.name == name:
                return m
        return None
        
    def _containerIterator(self):
        """ return an iterator over the parameter containers of the network, which are not empty. """
        for m in self.modules:
            if m.paramdim:
                yield m
            for c in self.connections[m]:
                if c.paramdim and not isinstance(c, SharedConnection):
                    yield c
        for c in self.recurrentConns:
            if c.paramdim and not isinstance(c, SharedConnection):
                yield c
        for mc in self.motherconnections:
            if mc.paramdim:
                yield mc
            
    def addModule(self, m):
        if isinstance(m, ModuleSlice): m = m.base
        if m not in self.modules:
            self.modules.append(m)
        if not m in self.connections:
            self.connections[m] = []
        if m.paramdim > 0:
            m.owner = self
        if m.sequential:
            self.sequential = True
        self.sorted = False

    def addInputModule(self, m):
        if isinstance(m, ModuleSlice): m = m.base
        if m not in self.inmodules:
            self.inmodules.append(m)
        self.addModule(m)
        
    def addOutputModule(self, m):
        if isinstance(m, ModuleSlice): m = m.base
        if m not in self.outmodules:
            self.outmodules.append(m)
        self.addModule(m)
        
    def addConnection(self, c):
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

    def addRecurrentConnection(self, c):
        self.sequential = True
        if isinstance(c, SharedConnection):
            if c.mother not in self.motherconnections:
                self.motherconnections.append(c.mother)
                c.mother.owner = self
        elif c.paramdim > 0:
            c.owner = self
        self.recurrentConns.append(c)
        self.sorted = False

    def reset(self):
        """ reset all component modules, and self. """
        Module.reset(self)
        for m in self.modules:
            m.reset()    
    
    def _setParameters(self, p, owner = None):        
        """ put slices of this array back into the modules """        
        ParameterContainer._setParameters(self, p, owner)
        index = 0
        for x in self._containerIterator():
            x._setParameters(self.params[index:index+x.paramdim], self)
            index += x.paramdim
    
    def _setDerivatives(self, d, owner = None):
        """ put slices of this array back into the modules """        
        ParameterContainer._setDerivatives(self, d, owner)
        index = 0
        for x in self._containerIterator():
            x._setDerivatives(self.derivs[index:index+x.paramdim], self)
            index += x.paramdim
        
    def _forwardImplementation(self, inbuf, outbuf):
        index = 0
        t = self.time
        for m in self.inmodules:
            m.inputbuffer[t] = inbuf[index:index + m.indim]
            index += m.indim
        
        if t > 0:
            for c in self.recurrentConns:
                c.forward(t-1, t)
        
        for m in self.modules:
            m.forward(t)
            for c in self.connections[m]:
                c.forward(t)
                
        index = 0
        for m in self.outmodules:
            outbuf[index:index + m.outdim] = m.outputbuffer[t]
            index += m.outdim
        
    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        index = 0
        t = self.time
        for m in self.outmodules:
            m.outputerror[t] = outerr[index:index + m.outdim]
            index += m.outdim
        
        if not self._isLastTimestep():
            for c in self.recurrentConns:
                c.backward(t, t+1)
        
        for m in reversed(self.modules):
            for c in self.connections[m]:
                c.backward(t)
            m.backward(t)
                
        index = 0
        for m in self.inmodules:
            inerr[index:index + m.indim] = m.inputerror[t]
            index += m.indim
        
    def _topologicalSort(self):
        """ update the network structure, and sort the modules topologically. """
        # Algorithm: R. E. Tarjan (1972), stolen from:
        #     http://www.bitformation.com/art/python_toposort.html
                
        # create a directed graph, including a counter of incoming connections
        graph = {}
        for node in self.modules:
            if not graph.has_key(node):
                # zero incoming connections
                graph[node] = [0]
        for c in combineLists(self.connections.values()):
            graph[c.inmod].append(c.outmod)
            # Update the count of incoming arcs in outnode.
            graph[c.outmod][0] +=1 

        # find all roots (nodes with zero incoming arcs).
        roots = [node for (node, nodeinfo) in graph.items() if nodeinfo[0] == 0]
        
        # we want the same ordering on all runs
        roots.sort(key = lambda x: x.name)        
        
        # repeatedly emit a root and remove it from the graph. Removing
        # a node may convert some of the node's direct children into roots.
        # Whenever that happens, we append the new roots to the list of
        # current roots.
        self.modules = []
        while len(roots) != 0:
            root = roots[0]
            roots = roots[1:]
            self.modules.append(root)
            for child in graph[root][1:]:
                graph[child][0] -= 1
                if graph[child][0] == 0:
                    roots.append(child)
            del graph[root]

        if len(graph) != 0:
            print graph
            raise Exception, 'There is a loop in the network!?'
        
    def sortModules(self):
        """ this method needs to be called after the network structure has changed, before it can be used. """
        if self.sorted:
            return
        # sort the modules
        self._topologicalSort()
        # also the connections
        for m in self.modules:
            self.connections[m].sort(key=lambda x: x.name)
            #print self.connections[m]
        self.recurrentConns.sort(key=lambda x: x.name)
        self.motherconnections.sort(key=lambda x: x.name)
            
            
        # create a single array with all parameters
        tmp = map(lambda pc: pc.getParameters(), self._containerIterator())
        self.paramdim = sum(map(size, tmp))
        self.params = zeros(self.paramdim)
        # TODO: speed-optimize, maybe
        index = 0
        for x in tmp:
            self.params[index:index+size(x)] = x
            index += size(x)
        self._setParameters(self.params)
        
        # same thing for derivatives
        tmp = map(lambda pc: pc.getDerivatives(), self._containerIterator())
        self.derivs = zeros(self.paramdim)
        index = 0
        for x in tmp:
            self.derivs[index:index+size(x)] = x
            index += size(x)
        self._setDerivatives(self.derivs)
        
        # determine the input and output dimensions of the network
        self.indim = 0
        for m in self.inmodules:
            self.indim += m.indim
        self.outdim = 0
        for m in self.outmodules:
            self.outdim += m.outdim
            
        # initialize the network buffers
        Module.__init__(self, self.indim, self.outdim, name = self.name)
        self.sorted = True

    def initParams(self, stdParams = 1.):
        ParameterContainer.initParams(self, self.paramdim, stdParams)
        # pass params and derivs down to each single module in network
        self._setParameters(self.params)
        self._setDerivatives(self.derivs)  
        
    def saveToFile(self, filename, protocol=0):
        """Save the current dataset to the given filename."""
        with file(filename, 'w+') as fp:
            cPickle.dump(self, fp, protocol=protocol)
