__author__ = 'Tom Schaul, tom@idsia.ch'

from swiping import SwipingNetwork
from pybrain.structure.modules import BiasUnit
from pybrain.structure.connections.shared import MotherConnection, SharedFullConnection
from pybrain.utilities import iterCombinations, tupleRemoveItem


class BorderSwipingNetwork(SwipingNetwork):
    """ Expands the swiping network architecture by border units (bias) and connections. """
   
    def __init__(self, inmesh = None, hiddenmesh = None, outmesh = None, predefined = {}, **args):
        if inmesh != None:
            args['dims'] = inmesh.dims
        SwipingNetwork.__init__(self, **args)
        if inmesh != None:
            self._verifyDimensions(inmesh, hiddenmesh, outmesh)
            self._buildSwipingStructure(inmesh, hiddenmesh, outmesh, predefined)
            self._buildBorderStructure(inmesh, hiddenmesh, outmesh, predefined)
            self.sortModules()
   
    def _buildBorderStructure(self, inmesh, hiddenmesh, outmesh, predefined = {}, **args):
        SwipingNetwork.__init__(self, inmesh, hiddenmesh, outmesh, predefined, **args)
        self.addModule(BiasUnit(name = 'bias'))
        
        # build the motherconnections for the borders
        if not 'bordconns' in predefined:
            predefined['bordconns'] = {}
        for dim, maxval in enumerate(self.dims):
            if dim > 0 and self.symmetricdimensions:
                predefined['bordconns'][dim] = predefined['bordconns'][0]
            elif dim not in predefined['bordconns']:
                predefined['bordconns'][dim] = {}
            tmp = predefined['bordconns'][dim]
            if len(self.dims) == 1:
                tmp[()] = MotherConnection(hiddenmesh.componentIndim, name = 'bconn')
            for t in iterCombinations(tupleRemoveItem(self.dims, dim)):                    
                tc = self._canonicForm(t, dim)
                if t != tc:
                    # the connections from the borders are symetrical, so we need seperate ones only up to the middle                
                    tmp[t] = tmp[tc]
                elif t not in tmp:
                    tmp[t] = MotherConnection(hiddenmesh.componentIndim, name = 'bconn'+str(dim)+str(t))
                
        # link the bordering units to the bias, using the correct connection
        for dim, maxval in enumerate(self.dims):            
            for unit in self._iterateOverUnits():
                bconn = predefined['bordconns'][dim][tupleRemoveItem(unit, dim)]
                hunits = []
                if unit[dim] == 0:
                    for swipe in range(self.swipes):
                        if (swipe/2**dim) % 2 == 0:
                            hunits.append(tuple(list(unit)+[swipe]))
                if unit[dim] == maxval-1:
                    for swipe in range(self.swipes):
                        if (swipe/2**dim) % 2 == 1:
                            hunits.append(tuple(list(unit)+[swipe]))
                for hunit in hunits:
                    self.addConnection(SharedFullConnection(bconn, self['bias'], hiddenmesh[hunit]))                

    def _canonicForm(self, tup, dim):
        """ determine if there is a symetrical tuple of lower coordinates
        @param dim: the removed coordinate. """
        canonic = []
        for dim, maxval in enumerate(tupleRemoveItem(self.dims, dim)):
            canonic.append(min(maxval-tup[dim], tup[dim]))
        return tuple(canonic)
