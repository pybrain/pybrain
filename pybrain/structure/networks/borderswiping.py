__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import zeros

from pybrain.structure.networks.swiping import SwipingNetwork
from pybrain.structure.modules import BiasUnit
from pybrain.structure.connections.shared import MotherConnection, SharedFullConnection
from pybrain.utilities import iterCombinations, tupleRemoveItem, reachable, decrementAny


class BorderSwipingNetwork(SwipingNetwork):
    """ Expands the swiping network architecture by border units (bias) and connections. """

    # if this flag is set, we extrapolate the values of unknown border connection weights
    # by initializing them to the closest match.
    extrapolateBorderValues = True

    # all border weights the same?
    simpleborders = False

    def __init__(self, inmesh = None, hiddenmesh = None, outmesh = None, **args):
        if not self.symmetricdirections:
            raise NotImplementedError("BorderSwipingNetworks are currently limited so direction-symmetric weights.")
        if inmesh != None:
            args['dims'] = inmesh.dims
        SwipingNetwork.__init__(self, **args)
        if inmesh != None:
            self._buildBorderStructure(inmesh, hiddenmesh, outmesh)
            self.sortModules()

    def _buildBorderStructure(self, inmesh, hiddenmesh, outmesh):
        self._buildSwipingStructure(inmesh, hiddenmesh, outmesh)
        self.addModule(BiasUnit(name = 'bias'))

        # build the motherconnections for the borders
        if self.simpleborders:
            if not 'borderconn' in self.predefined:
                self.predefined['borderconn'] = MotherConnection(hiddenmesh.componentIndim, name = 'bconn')
        else:
            if not 'bordconns' in self.predefined:
                self.predefined['bordconns'] = {}
            for dim, maxval in enumerate(self.dims):
                if dim > 0 and self.symmetricdimensions:
                    self.predefined['bordconns'][dim] = self.predefined['bordconns'][0]
                elif dim not in self.predefined['bordconns']:
                    self.predefined['bordconns'][dim] = {}
                tmp = self.predefined['bordconns'][dim].copy()
                if len(self.dims) == 1 and () not in tmp:
                    tmp[()] = MotherConnection(hiddenmesh.componentIndim, name = 'bconn')
                for t in iterCombinations(tupleRemoveItem(self.dims, dim)):
                    tc = self._canonicForm(t, dim)
                    if t == tc and t not in tmp:
                        # the connections from the borders are symmetrical,
                        # so we need separate ones only up to the middle
                        tmp[t] = MotherConnection(hiddenmesh.componentIndim, name = 'bconn'+str(dim)+str(t))
                        if self.extrapolateBorderValues:
                            p = self._extrapolateBorderAt(t, self.predefined['bordconns'][dim])
                            if p != None:
                                tmp[t].params[:] = p
                self.predefined['bordconns'][dim] = tmp

        # link the bordering units to the bias, using the correct connection
        for dim, maxval in enumerate(self.dims):
            for unit in self._iterateOverUnits():
                if self.simpleborders:
                    bconn = self.predefined['borderconn']
                else:
                    tc = self._canonicForm(tupleRemoveItem(unit, dim), dim)
                    bconn = self.predefined['bordconns'][dim][tc]
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
        """ determine if there is a symmetrical tuple of lower coordinates

        :key dim: the removed coordinate. """
        if not self.symmetricdimensions:
            return tup
        canonic = []
        for dim, maxval in enumerate(tupleRemoveItem(self.dims, dim)):
            canonic.append(min(maxval-1-tup[dim], tup[dim]))
        return tuple(canonic)

    def _extrapolateBorderAt(self, t, using):
        """ maybe we can use weights that are similar to neighboring borderconnections
        as initialization. """
        closest = reachable(decrementAny, [t], using.keys())
        if len(closest) > 0:
            params = zeros(using[closest.keys()[0]].paramdim)
            normalize = 0.
            for c, dist in closest.items():
                params += using[c].params / dist
                normalize += 1./dist
            params /= normalize
            return params
        return None
