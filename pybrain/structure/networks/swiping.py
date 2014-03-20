__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.connections.shared import MotherConnection, SharedFullConnection
from pybrain.utilities import iterCombinations

# TODO: special treatment for multi-dimensional lstm cells: identity connections on state buffers


class SwipingNetwork(FeedForwardNetwork):
    """ A network architecture that establishes shared connections between ModuleMeshes (of identical dimensions)
    so that the behavior becomes equivalent to one unit (in+hidden+out components at the same coordinate) swiping
    over a multidimensional input space and producing a multidimensional output. """

    # if all dimensions should be considered symmetric, their weights are shared
    symmetricdimensions = True

    # should the forward and backward directions be symmetric (for each dimension)?
    symmetricdirections = True

    # dimensions of the swiping grid
    dims = None

    def __init__(self, inmesh=None, hiddenmesh=None, outmesh=None, predefined=None, **args):
        if predefined != None:
            self.predefined = predefined
        else:
            self.predefined = {}
        super(SwipingNetwork, self).__init__(**args)

        # determine the dimensions
        if inmesh != None:
            self.setArgs(dims=inmesh.dims)
        elif self.dims == None:
            raise Exception('No dimensions specified, or derivable')

        self.swipes = 2 ** len(self.dims)

        if inmesh != None:
            self._buildSwipingStructure(inmesh, hiddenmesh, outmesh)
            self.sortModules()

    def _verifyDimensions(self, inmesh, hiddenmesh, outmesh):
        """ verify dimension matching between the meshes """
        assert self.dims == inmesh.dims
        assert outmesh.dims == self.dims
        assert tuple(hiddenmesh.dims[:-1]) == self.dims, '%s <-> %s' % (
                hiddenmesh.dims[:-1], self.dims)
        assert hiddenmesh.dims[-1] == self.swipes
        assert min(self.dims) > 1

    def _buildSwipingStructure(self, inmesh, hiddenmesh, outmesh):
        """
        :key inmesh: a mesh of input units
        :key hiddenmesh: a mesh of hidden units
        :key outmesh: a mesh of output units
        """
        self._verifyDimensions(inmesh, hiddenmesh, outmesh)

        # add the modules
        for c in inmesh:
            self.addInputModule(c)
        for c in outmesh:
            self.addOutputModule(c)
        for c in hiddenmesh:
            self.addModule(c)

        # create the motherconnections if they are not provided
        if 'inconn' not in self.predefined:
            self.predefined['inconn'] = MotherConnection(inmesh.componentOutdim * hiddenmesh.componentIndim, name='inconn')
        if 'outconn' not in self.predefined:
            self.predefined['outconn'] = MotherConnection(outmesh.componentIndim * hiddenmesh.componentOutdim, name='outconn')
        if 'hconns' not in self.predefined:
            self.predefined['hconns'] = {}
            for s in range(len(self.dims)):
                if self.symmetricdirections:
                    if s > 0 and self.symmetricdimensions:
                        self.predefined['hconns'][s] = self.predefined['hconns'][0]
                    else:
                        self.predefined['hconns'][s] = MotherConnection(hiddenmesh.componentIndim *
                                                        hiddenmesh.componentOutdim, name='hconn' + str(s))
                else:
                    for dir in ['-', '+']:
                        if s > 0 and self.symmetricdimensions:
                            self.predefined['hconns'][(s, dir)] = self.predefined['hconns'][(0, dir)]
                        else:
                            self.predefined['hconns'][(s, dir)] = MotherConnection(hiddenmesh.componentIndim *
                                                        hiddenmesh.componentOutdim, name='hconn' + str(s) + dir)

        # establish the connections
        for unit in self._iterateOverUnits():
            for swipe in range(self.swipes):
                hunit = tuple(list(unit) + [swipe])
                self.addConnection(SharedFullConnection(self.predefined['inconn'], inmesh[unit], hiddenmesh[hunit]))
                self.addConnection(SharedFullConnection(self.predefined['outconn'], hiddenmesh[hunit], outmesh[unit]))
                # one swiping connection along every dimension
                for dim, maxval in enumerate(self.dims):
                    # determine where the swipe is coming from in this direction:
                    # swipe directions are towards higher coordinates on dim D if the swipe%(2**D) = 0
                    # and towards lower coordinates otherwise.
                    previousunit = list(hunit)
                    if (swipe / 2 ** dim) % 2 == 0:
                        previousunit[dim] -= 1
                        dir = '+'
                    else:
                        previousunit[dim] += 1
                        dir = '-'

                    if self.symmetricdirections:
                        hconn = self.predefined['hconns'][dim]
                    else:
                        hconn = self.predefined['hconns'][(dim, dir)]

                    previousunit = tuple(previousunit)
                    if previousunit[dim] >= 0 and previousunit[dim] < maxval:
                        self.addConnection(SharedFullConnection(hconn, hiddenmesh[previousunit], hiddenmesh[hunit]))

    def _iterateOverUnits(self):
        """ iterate over the coordinates defines by the ranges of self.dims. """
        return iterCombinations(self.dims)

    def _printPredefined(self, dic=None, indent=0):
        """ print the weights of the Motherconnections in the self.predefined dictionary (recursively)"""
        if dic == None:
            dic = self.predefined
        for k, val in sorted(dic.items()):
            print(' ' * indent, k,)
            if isinstance(val, dict):
                print(':')
                self._printPredefined(val, indent + 2)
            elif isinstance(val, MotherConnection):
                print(val.params)
            else:
                print(val)

