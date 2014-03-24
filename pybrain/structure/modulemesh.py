__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.utilities import iterCombinations, Named
from pybrain.structure.moduleslice import ModuleSlice


class ModuleMesh(Named):
    """ An multi-dimensional array of modules, accessible by their coordinates.
    All modules need to have the same indim and outdim """

    def __init__(self, constructor, dimensions, name = None, baserename = False):
        """:arg constructor: a constructor method that returns a module
        :arg dimensions: tuple of dimensions. """
        self.dims = dimensions
        if name != None:
            self.name = name
        # a dict where the tuple of coordinates is the key
        self.components = {}
        for coord in iterCombinations(self.dims):
            tmp = constructor()
            self.components[coord] = tmp
            tmp.name = self.name + str(coord)
            if baserename and isinstance(tmp, ModuleSlice):
                tmp.base.name = tmp.name
        self.componentIndim = tmp.indim
        self.componentOutdim = tmp.outdim

    @staticmethod
    def constructWithLayers(layerclass, layersize, dimensions, name = None):
        """ create the mesh using constructors that build layers of a specified size and class. """
        c = lambda: layerclass(layersize)
        return ModuleMesh(c, dimensions, name)

    @staticmethod
    def viewOnFlatLayer(layer, dimensions, name = None):
        """ Produces a ModuleMesh that is a mesh-view on a flat module. """
        assert max(dimensions) > 1, "At least one dimension needs to be larger than one."
        def slicer():
            nbunits = reduce(lambda x, y: x*y, dimensions, 1)
            insize = layer.indim / nbunits
            outsize = layer.outdim / nbunits
            for index in range(nbunits):
                yield ModuleSlice(layer, insize*index, insize*(index+1), outsize*index, outsize*(index+1))
        c = slicer()
        return ModuleMesh(lambda: c.next(), dimensions, name)

    def __iter__(self):
        for coord in iterCombinations(self.dims):
            yield self.components[coord]

    def __getitem__(self, coord):
        return self.components[coord]

