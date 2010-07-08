__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.utilities import Named


class ModuleSlice(Named):
    """ A wrapper for using a particular input-output slice of a module's buffers.
    The constructors of connections between ModuleSlices need to ensure a correct use
    (i.e) do the slicing on the base module directly. """

    def __init__(self, base, inSliceFrom = 0, inSliceTo = None, outSliceFrom = 0, outSliceTo = None):
        """ :key base: the base module that is sliced """
        if isinstance(base, ModuleSlice):
            # tolerantly handle the case of a slice of another slice
            self.base = base.base
            self.inOffset = inSliceFrom + base.inSliceFrom
            self.outOffset = outSliceFrom + base.outSliceFrom
            if inSliceTo == None:
                inSliceTo = self.base.indim + base.inSliceFrom
            if outSliceTo == None:
                outSliceTo = self.base.outdim + base.outSliceFrom
            self.name = base.base.name
        else:
            self.base = base
            self.inOffset = inSliceFrom
            self.outOffset = outSliceFrom
            if inSliceTo == None:
                inSliceTo = self.base.indim
            if outSliceTo == None:
                outSliceTo = self.base.outdim
            self.name = base.name
        assert self.inOffset >= 0 and self.outOffset >= 0
        self.indim = inSliceTo - inSliceFrom
        self.outdim = outSliceTo - outSliceFrom
        self.name += ('-slice:('+str(self.inOffset)+','+str(self.indim+self.inOffset)+')('
                     +str(self.outOffset)+','+str(self.outdim+self.outOffset)+')')
        # some slicing is required
        assert self.indim+self.outdim < base.indim+base.outdim
