__author__ = 'Tom Schaul, tom@idsia.ch'

from pylab import figure, savefig, imshow, axes, axis, cm, show
from scipy import array, amin, amax, ndarray, reshape

from pybrain.supervised.trainers import Trainer
from pybrain.tools.customxml import NetworkReader
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.structure.connections.connection import Connection


class ColorMap:
    def __init__(self, mat, cmap=None, pixelspervalue=20, minvalue=None, maxvalue=None):
        """ Make a colormap image of a matrix or sequence of Matrix/Connection objects

        :key mat: the matrix to be used for the colormap.
        """
        if isinstance(mat, (ParameterContainer, Connection)):
            mat = reshape(mat.params, (mat.outdim, mat.indim))

        if not isinstance(mat, ndarray):
            raise ValueError("Don't know how to display a ColorMap for a matrix of type {}".format(type(mat)))
        if minvalue == None:
            minvalue = amin(mat)
        if maxvalue == None:
            maxvalue = amax(mat)
        if not cmap:
            cmap = cm.hot

        figsize = (array(mat.shape) / 100. * pixelspervalue)[::-1]
        self.fig = figure(figsize=figsize)
        axes([0, 0, 1, 1]) # Make the plot occupy the whole canvas
        axis('off')
        self.fig.set_size_inches(figsize)
        imshow(mat, cmap=cmap, clim=(minvalue, maxvalue), interpolation='nearest')

    def show(self, block=False):
        """ Display the last image drawn """
        try:
            show(block=block)
        except:
            show()

    def save(self, filename):
        """ save colormap to file"""
        savefig(filename, fig=self.fig, facecolor='black', edgecolor='black')


class ColorMaps:
    def __init__(self, mat, cmap=None, pixelspervalue=20, minvalue=None, maxvalue=None):
        """ Make a colormap image of a matrix or sequence of Matrix/Connection objects

        :key mat: the matrix to be used for the colormap.
        """
        self.colormaps = []
        if isinstance(mat, basestring):
            try:
                mat = NetworkReader().readFrom(mat)
            except:
                pass
        # FIXME: what does NetworkReader output? (Module? Layer?) need to handle it's type here

        if isinstance(mat, Trainer):
            connections = mat.module.connections.values()
            mat = []
            for conlist in connections:
                mat += conlist

        try:
            mat = [v for (k, v) in mat.iteritems()]
            if not all(isinstance(x, (ParameterContainer, Connection)) for x in mat):
                raise ValueError("Don't know how to display ColorMaps for a sequence of type {}".format(type(mat)))
        except:
            pass
            # from traceback import print_exc
            # print_exc()
        if isinstance(mat, list):
            self.colormaps = [ColorMap(m, cmap=cmap, pixelspervalue=pixelspervalue, minvalue=minvalue, maxvalue=maxvalue) for m in mat] 
        else:
            raise ValueError("Don't know how to display ColorMaps for a sequence of type {}".format(type(mat)))
       
    def show(self, block=False):
        """ Display the last image drawn """
        try:
            show(block=block)
        except:
            show()

    def save(self, filename):
        """ save colormaps to files"""
        for i, cm in enumerate(self.colormaps):
            cm.savefig('{:03d}-'.format(i) + filename, fig=self.fig, facecolor='black', edgecolor='black')

