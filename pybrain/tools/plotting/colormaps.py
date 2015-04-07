__author__ = 'Tom Schaul, tom@idsia.ch'

from pylab import figure, savefig, imshow, axes, axis, cm, show
from scipy import array, amin, amax, ndarray, reshape

from pybrain.structure.networks import Network
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
        if isinstance(cmap, basestring) and cmap.strip():
            cmap = getattr(cm, cmap.lower().strip())
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
    def __init__(self, mat, cmap=None, pixelspervalue=20, minvalue=None, maxvalue=None, show=True, block=False):
        """ Make a colormap image of a matrix or sequence of Matrix/Connection objects

        :key mat: the matrix to be used for the colormap.
        :key cmap: the matplotlib colormap (color scale) to use ('hot', 'hot_r', 'gray', 'gray_r', 'hsv', 'prism', pylab.cm.hot, etc)
        """
        self.colormaps = []
        if isinstance(mat, basestring):
            try:
                #nn = NetworkReader(mat, newfile=False)
                mat = NetworkReader(mat, newfile=False).readFrom(mat)
            except:
                pass

        try:  # if isinstance(mat, Trainer):
            mat = mat.module
        except:
            pass

        if isinstance(mat, Network):
            # connections is a dict with key: value pairs of Layer: Connection (ParameterContainer)
            mat = [connection for connection in mat.connections.values() if connection]
        
            # connections = mat.module.connections.values()
            # mat = []
            # for conlist in connections:
            #     mat += conlist

        try:
            mat = [v for (k, v) in mat.iteritems()]
            if not any(isinstance(m, (ParameterContainer, Connection)) for m in mat):
                raise ValueError("Don't know how to display ColorMaps for a sequence of type {} containing key, values of type {}: {}".format(
                                 type(mat), *[type(m) for m in mat.iteritems().next()]))
        except AttributeError:
            pass
            # from traceback import print_exc
            # print_exc()
        if isinstance(mat, list):
            for m in mat:
                if isinstance(m, list):
                    if len(m) == 1:
                        m = m[0]
                    else:
                        raise ValueError("Don't know how to display a ColorMap for a list containing more than one matrix: {}".format([type(m) for m in mat]))
                try:
                    self.colormaps = [ColorMap(m, cmap=cmap, pixelspervalue=pixelspervalue, minvalue=minvalue, maxvalue=maxvalue) ]
                except ValueError:
                    self.colormaps = [ColorMap(m[0], cmap=cmap, pixelspervalue=pixelspervalue, minvalue=minvalue, maxvalue=maxvalue) ]
        else:
            self.colormaps = [ColorMap(mat)]
            # raise ValueError("Don't know how to display ColorMaps for a sequence of type {}".format(type(mat)))
        if show:
            self.show(block=block)
       
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

