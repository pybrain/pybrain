__author__ = 'Tom Schaul, tom@idsia.ch'

from pylab import figure, savefig, imshow, axes, axis, cm, show
from scipy import array, amin, amax
from matplotlib.pyplot import colorbar
from numpy.linalg import norm

from pybrain.structure.modules import KohonenMap


class ColorMap:
    def __init__(self, mat, cmap=None, pixelspervalue=20, normalize=False, interpolation='nearest',
                 bar=True, minvalue=None, maxvalue=None, name=None):
        """ Make a colormap image of a matrix

        :key mat: the matrix to be used for the colormap.
        :key normalize:
        """
        if normalize:
            mat_norm = norm(mat)
            mat = array(map(lambda x: map(lambda y: y / mat_norm, x), mat))
        if minvalue == None:
            minvalue = amin(mat)
        if maxvalue == None:
            maxvalue = amax(mat)
        if not cmap:
            cmap = cm.hot
        figsize = (array(mat.shape) / 100. * pixelspervalue)[::-1]
        self.fig = figure(figsize=figsize)
        axes([0, 0, 1, 1])  # Make the plot occupy the whole canvas
        axis('off')
        self.fig.set_size_inches(figsize)
        self.im = imshow(mat, cmap=cmap, clim=(minvalue, maxvalue), interpolation=interpolation)
        if bar:
            colorbar()
        if name is not None:
            self.fig.canvas.set_window_title(name)


    def show(self):
        """ have the image popup """
        show()

    def save(self, filename):
        """ save colormap to file"""
        savefig(filename, fig=self.fig, facecolor='black', edgecolor='black')


def show_map(som, interpolation='nearest'):
    assert isinstance(som, KohonenMap)
    keys = som.keys if som.keys is not None else [i for i in xrange(som.nInput)]
    for i in xrange(som.nInput):
        ColorMap(som.neurons[:, :, i], bar=True, name=keys[i], interpolation=interpolation)
    # show()