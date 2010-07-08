__author__ = 'Tom Schaul, tom@idsia.ch'

from pylab import figure, savefig, imshow, axes, axis, cm, show
from scipy import array, amin, amax


class ColorMap:
    def __init__(self, mat, cmap=None, pixelspervalue=20, minvalue=None, maxvalue=None):
        """ Make a colormap image of a matrix

        :key mat: the matrix to be used for the colormap.
        """
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

    def show(self):
        """ have the image popup """
        show()

    def save(self, filename):
        """ save colormap to file"""
        savefig(filename, fig=self.fig, facecolor='black', edgecolor='black')

