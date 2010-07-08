# some utility code for nicely plotting 3D images of function fitness landscapes.

__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import zeros, r_, cos, sin, pi, array, dot, sqrt, diag
from scipy.linalg import svd
from pylab import figure, plot, show, meshgrid, contour, savefig, colorbar
from pybrain.rl.environments.functions import FunctionEnvironment
from inspect import isclass


def plotCovEllipse(emat, center, segments=50, color='y', transp=1.):
    """ Plots a covariance ellipse. """
    # compute a nb of points on the ellipse
    ex = zeros(segments + 1)
    ey = zeros(segments + 1)
    u, s, d = svd(emat)
    sm = dot(d, dot(diag(sqrt(s)), u))
    for i in range(segments + 1):
        circlex = cos((2 * pi * i) / float(segments))
        circley = sin((2 * pi * i) / float(segments))
        ex[i] = center[0] + sm[0, 0] * circlex + sm[0, 1] * circley
        ey[i] = center[1] + sm[1, 0] * circlex + sm[1, 1] * circley

    # plot them
    plot([center[0]], [center[1]], '+', color=color, alpha=transp)
    plot(ex, ey, '-', color=color, alpha=transp)
    return ex, ey


class FitnessPlotter:
    """ plot the function's values in the rectangular region specified by ranges. By default, plot in [-1,1] """
    def __init__(self, f, xmin= -1, xmax=1, ymin= -1, ymax=1, precision=50, newfig=True,
                 colorbar=False, cblabel=None):
        """ :key precision: how many steps along every dimension """
        if isinstance(f, FunctionEnvironment):
            assert f.xdim == 2
            self.f = lambda x, y: f(array([x, y]))
        elif isclass(f) and issubclass(f, FunctionEnvironment):
            tmp = f(2)
            self.f = lambda x, y: tmp(array([x, y]))
        else:
            self.f = f

        self.precision = precision
        self.colorbar = colorbar
        self.cblabel = cblabel
        self.xs = r_[xmin:xmax:self.precision * 1j]
        self.ys = r_[ymin:ymax:self.precision * 1j]
        self.zs = self._generateValMap()
        if newfig:
            self.fig = figure()

    def _generateValMap(self):
        """ generate the function fitness values for the current grid of x and y """
        vals = zeros((len(self.xs), len(self.ys)))
        for i, x in enumerate(self.xs):
            for j, y in enumerate(self.ys):
                vals[j, i] = self.f(x, y)
        return vals

    def plotAll(self, levels=50, popup=True):
        """ :key levels: how many fitness levels should be drawn."""
        tmp = contour(self.xs, self.ys, self.zs, levels)
        if self.colorbar:
            cb = colorbar(tmp)
            if self.cblabel != None:
                cb.set_label(self.cblabel)

        if popup: show()

    def addSamples(self, samples, rescale=True, color=''):
        """plot some sample points on the fitness landscape.

        :key rescale: should the plotting ranges be adjusted? """
        # split samples into x and y
        sx = zeros(len(samples))
        sy = zeros(len(samples))
        for i, s in enumerate(samples):
            sx[i] = s[0]
            sy[i] = s[1]
        if rescale:
            self._rescale(min(sx), max(sx), min(sy), max(sy))

        plot(sx, sy, color + '+')

    def _rescale(self, xmin, xmax, ymin, ymax):
        self.xs = r_[min(xmin * 1.1, min(self.xs)):max(xmax * 1.1, max(self.xs)):self.precision * 1j]
        self.ys = r_[min(ymin * 1.1, min(self.ys)):max(ymax * 1.1, max(self.ys)):self.precision * 1j]
        self.zs = self._generateValMap()

    def addCovEllipse(self, emat, center, segments=50, rescale=True, color='c', transp=1.):
        """plot a covariance ellipse """
        ex, ey = plotCovEllipse(emat, center, segments, color, transp)
        if rescale:
            self._rescale(min(ex), max(ex), min(ey), max(ey))

    def saveAs(self, filename, format='.jpg'):
        savefig(filename + format)



