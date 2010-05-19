__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pylab import plot, figure, ion, Line2D, draw, arange
from pybrain.rl.environments.renderer import Renderer
import threading
import time


class SimpleRenderer(Renderer):
    def __init__(self):
        Renderer.__init__(self)

        self.dataLock = threading.Lock()
        self.stopRequest = False
        self.pathx = []
        self.pathy = []
        self.f = None
        self.min = -1
        self.max = 1
        self.fig = None
        self.color = 'red'

    def setFunction(self, f, rmin, rmax):
        self.dataLock.acquire()
        self.f = f
        self.min = rmin
        self.max = rmax
        self.dataLock.release()

    def updateData(self, data):
        self.dataLock.acquire()
        (x, y) = data
        self.pathx.append(x)
        self.pathy.append(y)
        self.dataLock.release()

    def reset(self):
        self.dataLock.acquire()
        self.pathx = []
        self.pathy = []
        self.dataLock.release()

    def stop(self):
        self.dataLock.acquire()
        self.stopRequest = True
        self.dataLock.release()

    def start(self):
        self.drawPlot()
        Renderer.start(self)

    def drawPlot(self):
        ion()
        self.fig = figure()
        axes = self.fig.add_subplot(111)

        # draw function
        xvalues = arange(self.min, self.max, 0.1)
        yvalues = map(self.f, xvalues)
        plot(xvalues, yvalues)

        # draw exploration path
        self.line = Line2D([], [], linewidth=3, color='red')
        axes.add_artist(self.line)
        self.line.set_clip_box(axes.bbox)

        # set axes limits
        axes.set_xlim(min(xvalues) - 0.5, max(xvalues) + 0.5)
        axes.set_ylim(min(yvalues) - 0.5, max(yvalues) + 0.5)

    def _render(self):
        while not self.stopRequest:
            self.dataLock.acquire()
            self.line.set_data(self.pathx, self.pathy)
            self.line.set_color(self.color)
            figure(self.fig.number)
            draw()
            self.dataLock.release()

            time.sleep(0.05)
        self.stopRequest = False

