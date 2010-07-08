__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pylab import ion, figure, draw, Rectangle, Line2D
from scipy import cos, sin
from pybrain.rl.environments.renderer import Renderer
import threading
import time


class CartPoleRenderer(Renderer):
    def __init__(self):
        Renderer.__init__(self)

        self.dataLock = threading.Lock()
        self.angle = 0.0
        self.angle_vel = 0.0
        self.pos = 0.0
        self.pos_vel = 0.0
        self.stopRequest = False

        # some drawing constants
        self.cartheight = 0.2
        self.cartwidth = 1.0
        self.polelength = 0.5
        self.plotlimits = [-4, 4, -0.5, 3]
        self.box = None
        self.pole = None

    def updateData(self, data):
        self.dataLock.acquire()
        (self.angle, self.angle_vel, self.pos, self.pos_vel) = data
        self.dataLock.release()

    def stop(self):
        self.stopRequest = True

    def start(self):
        self.drawPlot()
        Renderer.start(self)

    def drawPlot(self):
        ion()
        fig = figure(1)
        # draw cart
        axes = fig.add_subplot(111, aspect='equal')
        self.box = Rectangle(xy=(self.pos - self.cartwidth / 2.0, -self.cartheight), width=self.cartwidth, height=self.cartheight)
        axes.add_artist(self.box)
        self.box.set_clip_box(axes.bbox)

        # draw pole
        self.pole = Line2D([self.pos, self.pos + sin(self.angle)], [0, cos(self.angle)], linewidth=3, color='black')
        axes.add_artist(self.pole)
        self.pole.set_clip_box(axes.bbox)

        # set axes limits
        axes.set_xlim(-2.5, 2.5)
        axes.set_ylim(-0.5, 2)

    def _render(self):
        while not self.stopRequest:
            if self.angle < 0.05 and abs(self.pos) < 0.05:
                self.box.set_facecolor('green')
            else:
                self.box.set_facecolor('blue')

            self.box.set_x(self.pos - self.cartwidth / 2.0)
            self.pole.set_xdata([self.pos, self.pos + self.polelength * sin(self.angle)])
            self.pole.set_ydata([0, self.polelength * cos(self.angle)])
            draw()
            time.sleep(0.05)
        self.stopRequest = False
