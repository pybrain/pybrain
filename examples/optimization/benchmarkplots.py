#!/usr/bin/env python
""" A little script to do contour-plots of a couple of the widely used optimization benchmarks. """

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.tools.plotting.fitnesslandscapes import FitnessPlotter
from pybrain.rl.environments.functions.multimodal import BraninFunction,\
    RastriginFunction
from pybrain.rl.environments.functions.unimodal import RosenbrockFunction,\
    GlasmachersFunction
import pylab


FitnessPlotter(BraninFunction(), -5., 10., 0., 15.).plotAll(popup = False)
pylab.title('Branin')

FitnessPlotter(RastriginFunction(2)).plotAll(popup = False)
pylab.title('Rastrigin')

FitnessPlotter(RosenbrockFunction(2), -2., 2., -2., 2.).plotAll(popup = False)
pylab.title('Rosenbrock')

FitnessPlotter(GlasmachersFunction(2), -2., 2., -2., 2.).plotAll(popup = False)
pylab.title('Glasmachers')

pylab.show()