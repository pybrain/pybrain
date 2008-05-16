__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import array, dot, sqrt, sin, cos, diag, pi
from numpy.random import randn, multivariate_normal
from scipy.linalg import svd
from pybrain.tools.functions import multivariateCauchy
from pybrain.tools.plotting import FitnessPlotter

angle = pi*0.7
rot = array([[cos(angle), sin(angle)],[sin(-angle), cos(-angle)]])
F = FitnessPlotter(lambda x, y: 1)
mu = array([2, 2])
diags = array([1, 0.01])
sigma = dot(diag(diags), rot)
#sigma = array([[1., -1.],[-1., 0.9]])

nb = 1000
# plot a number of samples drawn according to a multi-variate distribution
samples = map(lambda x: multivariate_normal(mu, sigma), range(nb))
F.addSamples(samples, color = 'r')


u, s, d = svd(sigma)
    
# compare with decomposing, drawing from univariate and recomposing
def multivariate_normal2():
    return mu + dot(d, dot(sqrt(s)*randn(len(mu)), u)) + 1
     
samples = map(lambda x: multivariate_normal2(), range(nb))
#samples = map(lambda x: multivariateCauchy(mu, sigma, onlyDiagonal= True), range(nb))
F.addSamples(samples, color = 'b')

F.addCovEllipse(sigma, mu, color = 'y')


F.plotAll(popup = True)
