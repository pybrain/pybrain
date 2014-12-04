from __future__ import print_function

#!/usr/bin/env python
""" A simple example on how to use the GaussianProcess class
in pybrain, for one and two dimensions. """

__author__ = "Thomas Rueckstiess, ruecksti@in.tum.de"

from pybrain.auxiliary import GaussianProcess
from pybrain.datasets import SupervisedDataSet
from scipy import mgrid, sin, cos, array, ravel
from pylab import show, figure

ds = SupervisedDataSet(1, 1)
gp = GaussianProcess(indim=1, start=-3, stop=3, step=0.05)
figure()

x = mgrid[-3:3:0.2]
y = 0.1*x**2 + x + 1
z = sin(x) + 0.5*cos(y)

ds.addSample(-2.5, -1)
ds.addSample(-1.0, 3)
gp.mean = 0

# new feature "autonoise" adds uncertainty to data depending on
# it's distance to other points in the dataset. not tested much yet.
# gp.autonoise = True

gp.trainOnDataset(ds)
gp.plotCurves(showSamples=True)

# you can also test the gp on single points, but this deletes the
# original testing grid. it can be restored with a call to _buildGrid()
print(gp.testOnArray(array([[0.4]])))


# --- example on how to use the GP in 2 dimensions

ds = SupervisedDataSet(2,1)
gp = GaussianProcess(indim=2, start=0, stop=5, step=0.25)
figure()

x,y = mgrid[0:5:4j, 0:5:4j]
z = cos(x)*sin(y)
(x, y, z) = list(map(ravel, [x, y, z]))

for i,j,k in zip(x, y, z):
    ds.addSample([i, j], [k])

print("preparing plots. this can take a few seconds...")
gp.trainOnDataset(ds)
gp.plotCurves()

show()