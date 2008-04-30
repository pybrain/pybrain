""" plot the re-read results produced by femexperiments. """

__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import array

from nesexperiments import pickleReadDict
from pybrain.tools.plotting import plotFitnessProgession

#fname, cut, bsize = 'dim15results-b', 10, 50
fname, cut, bsize = 'dim5results', 9, 25

if __name__ == '__main__':
    folder = '../temp/fem/' 
    readin = pickleReadDict(folder+fname)
    onlymuEvals = {}
    for k, val in readin.items():
        onlymuEvals[k[:-cut]] = map(lambda x: -array(x), val[2])
    plotFitnessProgession(onlymuEvals, batchsize = bsize, 
                          #onlysuccessful = False,
                          #varyplotsymbols = True,
                          )
 