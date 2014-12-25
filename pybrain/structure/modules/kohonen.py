__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from scipy import random
from scipy.ndimage import minimum_position
from scipy import mgrid, zeros, tile, array, floor, sum

from pybrain.structure.modules.module import Module


class KohonenMap(Module):
    """ Implements a Self-Organizing Map (SOM), also known as a Kohonen Map.
        Clusters the inputs in unsupervised fashion while conserving their
        neighbourhood relationship on a 2-dimensional grid. There are two
        versions: With the outputFullMap option set to True, it outputs
        the full Kohonen map to the next layer, set to False it will only
        return 2 values: the x and y coordinate of the winner neuron. """

    def __init__(self, dim, nNeurons, name=None, outputFullMap=False):
        if outputFullMap:
            outdim = nNeurons ** 2
        else:
            outdim = 2
        Module.__init__(self, dim, outdim, name)

        # switch modes
        self.outputFullMap = outputFullMap

        # create neurons
        self.neurons = random.random((nNeurons, nNeurons, dim))
        self.difference = zeros(self.neurons.shape)
        self.winner = zeros(2)
        self.nInput = dim
        self.nNeurons = nNeurons
        self.neighbours = nNeurons
        self.learningrate = 0.01
        self.neighbourdecay = 0.9999

        # distance matrix
        distx, disty = mgrid[0:self.nNeurons, 0:self.nNeurons]
        self.distmatrix = zeros((self.nNeurons, self.nNeurons, 2))
        self.distmatrix[:, :, 0] = distx
        self.distmatrix[:, :, 1] = disty


    def _forwardImplementation(self, inbuf, outbuf):
        """ assigns one of the neurons to the input given in inbuf and writes
            the neuron's coordinates to outbuf. """
        # calculate the winner neuron with lowest error (square difference)
        self.difference = self.neurons - tile(inbuf, (self.nNeurons, self.nNeurons, 1))
        error = sum(self.difference ** 2, 2)
        self.winner = array(minimum_position(error))
        if not self.outputFullMap:
            outbuf[:] = self.winner


    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        """ trains the kohonen map in unsupervised manner, moving the
            closest neuron and its neighbours closer to the input pattern. """

        # calculate neighbourhood and limit to edge of matrix
        n = floor(self.neighbours)
        self.neighbours *= self.neighbourdecay
        tl = (self.winner - n)
        br = (self.winner + n + 1)
        tl[tl < 0] = 0
        br[br > self.nNeurons + 1] = self.nNeurons + 1

        # calculate distance matrix
        tempm = 1 - sum(abs(self.distmatrix - self.winner.reshape(1, 1, 2)), 2) / self.nNeurons
        tempm[tempm < 0] = 0
        distm = zeros((self.nNeurons, self.nNeurons, self.nInput))
        for i in range(self.nInput):
            distm[:, :, i] = tempm
            distm[:, :, i] = tempm

        self.neurons[tl[0]:br[0], tl[1]:br[1]] -= self.learningrate * self.difference[tl[0]:br[0], tl[1]:br[1]] * distm[tl[0]:br[0], tl[1]:br[1]]

