__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import ones, dot

from pybrain.datasets.sequential import SequentialDataSet
from pybrain.utilities import fListToString


# CHECKME: does this provide for importance-datasets in the non-sequential case
# maybe there should be a second class - or another structure!


class ImportanceDataSet(SequentialDataSet):
    """ Allows setting an importance value for each of the targets of a sample. """

    def __init__(self, indim, targetdim):
        SequentialDataSet.__init__(self, indim, targetdim)
        self.addField('importance', targetdim)
        self.link.append('importance')

    def addSample(self, inp, target, importance=None):
        """ adds a new sample consisting of input, target and importance.

            :arg inp: the input of the sample
            :arg target: the target of the sample
            :key importance: the importance of the sample. If left None, the
                 importance will be set to 1.0
        """
        if importance == None:
            importance = ones(len(target))
        self.appendLinked(inp, target, importance)

    def _evaluateSequence(self, f, seq, verbose = False):
        """ return the importance-ponderated MSE over one sequence. """
        totalError = 0
        ponderation = 0.
        for input, target, importance in seq:
            res = f(input)
            e = 0.5 * dot(importance.flatten(), ((target-res).flatten()**2))
            totalError += e
            ponderation += sum(importance)
            if verbose:
                print(    'out:       ', fListToString(list(res)))
                print(    'correct:   ', fListToString(target))
                print(    'importance:', fListToString(importance))
                print(    'error: % .8f' % e)
        return totalError, ponderation

