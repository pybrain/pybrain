__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from random import sample
from scipy import isscalar

from dataset import DataSet
from pybrain.utilities import fListToString
    
class SupervisedDataSet(DataSet): 
    """ SupervisedDataSet has 2 fields: input, target. It is mostly used for supervised learning,
        e.g. neural networks. The functions addSample and getSample are added for
        convenience as wrappers for addLinked and getLinked, to stay in the terminology of supervised learning."""
    
    def __init__(self, inp, target):
        """ initialize the supervised dataset.
            @param inp: the input dimension (scalar)
            @param target: the target dimension (scalar)
        """
        DataSet.__init__(self)
        if isscalar(inp):
            # add input and target fields and link them
            self.addField('input', inp)
            self.addField('target', target)
        else:
            self.setField('input', inp)
            self.setField('target', target)
        
        self.linkFields(['input', 'target'])    
        
        # reset the index marker
        self.index = 0
        
        # the input and target dimensions
        self.indim = self.getDimension('input')
        self.outdim = self.getDimension('target')
         
    def __reduce__(self):
        _, _, state, lst, dct = super(SupervisedDataSet, self).__reduce__()
        creator = self.__class__
        args = self.indim, self.outdim
        return creator, args, state, [], {}
        
    def addSample(self, inp, target):
        """ adds a new sample consisting of input, target.
            @param input: the input of the sample
            @param target: the target of the sample
        """
        self.appendLinked(inp, target)
    
    def getSample(self, index=None):
        """ This function is simply a wrapper function for the generic DataSet getLinked() function.
            @param index: the index of the row to be returned. if index=None, the current row is returned """
        return self.getLinked(index)
        
    def setField(self, label, arr, **kwargs): 
        """ sets the given array as the new array of field 'label'
            @param label: the name of the field
            @param arr: the new array for that field """
        DataSet.setField(self, label, arr, **kwargs)
        # refresh dimensions, in case any of these fields were modified
        if label == 'input':
            self.indim = self.getDimension('input')
        elif label == 'target':
            self.outdim = self.getDimension('target')
    
    def _provideSequences(self):
        """ return an iterator over sequence lists, although the dataset contains only single samples. """
        return iter(map(lambda x: [x], iter(self)))
    
    def evaluateMSE(self, f, **args):
        """ Evaluate the predictions of a function on the dataset
        and return the Mean Squared Error (incorporating importance). """
        ponderation = 0.
        totalError = 0
        for seq in self._provideSequences():
            e, p = self._evaluateSequence(f, seq, **args)
            totalError += e
            ponderation += p  
        assert ponderation > 0          
        return totalError/ponderation        
        
    def _evaluateSequence(self, f, seq, verbose = False):
        """ return the ponderated MSE over one sequence. """
        totalError = 0.
        ponderation = 0.
        for input, target in seq:
            res = f(input)
            e = 0.5 * sum((target-res).flatten()**2)
            totalError += e
            ponderation += len(target)
            if verbose:
                print     'out:    ', fListToString( list( res ) )
                print     'correct:', fListToString( target )
                print     'error: % .8f' % e
        return totalError, ponderation                
        
    def evaluateModuleMSE(self, module, averageOver = 1, **args):
        """ Evaluate the predictions of a module on a dataset
        and return the MSE (potentially average over a number of epochs). """
        res = 0.
        for dummy in range(averageOver):
            module.reset()
            res += self.evaluateMSE(module.activate, **args)
        return res/averageOver
        
    def splitWithProportion(self, proportion = 0.5):
        """ produce two new datasets, the first one containing the given fraction of the samples """
        leftIndices = sample(range(len(self)), int(len(self)*proportion))
        leftDs = self.copy()
        rightDs = self.copy()
        leftDs.clear()
        rightDs.clear()
        index = 0
        for sp in self:
            if index in leftIndices:
                leftDs.addSample(*sp)
            else:
                rightDs.addSample(*sp)
            index += 1
        return leftDs, rightDs
        
        
        
        