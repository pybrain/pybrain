__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.utilities import abstractMethod
from pybrain.rl.learners.rllearner import RLLearner
from scipy import zeros

class FDLearner(RLLearner):
    
    def __init__(self):
        # store original parameters in here
        self.original = None
    
    def setData(self, ds):
        RLLearner.setData(self, ds)
        # add the field for parameter deltas to the dataset (unlinked)
        if self.module:
            self.ds.addField('deltas', self.module.paramdim)
    
    def setModule(self, module):
        """ sets the module for the learner and copies the initial parameters. """
        RLLearner.setModule(self, module)
        self.original = self.module.params.copy()
        if self.ds:
            self.ds.addField('deltas', self.module.paramdim)
    
    def learn(self):
        """ learn on the current dataset, for a single epoch
            @note: has to be implemented by all subclasses. """
        abstractMethod()
    
    def perturbate(self):
        """ perturb the parameters. """
        pass
        
    def disableLearning(self):
        self.module._setParameters(self.original)
    
    def enableLearning(self):
        try:
            lastdelta = self.ds.getField('deltas')[-1,:]
        except IndexError:
            # deltas still empty, use zero vector instead
            lastdelta = zeros((1, self.module.paramdim))
        # revert parameters back to perturbed ones with last deltas
        self.module._setParameters(self.original + lastdelta)
                                     
