__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.utilities import abstractMethod
from pybrain.rl.learners.datasetlearner import DataSetLearner
from pybrain.optimization.optimizer import BlackBoxOptimizer
from scipy import zeros


class FDLearner(DataSetLearner, BlackBoxOptimizer):
    """ FDLearner is the base class for all Finite Difference Learners. It 
        implements basic common functionality for all FD learners, but 
        can't be used by itself. """
            
    def setData(self, ds):
        """ sets the dataset for the learner. """
        DataSetLearner.setData(self, ds)
        # add the field for parameter deltas to the dataset (unlinked)
        if self.module:
            self.ds.addField('deltas', self.module.paramdim)
    
    def setModule(self, module):
        """ sets the module for the learner and copies the initial parameters. """
        DataSetLearner.setModule(self, module)
        self.original = self.module.params.copy()
        if self.ds:
            self.ds.addField('deltas', self.module.paramdim)
    
    def learn(self):
        """ learn on the current dataset, for a single epoch
            @note: has to be implemented by all subclasses. """
        abstractMethod()
    
    def perturbate(self):
        """ perturb the parameters. 
            @note: has to be implemented by all subclasses. """
        abstractMethod()
        
    def disableLearning(self):
        """ disables learning and replaces the current (possibly perturbed)
            parameters with the original ones. """
        self.module._setParameters(self.original)
    
    def enableLearning(self):
        """ enables learning and changes the parameters back to how they were
            before disabling learning (by adding the latest delta to the parameters). """
            
        try:
            lastdelta = self.ds.getField('deltas')[-1,:]
        except IndexError:
            # deltas still empty, use zero vector instead
            lastdelta = zeros((1, self.module.paramdim))
        # revert parameters back to perturbed ones with last deltas
        self.module._setParameters(self.original + lastdelta)
                                     
