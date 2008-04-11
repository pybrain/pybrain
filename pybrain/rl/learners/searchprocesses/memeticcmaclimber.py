from scipy import zeros, ravel

from pybrain.rl.evolvables import MaskedParameters
from pybrain.rl.learners import CMAES
from searchprocess import SearchProcess
from pybrain.rl.environments.functions.episodicevaluators import EpisodicEvaluator
from pybrain.rl.environments.functions import FunctionEnvironment
from pybrain.rl.environments.functions import OppositeFunction


class MemeticCMAClimber(SearchProcess):
    """A combination of CMA-ES with the memetic climber"""
    
    localSteps = 50
    
    def __init__(self, evolvable, task, localSteps = None, desiredFitness = None):
        """ @param localSteps: nb of weight mutations before a mask mutation happens. """
        assert isinstance(evolvable, MaskedParameters)
        self.desiredFitness = desiredFitness
        if localSteps != None:
            self.localSteps = localSteps
        self.steps = 0
        self.evolvable = evolvable
        self.task = task
        self.evaluator = EvaluatorWrapper(self.evolvable, self.task)
        
    def _oneStep(self, verbose = False):
        self.bestFitness = self.evaluator.controlledExecute(self.evaluator.getx0())    
        self.challenger = self.evolvable.copy()
        self.challenger.mutate(mask=True,weights=False)
        self.evaluator = EvaluatorWrapper(self.challenger, self.task)
        cma = CMAES(OppositeFunction(self.evaluator), x0=self.evaluator.getx0(), maxEvals=self.localSteps)
        best = ravel(cma.optimize())
        challengerFitness = self.evaluator.controlledExecute(best)
        
        if verbose:
            print self.steps, challengerFitness, self.bestFitness
            #print 'new mask:', map(int, self.challenger.mask), sum(self.challenger.maskableParams*self.challenger.mask)
            #print 'old mask:', map(int, self.evolvable.mask), sum(self.evolvable.maskableParams*self.evolvable.mask)
            
        if (challengerFitness >= self.bestFitness):
            self.bestFitness = challengerFitness
            self.evolvable = self.challenger
        else:
            self.evaluator = EvaluatorWrapper(self.evolvable, self.task)
            
    
class EvaluatorWrapper(EpisodicEvaluator):
    """Wraps an evaluator that contains a maskedparameters object to allow 
    passing the array in the maskedparameters without (masked) zeros"""
    
    def __init__(self, module, task):
        assert isinstance (module, MaskedParameters)
        self.module = module
        self.task = task
        FunctionEnvironment.__init__(self, xdim = sum(module.mask))
    
    def f(self, x):
        """ set x as parameters in the module, then run one episode """
        paramcount = 0
        l = len(self.module.maskableParams)
        #print l, x.shape
        #assert l == len(x)
        for i in range(l):
            if self.module.mask[i] == True:
                self.module.maskableParams[i] = x[paramcount]
                paramcount += 1
        self.module._applyMask()
        res = self.task.evaluateModule(self.module)
        return res
    
    def getx0(self):
        x = zeros(sum(self.module.mask))
        paramcount = 0
        for i in range(len(self.module.maskableParams)):
            if self.module.mask[i] == True:
                x[paramcount] = self.module.maskableParams[i] 
                paramcount += 1    
        return x
    
    
    