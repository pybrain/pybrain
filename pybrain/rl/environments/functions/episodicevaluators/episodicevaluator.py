__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.environments.functions import FunctionEnvironment


class EpisodicEvaluator(FunctionEnvironment):
    """ Treat a module and a task as a function: the input are the module's trainable parameters
    and the output is the total reward over an episode. """    
    
    desiredValue = None
    
    def __init__(self, module, task):
        self.module = module
        self.task = task
        FunctionEnvironment.__init__(self, xdim = module.paramdim)
            
    def f(self, x):
        """ set x as parameters in the module, then run one episode """
        self.module._setParameters(x)
        res = self.task.evaluateModule(self.module)
        return res
        
