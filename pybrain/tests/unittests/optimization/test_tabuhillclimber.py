# -*- coding: utf-8 -*-  
""" 

Example Usage: Runs a TabuHillClimber underconditions were it it virtually guanteed to find the global optimum if functioning correctly.                                                                                                                     
==================== 
>>> from pybrain.rl.environments.functions.unimodal import AttractiveSectorFunction
>>> from pybrain.optimization.tabusearch import TabuHillClimber
>>> testOptimizer=TabuHillClimber(AttractiveSectorFunction(),[2])      
>>> testOptimizer.maxLearningSteps=200
>>> def tabuGenerator(old,new):
...     return lambda input:abs(old.params[0]-new.params[0])*1.5>abs(old.params[0]-input.params[0])
... 
>>> testOptimizer.tabuSetUp(tabuGenerator,1)
>>> print testOptimizer.learn()
(array([0]), 9.9999999999998495e-91)
"""

from pybrain.tests import runModuleTestSuite
from pybrain.optimization.tabusearch import TabuHillClimber

if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))
