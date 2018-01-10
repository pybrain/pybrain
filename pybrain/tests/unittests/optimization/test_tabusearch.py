# -*- coding: utf-8 -*-  
""" 

Example Usage:
====================
Run a TabuHillClimber under conditions were it is virtually guanteed to find the global optimum if functioning corr\
ectly.                
>>> from pybrain.rl.environments.functions.unimodal import AttractiveSectorFunction
>>> from pybrain.optimization.tabusearch import TabuHillClimber, TabuRandomSearch
>>> tabuHillClimber=TabuHillClimber(AttractiveSectorFunction(),[2])      
>>> tabuHillClimber.maxLearningSteps=200
>>> def tabuGenerator(old,new):
...     return lambda input:abs(old.params[0]-new.params[0])*1.5>abs(old.params[0]-input.params[0])
... 
>>> tabuHillClimber.tabuSetUp(tabuGenerator,1)
>>> print tabuHillClimber.learn()
(array([0]), 9.9999999999998495e-91)

Repeat for TabuRandomSearch
>>> from pybrain.rl.environments.functions.multimodal import FunnelFunction 
>>> tabuRandomSearch=TabuRandomSearch(FunnelFunction(2),[2,.5])
>>> tabuRandomSearch.tabuSetUp(tabuGenerator,1, maxTabuList=3,tabuList=[])
>>> tabuRandomSearch.maxLearningSteps=20
>>> tabuRandomSearch.maxEvaluations=None
>>> tabuRandomSearch.minimize=False
>>> print(tabuRandomSearch.learn())
(array([ 2. ,  0.5]), 4.25)
>>> tabuRandomSearch._setUp
True


"""

from pybrain.tests import runModuleTestSuite
from pybrain.optimization.tabusearch import TabuHillClimber

if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))
