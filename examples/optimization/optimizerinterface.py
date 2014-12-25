from __future__ import print_function

#!/usr/bin/env python
"""
Illustrating the interface of black-box optimizers on a few simple problems:
- how to initialize when:
    * optimizing the parameters for a function
    * optimizing a neural network controller for a task
- how to set meta-parameters
- how to learn
- how to interrupt learning and continue where you left off
- how to access the information gathered during learning
"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import array

from pybrain.optimization import * #@UnusedWildImport
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.rl.environments.functions.unimodal import TabletFunction
from pybrain.rl.environments.shipsteer.northwardtask import GoNorthwardTask
from pybrain.tools.shortcuts import buildNetwork


# this script uses a very simple learning algorithm: hill-climbing.
# all other black-box optimizers can be user in the same way.
# Try it yourself, change the following line to use e.g. GA, CMAES, MemeticSearch or NelderMead
algo = HillClimber
#algo = GA
#algo = MemeticSearch
#algo = NelderMead
algo = CMAES
print('Algorithm:', algo.__name__)


# ------------------------
# ---- Initialization ----
# ------------------------

# here's the default way of setting it up: provide a function and an initial point
f = TabletFunction(2)
x0 = [2.1, 4]
l = algo(f, x0)

# f can also be a simple lambda function
l = algo(lambda x: sum(x)**2, x0)

# in the case of continuous optimization, the initial point
# can be provided as a list (above), an array...
l = algo(f, array(x0))

# ... or a ParameterContainer
pcontainer = ParameterContainer(2)
pcontainer._setParameters(x0)
l = algo(f, pcontainer)

# the initial point can be omitted if:
# a) the problem dimension is specified manually
l = algo(f, numParameters = 2)

# b) the function is a FunctionEnvironment that specifies the problem dimension itself
l = algo(f)

# but if none is the case this leads to an error:
try:
    l = algo(lambda x: sum(x)**2)
except ValueError as e:
    print('Error caught:', e)

# Initialization can also take place in 2 steps, first with the settings and then with the
# evaluator function:
l = algo()
l.setEvaluator(f)

# Learning can only be called after the second step, otherwise:
l = algo()
try:
    l.learn(0)
except AssertionError as e:
    print('Error caught:', e)
l.setEvaluator(f)
# no error anymore
l.learn(0)

# a very similar interface can be used to optimize the parameters of a Module
# (here a neural network controller) on an EpisodicTask
task = GoNorthwardTask()
nnet = buildNetwork(task.outdim, 2, task.indim)
l = algo(task, nnet)

# Normally optimization algorithms have reasonable default settings (meta-parameters).
# In case you want to be more specific, use keyword arguments, like this:

l = ES(f, mu = 10, lambada = 20)
l = OriginalNES(f, batchSize = 25, importanceMixing = False)

# if you mistype a keyword, or specify one that is not applicable,
# you will see a warning (but the initialization still takes place, ignoring those).
l = algo(f, batchSise = 10, theMiddleOfTheTutorial = 'here')


# -----------------------
# ----   Learning    ----
# -----------------------

# Learning is even simpler:
print(l.learn(5))

# The return values are the best point found, and its fitness
# (the argument indicates the number of learning steps/generations).

# The argument is not mandatory, in that case it will run until
# one of the stopping criteria is reached. For example:
# a) maximal number of evaluations (accessible in .numEvaluations)
l = algo(f, maxEvaluations = 20)
l.learn()
print(l.learn(), 'in', l.numEvaluations, 'evaluations.')

# b) desiredValue
l = algo(f, desiredEvaluation = 10)
print(l.learn(), ': fitness below 10 (we minimize the function).')

# c) maximal number of learning steps
l = algo(f, maxLearningSteps = 25)
l.learn()
print(l.learn(), 'in', l.numLearningSteps, 'learning steps.')

# it is possible to continue learning from where we left off, for a
# specific number of additional learning steps:
print(l.learn(75), 'in', l.numLearningSteps, 'total learning steps.')

# Finally you can set storage settings and then access all evaluations made
# during learning, e.g. for plotting:
l = algo(f, x0, storeAllEvaluations = True, storeAllEvaluated = True, maxEvaluations = 150)
l.learn()
try:
    import pylab
    pylab.plot(list(map(abs,l._allEvaluations)))
    pylab.semilogy()
    pylab.show()
except ImportError as e:
    print('No plotting:', e)

