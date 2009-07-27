#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-

#@PydevCodeAnalysisIgnore

import doctest
import math
import random
import sys

from math import log

import scipy

from arac.pybrainbridge import _FeedForwardNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import MdrnnLayer, LinearLayer, IdentityConnection, \
    SigmoidLayer, SoftmaxLayer
from pybrain.rl.environments.functions import FunctionEnvironment
from pybrain.rl.learners.blackboxoptimizers.fem import FEM
from pybrain.rl.learners.blackboxoptimizers.cmaes import CMAES
from pybrain.rl.learners.blackboxoptimizers.neldermead import NelderMead
from pybrain.rl.learners.blackboxoptimizers.bfgs import BFGS
from pybrain.rl.learners.blackboxoptimizers.pso import ParticleSwarmOptimizer, ring

from pybrainexamples.datasets.mnist import makeMnistDataSets


class MnistMdrnn(FunctionEnvironment):

    def __init__(self):
        self.width = 28
        self.height = 28
        self.testds, self.trainds = \
            makeMnistDataSets('/Users/bayerj/Desktop/MNIST/')
        
        # Initialize MDRNN
        self.net = _FeedForwardNetwork()
        inlayer = LinearLayer(self.width * self.height)
        hiddenlayer = MdrnnLayer(timedim=2, 
                                 shape=(self.width, self.height), 
                                 blockshape=(1, 1), 
                                 hiddendim=4,
                                 outsize=10,
                                 name='mdrnn')
        outlayer = SigmoidLayer(self.width * self.height * 10)
        con1 = IdentityConnection(inlayer, hiddenlayer)
        con2 = IdentityConnection(hiddenlayer, outlayer)
        self.net.addInputModule(inlayer)
        self.net.addModule(hiddenlayer)
        self.net.addOutputModule(outlayer)
        self.net.addConnection(con1)
        self.net.addConnection(con2)

        self.net.sortModules()
    
    def f(self, x):
        self.net['mdrnn'].params[:] = x
        error = 0
        for (inpt, target) in self.trainds:
            output = self.net.activate(inpt)
            indic = output.reshape(self.width * self.height, 10).sum(axis=0)
            diff = indic - target
            error += scipy.inner(diff, diff)
        return error / len(self.trainds)
        
    def eval(self, x):
        self.net['mdrnn'].params[:] = x
        correct_answers = 0.
        for (inpt, target) in self.trainds:
            indic = output.reshape(self.width * self.height, 1).sum(axis=0)
            klass = output.argmax()
            target = target.argmax()
            if klass == target:
                correct_answers += 1
        return correct_answers / len(self.trainds)


def listener(evaluator, evaluation):
    print "%.5f" % evaluation,
    correct_rate = evalCorrectAnswers(optimizer.bestEvaluable)
    print "Correct rate: %.3f" %  correct_rate
    print "`" * 80
    
    
def pso(env, evaluable):
    dim = scipy.size(evaluable)
    opt = ParticleSwarmOptimizer(env, evaluable, 
                                 50, neighbourfunction=ring,
                                 listener=listener,
                                 boundaries=[(-10, 10)] * dim)
                                  
    return opt
    
def cma(env, evaluable):
    return CMAES(env, evaluable, listener=listener)
    
    
def fem(env, evaluable):
    opt = FEM(env, evaluable, listener=listener)
    opt.forgetFactor = 0.7
    opt.batchSize = 25
    return opt
    
    
def neldermead(env, evaluable):
    opt = NelderMead(env, evaluable, listener=listener)
    opt.verbose = True
    return opt

def bfgs(env, evaluable):
    opt = BFGS(env, evaluable, listener=listener)
    opt.verbose = True
    return opt
    

def evalCorrectAnswers(evaluable):
    global mdrnn_env
    mdrnn_env.eval(evaluable)
    
            
if __name__ == '__main__':
    # Don't execute if the tests fail
    f, _ = doctest.testmod()
    if f > 0:
        sys.exit()
    
    # Read the optimizer to use form the command line
    Optimizer = globals()[sys.argv[1]]
    
    # Always use the same randomization
    mdrnn_env = MnistMdrnn()
    optimizer = Optimizer(mdrnn_env, mdrnn_env.net['mdrnn'].params)
    optimizer.minimize = True
    try:
        optimizer.learn()
    except KeyboardInterrupt:
        print mdrnn_env.eval(optimizer.bestEvaluable)
