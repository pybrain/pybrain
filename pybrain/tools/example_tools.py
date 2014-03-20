#########################################################################
# Reinforcement Tools for printing, saving and loading for RL examples 
# 
# Requirements: scipy
#
# Author: Frank Sehnke, sehnke@in.tum.de
#########################################################################

from cPickle import load, dump
from scipy import array, sqrt
from pylab import errorbar, show


class ExTools():
    agent = None
    loadName = "none.wgt"
    saveName = "none.wgt"
    resuName = "none.dat"
    rl = []
    rll = []

    def __init__(self, batch = 2, prnts = 1, kind = "optimizer"):
        self.batch = batch
        self.prnts = prnts
        self.kind = kind

    # Method for loading a weight matrix and initialize the network
    def loadWeights(self, filename):
        filepointer = file(filename)
        self.agent.learner.current = load(filepointer)
        filepointer.close()
        self.agent.learner.gd.init(self.agent.learner.current)
        self.agent.learner.epsilon = 0.2
        self.agent.learner.initSigmas()

    # Method for saving the weight matrix    
    def saveWeights(self, filename, w):
        filepointer = file(filename, 'w+')
        dump(w, filepointer)
        filepointer.close()

    # Method for saving the weight matrix    
    def saveResults(self, filename, results):
        filepointer = file(filename, 'w+')
        dump(results, filepointer)
        filepointer.close()

    def printResults(self,resList, runs, updates):
        if self.kind == "optimizer":
            rLen = len(resList)
            avReward = array(resList).sum()/rLen
            print("Parameters:", self.agent.learner._bestFound())
            print("Experiment:", runs,
                " Evaluation:", (updates+1)*self.batch*self.prnts,
                " BestReward:", self.agent.learner.bestEvaluation,
                " AverageReward:", avReward)
            print()
            self.rl.append(avReward)
        else:
            avReward = resList
            #print("Parameters: ", self.agent.learner._bestFound())
            print(
                "Step: ", runs, "/", (updates+1)*self.batch*self.prnts,
                #"Best: ", self.agent.learner.bestEvaluation,
                "Base: ", avReward)
            #print()
            self.rl.append(avReward)

    def addExps(self):
        self.rll.append(self.rl)
        self.rl = []

    def showExps(self):
        nEx = len(self.rll)
        self.rll = array(self.rll)
        r = self.rll.sum(axis=0)/nEx
        d = self.rll-r
        v = (d**2).sum(axis=0)
        v = v/nEx
        stand = sqrt(v)
        errorbar(array(range(len(self.rll[0])))*self.prnts*self.batch+self.prnts*self.batch,r,stand)
        show()
