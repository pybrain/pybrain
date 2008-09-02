from pybrain.datasets import SequentialDataSet
from pybrain.auxiliary import GaussianProcess
from episodic import EpisodicExperiment
from scipy import mgrid
from pylab import *

__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

class ModelExperiment(EpisodicExperiment):
    """ An experiment that learns a model of its (action, state) pair
        with a Gaussian Process for each dimension of the state.
    """

    def __init__(self, task, agent):
        EpisodicExperiment.__init__(self, task, agent)
        
        # create model and trainer
        self.modelds = SequentialDataSet(1, 1)
        self.model = [GaussianProcess(indim=1, start=0, stop=500, step=20) for _ in range(self.task.indim)]
        
        # change hyper parameters for all gps
        for m in self.model:
            m.hyper = (20, 2.0, 0.01)
            m.autonoise = True
        
    def doEpisodes(self, number = 1):
        """ returns the rewards of each step as a list """

        all_rewards = []

        for dummy in range(number):
            self.stepid = 0
            rewards = []
            # the agent is informed of the start of the episode
            self.agent.newEpisode()
            self.task.reset()
            while not self.task.isFinished():
                r = self._oneInteraction()
                rewards.append(r)
            all_rewards.append(rewards)
        
            # train model
            self.modelds.clear()
        
        for i in range(self.agent.history.getNumSequences()):
            seq = self.agent.history.getSequence(i)
            state, action, dummy, dummy = seq
            
            l = len(action)
            inp = map(lambda x: int(floor(x)), mgrid[0:l-1:10j])
            self.modelds.setField('input', array([inp]).T)
            
            # only update y-direction
            # tar = state[inp, 1]
            # self.modelds.setField('target', array([tar]).T)
            # self.model[1].addDataset(self.modelds)
            
            # add training data to all gaussian processes
            for i,m in enumerate(self.model):
                tar = state[inp, i]
                self.modelds.setField('target', array([tar]).T)
                m.addDataset(self.modelds)
                
        [m._calculate() for m in self.model]
        self.model[1].plotCurves()
            
        return all_rewards

    def _oneInteraction(self):
        self.stepid += 1
        obs = self.task.getObservation()
        self.agent.integrateObservation(obs)
        action = self.agent.getAction()
        self.task.performAction(action)

        # predict with model
        modelobs = array([0, 0, 0])

        if self.stepid < self.model[0].stop:
            steps = self.model[0].step
            # linear interpolation between two adjacent gp states
            try:      
                modelobs = [ (1.0-float(self.stepid%steps)/steps) * self.model[i].pred_mean[int(floor(float(self.stepid)/steps))] +
                             (float(self.stepid%steps)/steps) * self.model[i].pred_mean[int(ceil(float(self.stepid)/steps))]
                             for i in range(len(action)) ]
            except IndexError:            
                modelobs = [self.model[i].pred_mean[int(floor(float(self.stepid/10)))] for i in range(len(action))]
        
        # tell environment about model obs
        self.task.env.model = [modelobs]
    
        reward = self.task.getReward()
        self.agent.giveReward(reward)
        return reward