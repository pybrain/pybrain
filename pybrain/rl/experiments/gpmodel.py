__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.datasets import SequentialDataSet
from pybrain.auxiliary import GaussianProcess
from episodic import EpisodicExperiment
from scipy import mgrid, array, floor, c_, r_, reshape


class ModelExperiment(EpisodicExperiment):
    """ An experiment that learns a model of its (action, state) pair
        with a Gaussian Process for each dimension of the state.
    """

    def __init__(self, task, agent):
        EpisodicExperiment.__init__(self, task, agent)
        
        # create model and training set (action dimension + 1 for time)
        self.modelds = SequentialDataSet(self.task.indim + 1, 1)
        self.model = [GaussianProcess(indim=self.modelds.getDimension('input'), 
                                      start=(-10, -10, 0), stop=(10, 10, 300), step=(5, 5, 100)) 
                      for _ in range(self.task.outdim)]
        
        # change hyper parameters for all gps
        for m in self.model:
            m.hyper = (20, 2.0, 0.01)
            # m.autonoise = True
        
    def doEpisodes(self, number = 1):
        """ returns the rewards of each step as a list and learns
            the model for each rollout. 
        """

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
        
        # clear model dataset (to retrain it)
        self.modelds.clear()
        print "retrain gp"
        [m.trainOnDataset(self.modelds) for m in self.model]
        
        for i in range(self.agent.history.getNumSequences()):
            seq = self.agent.history.getSequence(i)
            state, action, dummy, dummy = seq
            
            l = len(action)
            index = map(lambda x: int(floor(x)), mgrid[0:l-1:5j])
            action = action[index, :]
            inp = c_[action, array([index]).T]
            self.modelds.setField('input', inp)
            
            # add training data to all gaussian processes
            for i,m in enumerate(self.model):
                tar = state[index, i]
                self.modelds.setField('target', array([tar]).T)
                m.addDataset(self.modelds)
        
        # print "updating GPs..."
        # [m._calculate() for m in self.model]
        # print "done."   
        
        return all_rewards

    def _oneInteraction(self):
        self.stepid += 1
        obs = self.task.getObservation()
        self.agent.integrateObservation(obs)
        action = self.agent.getAction()
        self.task.performAction(action)

        # predict with model
        #modelobs = array([0, 0, 0])
        
        # time dimension        
        # if self.stepid < self.model[0].stop:
        #     steps = self.model[0].step
        #     
        #     # linear interpolation between two adjacent gp states
        #     try:      
        #         modelobs = [ (1.0-float(self.stepid%steps)/steps) * self.model[i].pred_mean[int(floor(float(self.stepid)/steps))] +
        #                      (float(self.stepid%steps)/steps) * self.model[i].pred_mean[int(ceil(float(self.stepid)/steps))]
        #                      for i in range(self.task.outdim) ]
        #     except IndexError:
              
        action = r_[action, array([self.stepid])]
        action = reshape(action, (1, 3))
        modelobs = [self.model[i].testOnArray(action) for i in range(self.task.outdim)]
        
        # tell environment about model obs
        self.task.env.model = [modelobs]
    
        reward = self.task.getReward()
        self.agent.giveReward(reward)
        return reward