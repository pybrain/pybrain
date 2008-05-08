__author__ = 'Tom Schaul, tom@idsia.ch'

from experiment import Experiment


class EpisodicExperiment(Experiment):
    """ The extension of Experiment to handle episodic tasks. """
    
    def doEpisodes(self, number = 1):
        """ returns the rewards of each step as a list """
        start = self.stepid
        all_rewards = []
        for dummy in range(number):
            rewards = []
            # the agent is informed of the start of the episode
            self.agent.newEpisode()
            self.task.reset()
            while not self.task.isFinished():
                r = self._oneInteraction()
                rewards.append(r)
            all_rewards.append(rewards)
        return all_rewards
        