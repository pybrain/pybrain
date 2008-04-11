__author__ = 'Tom Schaul, tom@idsia.ch'

from experiment import Experiment


class EpisodicExperiment(Experiment):
    """ The extension of Experiment to handle episodic tasks. """

    def doEpisodes(self, number = 1):
        """ returns the number of interactions done """
        start = self.stepid
        for dummy in range(number):
            # the agent is informed of the start of the episode
            self.agent.newEpisode()
            self.task.reset()
            while not self.task.isFinished():
                self._oneInteraction()
        return self.stepid - start