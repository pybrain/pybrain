__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from episodic import EpisodicExperiment
from scipy import arange


class QueuedExperiment(EpisodicExperiment):
    """ This experiment type runs n episodes at the beginning, followed by a learning step.
        From then on it removes the oldest episode, learns a new one, and executes another
        training step with the n current episodes. This way, learning happens after each
        episode, and each episode is considered n times for learning until discarded. """

    def run(self, queuelength, learningcycles=-1):
        # fill the queue with given number of episodes
        self._fillQueue(queuelength)

        # start the queue loop
        if learningcycles == -1:
            while True:
                # indefinite learning
                self._stepQueueLoop()
        else:
            for _ in arange(learningcycles):
                # learn the given number of times
                self._stepQueueLoop()


    def _fillQueue(self, queuelength):
        # reset agent (empty queue)
        self.agent.reset()
        # fill queue with first n episodes
        self.doEpisodes(queuelength)


    def _stepQueueLoop(self):
        # let agent learn with full queue
        self.agent.learn()
        # remove oldest episode
        self.agent.history.removeSequence(0)
        # execute one new episode
        self.doEpisodes(1)

