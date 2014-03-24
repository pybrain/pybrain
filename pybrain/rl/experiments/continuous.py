__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.experiments.experiment import Experiment


class ContinuousExperiment(Experiment):
    """ The extension of Experiment to handle continuous tasks. """

    def doInteractionsAndLearn(self, number = 1):
        """ Execute a number of steps while learning continuously.
            no reset is performed, such that consecutive calls to
            this function can be made.
        """
        for _ in range(number):
            self._oneInteraction()
            self.agent.learn()
        return self.stepid
