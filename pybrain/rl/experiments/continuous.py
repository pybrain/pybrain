__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from experiment import Experiment

class ContinuousExperiment(Experiment):
    """ The extension of Experiment to handle continuous tasks. """
    
    def doInteractionsAndLearn(self, number = 1):
        """ executes a number of steps while learning continuously.
            no reset is performed, such that consecutive calls to 
            this function can be made.
        """
        for dummy in range(number):
            self._oneInteraction()
            self.agent.learn()
