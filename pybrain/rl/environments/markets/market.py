from pybrain.utilities import Named
from pybrain.rl.environments.environment import Environment

class Market(Environment, Named):
    """ Market environment has states for the EMA's

    No goal as it needs to only maximise returns
    No theseus also?
    No init pos?

    It then also has 3 actions for what to do based on the trend
    """

    # directions
    Long = 'long'
    Short = 'short'
    Wait = 'wait'

    allActions = [Long, Short, Wait]

    def __init__(self, **args):
        self.setArgs(**args)
        self.reset()

    def reset(self):
        """ return to initial position (stochastically): """

    def performAction(self, action):
        """ perform this action on env
        """

    def getSensors(self):
        """ no clue what this does
        """
