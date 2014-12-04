"""

    >>> from pybrain.tools.rlgluebridge import adaptAgent
    >>> from rlglue.types import Action as RLGlueAction
    >>> from rlglue.types import Observation as RLGlueObservation

Let's take a pseudo agent to test with and make an object of it

    >>> klass = adaptAgent(PseudoPybrainAgent)
    >>> rlglue_agent = klass()

We can access the attributes of the original agent via the held attribute
`agent`

    >>> rlglue_agent
    <pybrain.tools.rlgluebridge.RlglueAgentAdapter object at ...>
    >>> type(rlglue_agent)
    <class 'pybrain.tools.rlgluebridge.RlglueAgentAdapter'>

Now let's see how the observations, actions and rewards are proxied to the
inner agent

    >>> rlglue_agent.agent_init()
    I was reseted

We need an observation to test with

    >>> obs = RLGlueObservation()
    >>> obs.doubleArray = [3.14, 42]

A first step

    >>> action = rlglue_agent.agent_start(obs)
    I saw [  3.14  42.  ]
    I did [ 2.7 -1. ]
    >>> action
    <rlglue.types.Action instance at ...>
    >>> action.doubleArray[0]
    2.7000000000000002
    >>> action.doubleArray[1]
    -1.0

Another step

    >>> action = rlglue_agent.agent_step(1, obs)
    I was given 1.00
    I saw [  3.14  42.  ]
    I did [ 2.7 -1. ]
    >>> action
    <rlglue.types.Action instance at ...>
    >>> action.doubleArray[0]
    2.7000000000000002
    >>> action.doubleArray[1]
    -1.0

And a last step

    >>> rlglue_agent.agent_end(0)
    I was given 0.00
    I got a new episode
    I was reseted


Now let's have a look on how we can save statistics of an agent running
rlglue.

    >>> from pybrain.tools.rlgluebridge import BenchmarkingAgent
    >>> agent = PseudoPybrainAgent()
    >>> agent = BenchmarkingAgent(agent)

    >>> agent.integrateObservation('obs')
    I saw obs
    >>> agent.getAction()
    I did [ 2.7 -1. ]
    array([ 2.7, -1. ])
    >>> agent.giveReward(0)
    I was given 0.00

    >>> agent.integrateObservation('obs2')
    I saw obs2
    >>> agent.getAction()
    I did [ 2.7 -1. ]
    array([ 2.7, -1. ])
    >>> agent.giveReward(1)
    I was given 1.00

    >>> agent.newEpisode()
    I got a new episode
    >>> print(agent.benchmark)
    Average Reward: dim(2, 1)
    [[ 0.5]]
    <BLANKLINE>
    Episode Length: dim(2, 1)
    [[ 2.]]
    <BLANKLINE>
    <BLANKLINE>

"""

__author__ = 'Justin Bayer, bayerj@in.tum.de'
_dependencies = ['rlglue']

from scipy import array

from pybrain.rl.agents import LearningAgent
from pybrain.tests import runModuleTestSuite


class PseudoPybrainAgent(LearningAgent):
    """A little Agent that follows the pybrain API for testing."""

    attribute = "my-attribute"
    learner = None
    learning = False

    def __init__(self): pass

    def integrateObservation(self, obs):
        print(("I saw %s" % obs))

    def getAction(self):
        action = array([2.7, -1])
        print(("I did %s" % action))
        return action

    def giveReward(self, r):
        """ Reward or punish the agent.

            :key r: reward, if C{r} is positive, punishment if C{r} is
                      negative
            :type r: double
        """
        print(("I was given %.2f" % float(r)))

    def newEpisode(self):
        print("I got a new episode")

    def reset(self):
        print("I was resetted")


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

