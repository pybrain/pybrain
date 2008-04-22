from pybrain.rl.experiments import Experiment
from pybrain.rl.environments.functions import SphereFunction
from pybrain.rl import FlatNetworkAgent
from pybrain.rl.tasks import Task


def testExperiment():
    # environment is a function
    F = SphereFunction(3)
    T = Task(F)
    T.getReward = lambda: 0
    
    # a simple agent
    A = FlatNetworkAgent(F.outdim, F.indim)
    
    # the function needs afirst observation
    initobs = [0]*F.outdim
    F.performAction(initobs)
    
    #connect them up
    E = Experiment(T, A)
    E.doInteractions(20)
    
if __name__ == '__main__':
    testExperiment()
    