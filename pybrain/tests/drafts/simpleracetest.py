from pybrain.rl.environments.simplerace.simpleracetask import SimpleraceTask
from pybrain.rl.environments.simplerace.simplecontroller import SimpleController
from pybrain.rl.experiments import Experiment

def testSimpleRace ():
    print "starting"
    task = SimpleraceTask()
    while (True):
        task.performAction([0.5, 0.5])
        print 'obs', task.getObservation(), 'reward', task.getTotalReward()
    
def testSimpleController ():
    task = SimpleraceTask ()
    agent = SimpleController ()
    experiment = Experiment (task, agent)
    experiment.doInteractions(900)

if __name__ == '__main__':
    testSimpleController ()