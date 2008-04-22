""" A script for trying plenty of variations of networks/algorithms on the cart-pole balancing task 

Short variation of cartpolecomarisons.py
"""

__author__ = 'Tom Schaul, tom@idsia.ch'

import time
from scipy import rand
from random import shuffle

from pybrain import buildNetwork, FullConnection
from pybrain.tests.helpers import sortedProfiling
from pybrain.utilities import storeCallResults
from pybrain.structure.evolvables.cheaplycopiable import CheaplyCopiable
from nesexperiments import pickleDumpDict
from pybrain.rl.tasks.polebalancing import CartPoleTask
from pybrain.rl.learners import CMAES, ES, HillClimber, WeightGuessing
from pybrain.rl.learners.meta import MemeticSearch, InnerMemeticSearch



# desired performance (cumulative reward of the task)
desiredValue = 50000

# maximal number of episodes
maxEvals = 30000

# initial weight range
weightRange = 1.

# extra (cartesian) observations?
extra = False

# task: non-markov double pole balancing for 100000 iterations - fixed starting conditions
thetask = CartPoleTask(numPoles = 2, markov = False, extraObservations = extra)

# networks: normal MLP with recurrent connections on hidden layer
# 2x2 variations - with and without bias, 3 or 6 hidden neurons.
networks = []
for h in [3,6]:
    for b in [False, True]:
        net = buildNetwork(thetask.outdim, h, thetask.indim, bias = b)
        net.addRecurrentConnection(FullConnection(net['hidden0'], net['hidden0'], name = 'rec'))
        net.sortModules()
        net.name = 'net'+str(h)+str(b)
        net._params *= weightRange
        net = CheaplyCopiable(net)
        net.name = net.name[:-5]
        networks.append(net)
        


learners = []
for n in networks:
    learners.append((n.name+'-Random', 
                     WeightGuessing(thetask, n)))
    learners.append((n.name+'-Climber', 
                     HillClimber(thetask, n)))
    learners.append((n.name+'-CMA', 
                     CMAES(thetask, n)))
    learners.append((n.name+'-ES-50+50', 
                     ES(thetask, n)))
    learners.append((n.name+'-ES-5+5', 
                     ES(thetask, n, mu = 5, lambada = 5)))
    learners.append((n.name+'-Memetic', 
                     MemeticSearch(thetask, n, localSteps = 50)))
    learners.append((n.name+'-MemeticCMA', 
                     MemeticSearch(thetask, n, CMAES, localSteps = 200)))
    learners.append((n.name+'-MemeticES-50+50', 
                     MemeticSearch(thetask, n, ES, localSteps = 500)))
    learners.append((n.name+'-MemeticES-5+5', 
                     MemeticSearch(thetask, n, ES, localSearchArgs = {'mu': 5, 'lambada': 5})))
    learners.append((n.name+'-InnerMemetic', 
                     InnerMemeticSearch(thetask, n)))
    learners.append((n.name+'-InnerMemeticES-5+5', 
                     InnerMemeticSearch(thetask, n, ES, localSearchArgs = {'mu': 5, 'lambada': 5})))
    

# shuffling of network types and algorithms on each run?
if True:
    shuffle(learners)
    
    
allfits = storeCallResults(thetask)
    
    
# run all algorithms on all networks
def runAll(repeat = 1):
    for dummy in range(repeat):
        for n, L in learners:
            try:
                if extra:
                    n = 'extra'+n
                print n,
                del allfits[0:]
                start = time.time()
                best, bestfit = L.learn(maxEvals)
                best._resetBuffers()
                t = time.time() - start
                print 'episodes', len(allfits), 'best', bestfit, 'time', t
                # storage
                pickleDumpDict('../temp/cartpole2/'+n+'--'+str(int(rand(1)*90000)+10000), {'net': best,
                                                                                           'allfits': allfits})
                
            except Exception, e:
                print 'Something went wrong', e
                
                
                
    
    
if __name__ == '__main__':
    if True:
        runAll(100)
    else:
        sortedProfiling('runAll(1)')