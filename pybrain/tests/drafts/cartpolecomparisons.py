""" A script for trying plenty of variations of networks/algorithms on the cart-pole balancing task """

__author__ = 'Tom Schaul, tom@idsia.ch'

import time
from scipy import rand

from pybrain import buildNetwork, FullConnection
from pybrain.tests.helpers import sortedProfiling
from pybrain.utilities import storeCallResults
from pybrain.structure.evolvables.cheaplycopiable import CheaplyCopiable
from nesexperiments import pickleDumpDict
from pybrain.rl.tasks.polebalancing import CartPoleTask
from pybrain.rl.learners import CMAES, ES, HillClimber, WeightGuessing, WeightMaskGuessing
from pybrain.rl.learners.meta import MemeticSearch, InnerMemeticSearch, InnerInverseMemeticSearch
from pybrain.rl.learners.meta.inversememetic import InverseMemeticSearch


# desired performance (cumulative reward of the task)
desiredValue = 50000

# network parameters
weightRange = 1.
hidden = 4
bias = True


learners = [# previously done
            ('Random', WeightGuessing, {}),
            ('Climber', HillClimber, {}),
            ('CMA', CMAES, {}),
            ('ES-50+50', ES, {}),            
            ('ES-5+5', ES, {'mu' : 5, 'lambada': 5}),
            # failed/new
            ('LongMemetic', MemeticSearch, {'localSteps': 100}),
            ('MemeticCMA', MemeticSearch, {'localSearch': CMAES, 'localSteps': 500}),
            ('RandomMasks', WeightMaskGuessing, {}),
            ('LongInnerMemetic', InnerMemeticSearch, {'localSteps': 50}),
            ('ShortInverseMemetic', InverseMemeticSearch, {'localSteps': 10}),
            ('LongInverseMemetic', InverseMemeticSearch, {'localSteps': 100}),
            ('InnerInverseMemetic', InnerInverseMemeticSearch, {'localSteps': 5}),
            # doomed
            #('Memetic', MemeticSearch, {'localSteps': 10}),
            #('InnerMemetic', InnerMemeticSearch, {'localSteps': 5}),
            #('MemeticES-50+50', MemeticSearch, {'localSearch': ES, 'localSteps': 500}),
            #('MemeticES-5+5', MemeticSearch, {'localSearch': ES, 'localSearchArgs': {'mu': 5, 'lambada': 5}}),
            #('InnerMemeticES-5+5', InnerMemeticSearch, {'localSearch':ES, 
            #                                            'localSearchArgs': {'mu': 5, 'lambada': 5}}),
            ]
        
    
def runAll(maxEvals = 30000, repeat = 1000):
    for u in range(repeat*3):
        
        # tasks: non-markov double pole balancing for 100000 iterations - fixed starting conditions        
        if u % 3 == 0:
            task = CartPoleTask(numPoles = 2, markov = False)
        elif u % 3 == 1:
            task = CartPoleTask(numPoles = 2, markov = False, extraObservations = True)
        else:
            task = CartPoleTask(numPoles = 2, markov = False, extraRandoms = 4)
        
        allfits = storeCallResults(task)
        net = buildNetwork(task.outdim, hidden, task.indim, bias = bias, outputbias = False)
        net.addRecurrentConnection(FullConnection(net['hidden0'], net['hidden0'], name = 'rec'))
        net.sortModules()
        net.name = 'net'+str(hidden)+str(bias)
        net._params *= weightRange
        net = CheaplyCopiable(net)
        net.name = net.name[:-5]

        # run all algorithms on the task
        for ln, L, args in learners:
            try:
                if task.extraObservations:
                    name = 'extra-'+net.name+'-'+ln
                elif task.extraRandoms > 0:
                    name = 'rand-'+net.name+'-'+ln
                else:
                    name = 'base-'+net.name+'-'+ln
                id = int(rand(1)*90000)+10000
                print name, id
                del allfits[0:]
                
                start = time.time()
                net.randomize()
                best, bestfit = L(task, net, maxEvaluations = maxEvals, **args).learn()
                t = time.time() - start
                
                print ' '*20, 'episodes', len(allfits), 'best', bestfit, 'time', t
                
                # storage
                best._resetBuffers()
                pickleDumpDict('../temp/cartpole4/'+name+'--'+str(id), 
                               {'net': best, 'allfits': allfits, 'time': t})
                
            except Exception, e:
                print 'Something went wrong', e
                
                
                    
        
    
if __name__ == '__main__':
    if True:
        runAll()
    else:
        sortedProfiling('runAll(1000, 1)')