""" Script for (online) FEM experiments on pole-balancing """

import time
from scipy import rand

from pybrain.utilities import storeCallResults
from pybrain.rl.learners import FEM
from pybrain.rl.tasks.polebalancing.cartpoleenv import CartPoleTask
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.connections.full import FullConnection



# storage tag for this batch
tag = 'wewe'

args = {'batchsize': 50,
       'onlineLearning': True,
       'ranking': 'toplinear',
       'topselection': 5,
       'maxupdate': 0.05,
       'superelitist': True,
       'maxEvaluations': 10000,
       #'verbose': True,
       }


def runAll(repeat = 1):
    f = CartPoleTask(2, markov = False)
    net = buildNetwork(f.outdim, 3, f.indim, bias = False, outputbias = False)
    net.addRecurrentConnection(FullConnection(net['hidden0'], net['hidden0'], name = 'rec'))
    net.sortModules()

    for dummy in range(repeat):
        net.randomize()
        res = storeCallResults(f)
        
        name = tag+'-'+'polebalancing'
        id = int(rand(1)*90000)+10000
        print name, id, args
        start = time.time()
        try:
            l = FEM(f, net, **args)
            best, bestfit = l.learn()
            best._resetBuffers()
            used = time.time() - start
            evals = len(res)
            print 'result', bestfit, 'in', evals, 'evalautions, using', used, 'seconds.'
            if not max(res) > f.desiredValue:
                print 'NOT FOUND'
                name += '-fail'
            print
            
            # storage
            from nesexperiments import pickleDumpDict
            pickleDumpDict('../temp/fem/'+name+'-'+str(id), {'allevals': res, 'muevals': l.muevals, 
                                                             'args': args,
                                                             'net': best,
                                                             })
        except Exception, e:
            print 'Ooops', e
            
if __name__ == '__main__':
    runAll(100000)
