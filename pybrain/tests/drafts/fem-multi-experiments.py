""" Script for (online) FEM experiments on continous multimodal benchmark functions """

import time
from scipy import randn, rand

from pybrain.utilities import storeCallResults
from pybrain.rl.learners import FEM
from pybrain.rl.environments.functions import RotateFunction, TranslateFunction
from pybrain.rl.environments.functions.multimodal import GriewankFunction
from pybrain.rl.environments.functions.multimodal import AckleyFunction
from pybrain.rl.environments.functions.multimodal import RastriginFunction
from pybrain.rl.environments.functions.multimodal import WeierstrassFunction


# storage tag for this batch
tag = 'good-multi-'

basefunctions = [GriewankFunction, AckleyFunction,RastriginFunction, WeierstrassFunction,
                 ]
dim = 2

defaultargs = {'batchsize': 25,
               'onlineLearning': True,
               'ranking': 'toplinear',
               'topselection': 10,
               'maxupdate': 0.02,
               'maxEvaluations': 5000,
               }


particulars = {}

distances = [1., 10., 100.]


def runAll(repeat = 1):
    for dummy in range(repeat):
        for dist in distances:
            for basef in basefunctions:
                f = TranslateFunction(RotateFunction(basef(dim)), distance = dist)
                x0 = randn(dim)
                f.desiredValue = -0.01
    
                res = storeCallResults(f)
                
                args = defaultargs.copy()
                if (basef, dim) in particulars:
                    for k, val in particulars[(basef, dist)].items():
                        args[k] = val
                
                
                name = tag+'-'+basef.__name__+str(dist)
                id = int(rand(1)*90000)+10000
                print name, id #, args
                start = time.time()
                try:
                    l = FEM(f, x0, **args)
                    best, bestfit = l.learn()
                
                    used = time.time() - start
                    evals = len(res)
                    print 'result', bestfit, 'in', evals, 'evalautions, using', used, 'seconds.'
                    if bestfit > f.desiredValue:
                        print 'FOUND'
                        name += '-S'
                    else:
                        print 'NOT FOUND'
                    print 
                    
                    # storage
                    from nesexperiments import pickleDumpDict
                    pickleDumpDict('../temp/fem/'+name+'-'+str(id), {'allevals': res, 'muevals': l.muevals, 
                                                                     'args': args})
                except Exception, e:
                    print 'Ooops', e
            
if __name__ == '__main__':
    runAll(100000)
