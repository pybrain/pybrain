""" Script for (online) FEM experiments on continous unimodal benchmark functions """

import time
from scipy import randn, rand

from pybrain.utilities import storeCallResults
from pybrain.rl.learners import FEM
from pybrain.rl.environments.functions import SphereFunction, RosenbrockFunction, CigarFunction, RotateFunction, TranslateFunction
from pybrain.rl.environments.functions.unimodal import SchwefelFunction
from pybrain.rl.environments.functions.unimodal import TabletFunction
from pybrain.rl.environments.functions.unimodal import DiffPowFunction
from pybrain.rl.environments.functions.unimodal import ElliFunction


# storage tag for this batch
tag = 'good'

basefunctions = [SphereFunction, CigarFunction, SchwefelFunction, TabletFunction, 
                 DiffPowFunction, ElliFunction, RosenbrockFunction
                 ]
dims = [5, 15]

defaultargs = {'batchsize': 25,
               'onlineLearning': True,
               'ranking': 'toplinear',
               'topselection': 10,
               'maxupdate': 0.02,
               }


particulars = {(SphereFunction, 5): {'maxupdate': 0.03},
               (ElliFunction, 5): {'maxupdate': 0.03},
               (DiffPowFunction, 5): {'maxupdate': 0.03},
               (SchwefelFunction, 5): {'maxupdate': 0.03},
               (CigarFunction, 5): {'maxupdate': 0.03},
               (TabletFunction, 5): {'maxupdate': 0.03},
               (RosenbrockFunction, 5): {'maxupdate': 0.01, 'maxEvaluations':100000},
               (RosenbrockFunction, 15): {'maxupdate': 0.005, 'maxEvaluations':0},
               }


def runAll(repeat = 1):
    for dummy in range(repeat):
        for dim in dims:
            for basef in basefunctions:
                f = TranslateFunction(RotateFunction(basef(dim)))
                x0 = randn(dim)
                f.desiredValue = -1e-10
    
                res = storeCallResults(f)
                
                args = defaultargs.copy()
                if dim == 15:
                    args['maxEvaluations'] = 50000
                else:
                    args['maxEvaluations'] = 30000
                
                if (basef, dim) in particulars:
                    for k, val in particulars[(basef, dim)].items():
                        args[k] = val
                
                
                name = tag+'-'+basef.__name__+str(dim)
                id = int(rand(1)*90000)+10000
                print name, id, args
                start = time.time()
                try:
                    l = FEM(f, x0, **args)
                    best, bestfit = l.learn()
                
                    used = time.time() - start
                    evals = len(res)
                    print 'result', bestfit, 'in', evals, 'evalautions, using', used, 'seconds.'
                    print 
                    
                    # storage
                    from nesexperiments import pickleDumpDict
                    pickleDumpDict('../temp/fem/'+name+'-'+str(id), {'allevals': res, 'muevals': l.muevals, 
                                                                     'args': args})
                except Exception, e:
                    print 'Ooops', e
            
if __name__ == '__main__':
    runAll(100000)
