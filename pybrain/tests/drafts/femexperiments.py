""" Script for (online) FEM experiments on continous benchmark functions """

import time
from random import choice
from scipy import randn, rand

from pybrain.utilities import storeCallResults
from pybrain.rl.learners import FEM
from pybrain.rl.environments.functions import SphereFunction, RosenbrockFunction, CigarFunction, RotateFunction, TranslateFunction


basefunctions = [SphereFunction, RosenbrockFunction, CigarFunction]
dims = [2, 5, 15]

budgets = [500, 1000, 2000]
batchsizes = [50, 100]
learningRates = [0.1, 0.5]
shapings = ['top', 'toplinear', 'smooth']
ginis = [0.2, 0.02]
explorations = [1., 2.]
        

def runAll(repeat = 1):
    for dummy in range(repeat):
        for dim in dims:
            for basef in basefunctions:
                f = TranslateFunction(RotateFunction(basef(dim)))
                x0 = randn(dim)
                name = basef.__name__+str(dim)
                f.desiredValue = 1e-6
    
                res = storeCallResults(f)
                
                args = {'onlineLearning': True,
                        'batchsize': choice(batchsizes),
                        'ranking': choice(shapings),
                        'maxupdate': choice(learningRates),
                        'maxEvaluations': dim * choice(budgets),
                        'elitist': choice([True, False]),
                        'unlawfulExploration': choice(explorations),
                        'verbose': False,
                        }
                if args['ranking'] == 'smooth':
                    args['gini'] = choice(ginis)
                    args['giniPlusX'] = 0.15
                else:
                    args['topselection'] = choice([2, 0.1*args['batchsize'], 0.333*args['batchsize']])
                
                id = int(rand(1)*90000)+10000
                print name, id
                print ' '*20, args
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
                                                                 'best': best, 'args': args})
                except Exception, e:
                    print 'Ooops', e
            
if __name__ == '__main__':
    runAll(100000)
