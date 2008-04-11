"""

Setup for a number of experiments of using NES. 
All parameters are predefined.

"""
from pybrain.rl.environments.functions.unbounded import UnboundedFunctionEnvironment

import time
from random import random
from scipy import array

import pybrain.rl.environments.functions as testfunctions
from pybrain.rl.environments.functions import OppositeFunction, TranslateFunction, RotateFunction
from pybrain.rl.learners.blackboxoptimizers import CMAES, NaturalEvolutionStrategies


__author__ = 'Tom Schaul, tom@idsia.ch'


def pickleDumpDict(name, d):
    """ pickle-dump a variable into a file """
    import pickle
    try:        
        f = open(name+'.pickle', 'w')
        pickle.dump(d, f)
        f.close()        
        return True
    except Exception, e:
        print 'Error writing into', name, ':', str(e)
        return False

    
def pickleReadDict(name):
    """ pickle-read a (defult: dictionnary) variable from a file """
    import pickle
    try:
        f = open(name+'.pickle')
        val = pickle.load(f)
        f.close()        
    except Exception, e:
        print 'Nothing read from', name,':', str(e)
        val = {}
    return val


class ExperimentalData:
    """ stores experimental Data on a set of experiments """
    def __init__(self, filename = 'default'):
        # not-overwrite security
        self.now = 1
        self.filename = filename
        self.expids = {}
        self.fullResults = {}        
        
    def registerExperiment(self, fname, dimension, expid, cmaid):
        self.expids[expid] = {'fname': fname,
                              'dim': dimension,
                              'cma': False,
                              }
        self.expids[cmaid] = {'fname': fname,
                              'dim': dimension,
                              'cma': True,
                              }
        
    def addRunData(self, expid, data):
        if expid not in self.fullResults:
            self.fullResults[expid] = []
        self.fullResults[expid].append(data)
        self.save(self.filename)

    def save(self, filename):
        """ append the experimental results into a file """
        # TODO: maybe do all this in XML for readablity...?
        self.now = -self.now
        if self.now > 0:
            filename = filename+'-b'
        pickleDumpDict(filename, self.fullResults)
        pickleDumpDict(filename+'i', self.expids)
    
    @staticmethod
    def load(filename):
        """ create an object from the specified file """
        res = ExperimentalData()
        res.filename = filename
        res.fullResults = pickleReadDict(filename)
        res.expids = pickleReadDict(filename+'i')
        return res
    
            


class NESExperiment:
    """"""
    
    folder = '../temp/nes/'
    repeat = 10
    cma = True
    
    def __init__(self, f, dim, **params):
        # create function
        if 'distance' in params:
            self.f = RotateFunction(TranslateFunction(f(dim), params['distance']))
            del params['distance']
        else:
            self.f = RotateFunction(TranslateFunction(f(dim)))
            

        # identification
        self.id = f.__name__+str(self.f.xdim)+'-'
        if 'id' in params:
            self.id += params['id']
            del params['id']
        else:
            self.id += str(int(random()*1000))
        self.cmaid = 'cma'+self.id
        
        # prepare storage
        self.store = ExperimentalData(self.folder+self.id)
        
        # register for storage
        self.store.registerExperiment(self.f.__class__.__name__, self.f.xdim, self.id, self.cmaid)
        
        # prepare parameters
        if 'repeat' in params: 
            self.repeat = params['repeat']
            del params['repeat']
        if 'cma' in params: 
            self.cma = params['cma']
            del params['cma']
        self.params = params
        
        # cma parameters
        self.cmaparams = {}
        if 'maxEvals' in self.params:
            self.cmaparams['maxEvals'] = self.params['maxEvals']
        if 'stopPrecision' in self.params:
            self.cmaparams['stopPrecision'] = self.params['stopPrecision']
            self.params['stopPrecision'] = -self.params['stopPrecision']
        if 'x0' in self.params:
            self.cmaparams['x0'] = self.params['x0']
            
        
        print 'Now running:', self.id
        
        
    def run(self):
        """ run the experiment, repeatedly, and on CMA """
        done = 0
        successes = 0
        cmasuccesses = 0
        
        while done < self.repeat:
            try:
                if self.cma:
                    info = {}
                    self.f.reset()
                    c = CMAES(self.f, **self.cmaparams)
                    starttime = time.time()
                    info['res'] = c.optimize()
                    info['params'] = self.cmaparams
                    info['offset'] = self.f.xopt
                    info['rotation'] = self.f.M
                    info['time'] = time.time() - starttime
                    evaluations = c.tfun.vallist
                    info['success'] = min(evaluations) < c.stopPrecision
                    if info['success']:
                        cmasuccesses += 1
                    info['best'] = min(evaluations)
                    info['nbEvals'] = len(evaluations)    
                    # store only every 10th point
                    info['evalsPerGen'] = 10
                    tmp = []
                    for i in range(len(evaluations)):
                        if i%10 == 0:
                            tmp.append(evaluations[i])
                    info['xVals'] = tmp            
                    info['gens'] = len(tmp)    
                    print 'CMA used:', info['nbEvals'], 'best fitness:', info['best'], self.cmaparams
                    self.store.addRunData(self.cmaid, info)
                
                info = {}
                self.f.reset()
                e = NaturalEvolutionStrategies(OppositeFunction(self.f), **self.params)
                starttime = time.time()
                info['res'] = e.optimize()
                info['params'] = self.params
                info['offset'] = self.f.xopt
                info['rotation'] = self.f.M
                info['time'] = time.time() - starttime
                evaluations = map(lambda x:-x, e.tfun.vallist)
                info['success'] = min(evaluations) <= -e.stopPrecision
                if info['success']:
                   successes += 1
                info['best'] = min(evaluations)
                info['nbEvals'] = len(evaluations)   
                gens = len(evaluations)/(e.lambd+e.mu)
                info['gens'] = gens
                info['evalsPerGen'] = (e.lambd+e.mu)
                tmp = array(evaluations).reshape(gens, e.lambd+e.mu)
                info['xVals'] = tmp[:, 0:e.mu] 
                print 'NES used:', info['nbEvals'], 'best fitness:', info['best'], self.params
                self.store.addRunData(self.id, info)
                
                
                done += 1
            except Exception, e:
                print 'Something went wrong...',
                print e
        return successes, cmasuccesses
                
def fullRun():
    """ run all variations of experiments, and repeat them. """
    for dim in dimensions:
        for f in reversed(allfunctions):
            params = commonParameters.copy()
            if dim in variableParameters:
                for p, val in variableParameters[dim].items():
                    params[p] = val            
            e = NESExperiment(f, dim, **params)
            e.run()




commonParameters = {#'mu' : 1,
                    'stopPrecision' : 1e-6, 
                    'gini' : 0.05,
                    'ranking' : 'smooth', 
                    'slidingbatch' : False, 
                    #'importanceSampling' : False,
                    'repeat': 10,
                    'silent': True,
                    'returnall': True,
                    }

# default parameters dependent on dimension
variableParameters = {2: {'lambd': 20, 'lr' : 0.002, 'maxEvals' : 5e3},
                      5: {'lambd': 100, 'lr' : 0.0005, 'maxEvals' : 2e4},
                      10: {'lambd': 200, 'lr' : 0.00005, 'maxEvals' : 5e4},
                      15: {'lambd': 450, 'lr' : 0.00001, 'maxEvals' : 1e6},
                      }


mainfunctions = [# the main ones
                testfunctions.SphereFunction,
                testfunctions.SchwefelFunction,
                testfunctions.RosenbrockFunction,
                testfunctions.CigarFunction,
                testfunctions.TabletFunction,
                ]
otherfunctions = [# less important ones
                  testfunctions.ElliFunction,
                  testfunctions.DiffPowFunction,
                  testfunctions.SharpRFunction,
                  testfunctions.ParabRFunction,
                  ]
multimodalfunctions = [# more multi-modal ones
                       testfunctions.RastriginFunction,
                       testfunctions.AckleyFunction,
                       testfunctions.WeierstrassFunction,
                       #testfunctions.Schwefel_2_13Function,
                       testfunctions.GriewankFunction,
                       ]

allfunctions = mainfunctions + otherfunctions + multimodalfunctions

# to be tested
dimensions = [15]          


# successful runs in the format: (dimension, function, non-default parameters), # comment
goodSettings = [(2, testfunctions.SphereFunction, {'lr': 0.05, 'lambd': 15, 'repeat': 5}),
                (5, testfunctions.SchwefelFunction, {'lr': 0.02, 'lambd': 45, 'repeat': 6, 'maxEvals': 3e3}),
                (2, testfunctions.RastriginFunction, {'lr': 0.02, 'lambd': 20, 'repeat': 20, 'maxEvals': 2e3, 
                                                      'stopPrecision': 0.01, 'distance': 2, 'initSigmaCoeff' : 1}),
                (15, testfunctions.SchwefelFunction, {'lr': 0.01, 'lambd': 250, 'repeat': 2, 'silent': False, 
                                                      'maxEvals': 2e4}),
                (30, testfunctions.SchwefelFunction, {'lr': 0.002, 'lambd': 1000, 'repeat': 1, 
                                                      'cma': False, 'silent': False, 'maxEvals': 1e5}),
                (2, testfunctions.SharpRFunction, {'lr': 0.05, 'lambd': 15, 'repeat': 1,
                                                   'silent' : False, 'stopPrecision': -10000}),
                (15, testfunctions.RosenbrockFunction, {'lr': 0.002, 'lambd': 250, 'repeat': 1, 'silent': False, 
                                                      'maxEvals': 4e4, 'cma': False}),
                
                ]
                
                
excellentSettings = [(2, testfunctions.SchwefelFunction, {'lr': 0.05, 'lambd': 15, 'repeat': 5}),
                     ]


def runWithSetting(setting):
    dim, f, inParams = setting
    params = commonParameters.copy()
    if dim in variableParameters:
        for p, val in variableParameters[dim].items():
            params[p] = val   
    for p, val in inParams.items():
        params[p] = val            
    params['id'] = 'simple'
    e = NESExperiment(f, dim, **params)
    e.run()


def singleSmallRun():
    """ run one experiment, with a low dimension, low evaluation limtation, and repeat 
    only twice. Randomly pick the function. """
    params = commonParameters.copy()
    
    
    # decide on the settings for the experiment:
    #f = testfunctions.SphereFunction
    f = testfunctions.WeierstrassFunction
    
    dim = 15
    for p, val in variableParameters[dim].items():
        params[p] = val
    
    params['repeat'] = 5
    params['lr'] = 0.01
    params['id'] = 'simple'
    params['maxEvals'] = 2e4
    params['cma'] = True
    params['lambd'] = 245
    params['initSigmaCoeff'] = .5
    params['silent'] = False
    #x0dist = 0.1
    #x0 = rand(dim, 1)-0.5
    #params['x0'] = x0/norm(x0.flatten())*x0dist
    
    e = NESExperiment(f, dim, **params)
    e.run()


# parameters for the publication-runs
unimodalParameters = [{'stopPrecision' : 1e-6, 
                       'gini' : 0.05,
                        'ranking' : 'smooth',                         
                        'repeat': 1,
                        'silent': False,
                        #'ridge': False,
                        'cma': False,
                        'initSigmaCoeff' : 0.3,
                        'initialsearchrange': 1.0,
                        },
                        
                        {2 : {'lambd':20,  'lr':0.05},
                         5:  {'lambd':50,  'lr':0.02, 'maxEvals': 1e4},
                         15: {'lambd':int((15+15*15)*2.5), 'lr':0.002, 'maxEvals': 1e6},
                         }]

# parameters for the publication-runs
multimodalParameters = {'stopPrecision' : 1e-2, 
                        'gini' : 0.05,
                        'ranking' : 'smooth',                         
                        'repeat': 100,
                        'silent': True,
                        'lambd':20,  
                        'lr':0.05,
                        'initSigmaCoeff' : 2,
                        'maxEvals': 5e3,
                        }

multimodalDistances = [1,3,10,30,100]


def multiModalRun():
    for d in multimodalDistances:
        print 'Distance', d
        print
        for f in multimodalfunctions:
            dim = 2
            params = multimodalParameters.copy()
            params['distance'] = d
            params['id'] = 'multi-d'+str(d)
            e = NESExperiment(f, dim, **params)
            print e.run()
            print
        
def uniModalRun():
    for dim in dimensions:
        for f in [testfunctions.RosenbrockFunction, testfunctions.TabletFunction]:
            params = unimodalParameters[0].copy()
            if dim in unimodalParameters[1]:
                for p, val in unimodalParameters[1][dim].items():
                    params[p] = val
            if isinstance(f, UnboundedFunctionEnvironment):
                params['stopPrecision'] = -1000
            params['id'] = 'uni'
            e = NESExperiment(f, dim, **params)
            e.run()
            print 
            
        
    

if __name__ == '__main__':
    #fullRun()
    #singleSmallRun()
    #runWithSetting(goodSettings[-1])
    #multiModalRun()
    uniModalRun()