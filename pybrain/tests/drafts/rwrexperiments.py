__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import randn, mean, var, sqrt, array
from random import random, choice

from pybrain.utilities import Named
from pybrain.rl.learners import RWR
from pybrain.rl.tasks.pomdp import TrivialMaze, CheeseMaze, TMaze, ShuttleDocking, FourByThreeMaze, TigerTask

from nesexperiments import pickleDumpDict, pickleReadDict


def buildNet(inn, hidden, outn):
    """ lstm network with direct connections to the output layer (which is a softmax). """
    from pybrain import FullConnection, SoftmaxLayer, LSTMLayer, buildNetwork
    
    N = buildNetwork(inn, hidden, outn, 
                     hiddenclass = LSTMLayer, outclass = SoftmaxLayer)
    N.addConnection(FullConnection(N['in'], N['out']))
    N.sortModules()    
    return N



class ExperimentalData:
    """ stores experimental Data on a set of experiments """
    def __init__(self, filename = 'default'):
        # not-overwrite security
        self.now = 1
        self.filename = filename
        self.expids = {}
        self.fullResults = {}        
        
    def registerExperiment(self, expid):
        self.expids[expid] = True
        
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



class RwrExperiment(Named):
    """ run rwr on a task with a set of settings. 
    Save the results in a file. """
    
    task = None
    maxBatches = 200
    repeat = 1
    useValueFunction = False
    
    initialWeightRange = 0.01
    hidden = 2
    
    learningRate = 0.005
    momentum = 0.9
    validationProportion = 0.33
    maxEpochs = 20
    continueEpochs = 2
    
    maxSteps = 100
    batchSize = 100
    greedyRuns = None
    
    folder = '../temp/rwr/'
    
    def __init__(self, **args):
        if 'useValueFunction' in args and args['useValueFunction'] == True:
            self.valueLearningRate = 0.00001
            self.valueMomentum = 0.9
            #self.valueTrainEpochs = 5
            self.resetAllWeights = False
        self.setArgs(**args)
        if self.greedyRuns == None:
            self.greedyRuns = self.batchSize/2
        self.task.maxSteps = self.maxSteps
        self.net = buildNet(self.task.outdim, self.hidden, self.task.indim)
        self.settings = {'learningRate': self.learningRate, 
                         'momentum': self.momentum, 
                         'maxEpochs': self.maxEpochs,
                         'greedyRuns': self.greedyRuns, 
                         'batchSize': self.batchSize,
                         'validationProportion': self.validationProportion,
                         'continueEpochs': self.continueEpochs,
                         }
        if self.useValueFunction:
            self.vnet = buildNet(self.task.outdim, self.hidden, 1)
            self.settings['valueLearningRate'] = self.valueLearningRate
            self.settings['valueMomentum'] = self.valueMomentum
            #self.settings['valueTrainEpochs'] = self.valueTrainEpochs
            self.settings['resetAllWeights'] = self.resetAllWeights            
        
        # prepare storage
        self.id =  self.task.__class__.__name__
        if isinstance(self.task, TMaze):
            self.id += str(self.task.length)
        self.id += '-'+str(int(random()*9000)+1000)
        self.store = ExperimentalData(self.folder+self.id)
        self.store.registerExperiment(self.id)
        
        
    def initWeights(self):
        self.net.params[:] = self.initialWeightRange * randn(self.net.paramdim)
        if self.useValueFunction:
            self.vnet.params[:] = self.initialWeightRange * randn(self.vnet.paramdim)
        
    def run(self):
        print self.task
        print self.settings
        for r in range(self.repeat):
            print
            print '+'*60
            print 'Experiment', r
            self.initWeights()
            try:
                self.rwr = RWR(self.net, self.task, **self.settings)
                self.rwr.learn(self.maxBatches)
                self.save()
            except:
                print 'Something went wrong.'
        
    def save(self):
        """ Save the following information:
        - settings
        - final network
        - total steps
        - for every batch:
            -- greedy avg reward
            -- avg reward
            -- avg length
            -- avg initial discounted return
        """
        info = {}
        info['params'] = self.settings
        info['netWeights'] = self.net.params.copy()
        info['totalSteps'] = self.rwr.totalSteps
        info['totalEpisodes'] = self.rwr.totalEpisodes
        info['greedyAvg'] = tuple(self.rwr.greedyAvg)
        info['lengthAvg'] = tuple(self.rwr.lengthAvg)
        info['rewardAvg'] = tuple(self.rwr.rewardAvg)
        info['initr0Avg'] = tuple(self.rwr.initr0Avg)
        self.store.addRunData(self.id, info)
    
    
    
    
    

commonSettings = {'repeat': 25, 'batchSize': 60, 'learningRate': 0.002, 'hidden': 2,
                  'momentum':0.95, 'validationProportion': 0.33, 'maxSteps': 100,
                  'maxEpochs': 10, 'continueEpochs': 1, 'maxBatches': 100,
                  }

specificSettings = {TrivialMaze(): {'batchSize': 15, 'hidden': 1, 'maxBatches': 20,
                                    },
                    CheeseMaze(): {'batchSize': 75, 'maxSteps': 500,
                                   },
                    TigerTask(): {},
                    ShuttleDocking(): {},
                    FourByThreeMaze(): {},
                    TMaze(length = 3): {'batchSize': 75, 'greedyRuns': 8,
                                        },
                    TMaze(length = 5): {'batchSize': 75,'greedyRuns': 8,
                                        },
                    TMaze(length = 7): {'batchSize': 75, 'maxSteps': 200, 'greedyRuns': 8,
                                        },
                    TMaze(length = 10): {'batchSize': 150, 'maxSteps': 200, 'greedyRuns': 16,
                                         },
                    }

tasks = sorted(specificSettings.keys())

    
def runSingleExperiment(id):
    task = tasks[id]
    args = commonSettings.copy()
    for k, v in specificSettings[task].items():
        args[k] = v
    E = RwrExperiment(task = task, **args)
    E.run()
    
    
def runAllExperiments():
    for id in range(len(tasks)):
        runSingleExperiment(id)

    
def randomPolicyAvgReward(t, avgover = 20000):
    print t.name
    res = []
    for dummy in range(avgover):
        t.reset()
        r = 0
        s = 0
        while not t.isFinished():
            t.performAction(choice(range(t.actions)))
            r += t.getReward()
            s += 1
        res.append(r/float(s))
    res = array(res)
    print mean(res), '+-', sqrt(var(res))
    
    
if __name__ == '__main__':
    # 0 Trivial
    # 1 Cheese
    # 2 Tiger
    # 3 Shuttle
    # 4 4x3
    # 5 T3
    # 6 T5
    # 7 T7
    # 8 T10
    #runSingleExperiment(0)
    for t in tasks:
        randomPolicyAvgReward(t)
