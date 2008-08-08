""" A little test for comptitive coevolution on gomoku. """

__author__ = 'Tom Schaul, tom@idsia.ch'

import pylab
    
from pybrain.tools.plotting.ciaoplot import CiaoPlot    
from pybrain.structure.evolvables.cheaplycopiable import CheaplyCopiable
from pybrain.structure.networks.custom.capturegame import CaptureGameNetwork
from pybrain.rl.tasks.gomoku import GomokuTask, RelativeGomokuTask
from pybrain.rl.agents.gomokuplayers import KillingGomokuPlayer
from pybrain.rl.learners.search import CompetitiveCoevolution, MultiPopulationCoevolution, Coevolution
from pybrain.tools.xml import NetworkWriter
    
tag = 'we2-'

# parameters
size = 5
generations = 400
hsize = 10

popsize = 15
selProp = 0.334
elitist = True
temperature = 0.
relTaskAvg = 1

tournSize = 15
hallOfFameProp = 0.667
sharedSampling = True
beta = 1
absProp = 0.
mutationStd = 0.05

competitive = True
multipop = True
populations = 2

# experiment settings
ciao = False
absplot = True
scalingtest = False
storage = True

# the tasks:
absoluteTask = GomokuTask(size, averageOverGames = 100, alternateStarting = True, 
                          opponent = KillingGomokuPlayer, numMovesCoeff = 0.2)
relativeTask = RelativeGomokuTask(size, useNetworks = True, maxGames = relTaskAvg,
                                  minTemperature = temperature)

# the network
net = CaptureGameNetwork(size = size, hsize = hsize, simpleborders = True, 
                         #componentclass = MDLSTMLayer
                         )
net._params /= 10
net.mutationStd = mutationStd
net = CheaplyCopiable(net)

print net.name[:-5], 'has', net.paramdim, 'trainable parameters.'
    
res = []

if competitive: 
    lclass = CompetitiveCoevolution    
elif multipop:
    lclass = MultiPopulationCoevolution
else:
    lclass = Coevolution

seeds = []
for dummy in range(popsize):
    tmp = net.copy()
    tmp.randomize()
    tmp._params /= 10 # start with small values.
    seeds.append(tmp)

learner = lclass(relativeTask, 
                 seeds, 
                 elitism = elitist, 
                 parentChildAverage = beta,
                 tournamentSize = tournSize,
                 populationSize = popsize, 
                 numPops = populations,
                 selectionProportion = selProp,
                 hallOfFameEvaluation = hallOfFameProp,
                 absEvalProportion = absProp,
                 absEvaluator = absoluteTask,
                 verbose = True)

evals = generations * learner._stepsPerGeneration() * relTaskAvg

def buildName():
    name = tag+'Gomoku-'
    name += str(learner)
    #if competitive:
    #    name += 'Comp'
    #name += 'Coev'
    if relTaskAvg > 1:
        name += '-rA'+str(relTaskAvg)
    #if elitist:
    #    name += '-elit'
    name += '-T'+str(temperature)
    name += '-e'+str(evals)
    #name += '-pop'+str(popsize)
    #if tournSize != None:
    name += '-tSize'+str(tournSize)
    if beta < 1:
        name += '-pc_avg'+str(beta)
    if hallOfFameProp > 0:
        name += '-HoF'+str(hallOfFameProp)
    #if selProp != 0.5:
    #    name += '-selP'+str(selProp)
    if absProp > 0:
        name += '-absP'+str(absProp)
    #if mutationStd != 0.1:
    name += '-mut'+str(mutationStd)
    name += net.name[18:-5]
    return name

name = buildName()

print 'Experiment:', name
print

def storeResults():
    print ' --- Storing..',
    n = newnet.getBase()
    n.argdict['RUNRES'] = res[:]
    ps = []
    for h in learner.hallOfFame:
        ps.append(h.params.copy())
    n.argdict['HoF_PARAMS'] = ps
    n.argdict['HoBestFitnesses'] = learner.hallOfFitnesses
    NetworkWriter.writeToFile(n, '../temp/capturegame/2/'+name+'.xml')
    print '..done. --- '
    
for g in range(generations):
    newnet = learner.learn(learner._stepsPerGeneration())
    h = learner.hallOfFame[-1]
    res.append(absoluteTask(h))
    print res[-1], '    (evals:', learner.steps, '*', relTaskAvg, ')',
    print
    if g % 5 == 0 and g > 0 and storage and evals > 100:
        storeResults()        
    print
        
# store result
if storage and evals > 100:
    storeResults()

# plot CIAO diagram
if ciao:    
    hof = learner.hallOfFame
    if competitive and generations % 2 == 0:
        hof1 = hof[0::2]
        hof2 = hof[1::2]
    else:
        hof1 = hof
        hof2 = hof
    p = CiaoPlot(lambda x,y: relativeTask(x, y, 0), hof1, hof2)
    pylab.title('CIAO'+name)
    
if absplot:
    # plot the progression
    pylab.figure()
    if multipop:
        for i in range(populations):
            pylab.plot(res[i::populations])
    else:
        pylab.plot(res)
    pylab.title(name)
    
if scalingtest:
    # now, let's take the result, and compare its performance on a larger game-baord
    newsize = 11
    bignew = newnet.getBase().resizedTo(newsize)
    bigold = net.getBase().resizedTo(newsize)

    newtask = GomokuTask(newsize, averageOverGames = 100, alternateStarting = True,
                              opponent = KillingGomokuPlayer)
    print 'Old net on medium board score:', newtask(bigold)
    print 'New net on medium board score:', newtask(bignew)

if ciao:
    p.show()
elif absplot:
    pylab.show()