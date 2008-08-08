""" A little test for comptitive coevolution on the capturegame. """

__author__ = 'Tom Schaul, tom@idsia.ch'

import pylab
    
from pybrain.tools.plotting.ciaoplot import CiaoPlot    
from pybrain.structure.evolvables.cheaplycopiable import CheaplyCopiable
from pybrain.structure.networks.custom.capturegame import CaptureGameNetwork
from pybrain.rl.tasks.capturegame import CaptureGameTask, HandicapCaptureTask, RelativeCaptureTask
from pybrain.rl.agents.capturegameplayers.clientwrapper import ClientCapturePlayer
from pybrain.rl.learners.search import CompetitiveCoevolution, MultiPopulationCoevolution, Coevolution
from pybrain.rl.agents.capturegameplayers import KillingPlayer
from pybrain.tools.xml import NetworkWriter
    
tag = 'we1-'

# parameters
size = 7
generations = 400
hsize = 5

popsize = 15
selProp = 0.334
elitist = True
temperature = 0.
relTaskAvg = 1
presetGames = 0.

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
javaTest = False
handicapTest = False

# the tasks:
absoluteTask = CaptureGameTask(size, averageOverGames = 40, alternateStarting = True, 
                               opponent = KillingPlayer)
relativeTask = RelativeCaptureTask(size, useNetworks = True, maxGames = relTaskAvg,
                                   minTemperature = temperature, presetGamesProportion = presetGames,
                                   )

# the network
net = CaptureGameNetwork(size = size, hsize = hsize, simpleborders = True, 
                         #componentclass = MDLSTMLayer
                         )
net.mutationStd = mutationStd
net = CheaplyCopiable(net)

print net.name[:-5], 'has', net.paramdim, 'trainable parameters.'

    
res = []
hres = []   
jres = []

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
                 useSharedSampling = sharedSampling,
                 populationSize = popsize, 
                 numPops = populations,
                 selectionProportion = selProp,
                 hallOfFameEvaluation = hallOfFameProp,
                 absEvalProportion = absProp,
                 absEvaluator = absoluteTask,
                 verbose = True)

evals = generations * learner._stepsPerGeneration() * relTaskAvg

def buildName():
    name = tag
    name += str(learner)
    if relTaskAvg > 1:
        name += '-rA'+str(relTaskAvg)
    name += '-T'+str(temperature)
    name += '-e'+str(evals)
    name += '-tSize'+str(tournSize)
    if beta < 1:
        name += '-pc_avg'+str(beta)
    if sharedSampling:
        name += '-sharS'
    if hallOfFameProp > 0:
        name += '-HoF'+str(hallOfFameProp)
    if absProp > 0:
        name += '-absP'+str(absProp)
    name += '-mut'+str(mutationStd)
    name += net.name[18:-5]
    return name

name = buildName()

print 'Experiment:', name
print

if handicapTest:
    handicapTask = HandicapCaptureTask(size, opponent = KillingPlayer)

if javaTest:
    try:
        javaTask = CaptureGameTask(size, averageOverGames = 40, alternateStarting = True,
                                   opponent = ClientCapturePlayer)
        javaTask.opponent.randomPartMoves = 0.2
    except:
        print 'No server found.'
        javaTest = False

def storeResults():
    print ' --- Storing..',
    n = newnet.getBase()
    n.argdict['RUNRES'] = res[:]
    n.argdict['RUNRESH'] = hres[:]
    n.argdict['RUNRESJ'] = jres[:]
    ps = []
    for h in learner.hallOfFame:
        ps.append(h.params.copy())
    n.argdict['HoF_PARAMS'] = ps
    n.argdict['HoBestFitnesses'] = learner.hallOfFitnesses
    NetworkWriter.writeToFile(n, '../temp/capturegame/1/'+name+'.xml')
    print '..done. --- '
    
for g in range(generations):
    newnet = learner.learn(learner._stepsPerGeneration())
    h = learner.hallOfFame[-1]
    res.append(absoluteTask(h))
    print res[-1], '    (evals:', learner.steps, '*', relTaskAvg, ')',
    if handicapTest:
        hres.append(handicapTask(h))
        print 'Handicap: ', hres[-1]
    else:
        print
    if javaTest:
        try:
            jres.append(javaTask(h))
            print 'Java-play:', jres[-1]
        except:
            jres.append(0)
            print 'Server playing error.'

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
    if handicapTest:
        pylab.plot(hres)
    if javaTest:
        pylab.plot(jres)
    pylab.title(name)
    
if scalingtest:
    # now, let's take the result, and compare its performance on a larger game-baord
    newsize = 9
    bignew = newnet.getBase().resizedTo(newsize)
    bigold = net.getBase().resizedTo(newsize)

    newtask = CaptureGameTask(newsize, averageOverGames = 100, alternateStarting = True,
                              opponent = KillingPlayer)
    print 'Old net on medium board score:', newtask(bigold)
    print 'New net on medium board score:', newtask(bignew)

if ciao:
    p.show()
elif absplot:
    pylab.show()