from nesexperiments import pickleReadDict
from pybrain.rl.tasks.polebalancing.cartpoleenv import CartPoleTask
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.connections.full import FullConnection
import pylab

folder = '../temp/cartpole2/'
# 1200
iname = 'net3False-CMA--14777'

i2name = 'net3True-CMA--63465'
#100000
sname = 'net6True-CMA--28912'

task = CartPoleTask(2, markov = False)


# we want to read in a networks that solve the problem
solvingnet = pickleReadDict(folder+sname)['net']
solvingnet.name = 'solved'
intermediate = pickleReadDict(folder+iname)['net']
intermediate.name = 'intermediate'
intermediate2 = pickleReadDict(folder+i2name)['net']
intermediate2.name = 'intermediate'
randomnet = buildNetwork(task.outdim, 3, task.indim)
randomnet.addRecurrentConnection(FullConnection(randomnet['hidden0'], randomnet['hidden0'], name = 'rec'))
randomnet.sortModules()
randomnet.name = 'random'
    

nets = [solvingnet, randomnet, intermediate, intermediate2]


for net in nets:
    allobs = []
    task.reset()
    net.reset()
    while not task.isFinished():
        obs = task.getObservation()
        task.performAction(net.activate(obs))
        allobs.append(obs)  
    if len(allobs) > 2000:
        allobs = allobs[-1000:]
    res1 = map(lambda x: x[0], allobs)
    res2 = map(lambda x: x[1], allobs)
    res3 = map(lambda x: x[2], allobs)
    pylab.figure()
    pylab.title(net.name)
    pylab.plot(res1, ':')
    pylab.plot(res2, '-.')
    pylab.plot(res3, '-')
        
pylab.show()