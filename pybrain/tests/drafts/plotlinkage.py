__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import zeros, cos, sin

from pybrain.rl.environments.functions import SphereFunction, OppositeFunction, FunctionEnvironment, RosenbrockFunction
from pybrain.rl.learners import LiMaG
from pybrain.tools.plotting import ColorMap, FitnessPlotter


class LinkedFunction(FunctionEnvironment):
    xdimMin = 2
    
    def f(self, x):
        #return x[0]**2 + x[1]**2 - 100 * x[0] * x[1] + sum(x[2:])
        #return (-2 * x[0]**2 - x[1] + 50 + 2 * x[1]**2 * x[0])**2 + 0.1 * sum((x[2:]-1)**2)
        return 100 * (cos(x[0]*x[1]**2*1e5) + 1) + 10* (sin(x[3]*x[2]**3*1e2) + 1) + sum((x[4:]-1)**2)
        #return 10* (int(abs(x[0]*x[1]*100) % int(abs(x[1]+20)))) + sum((x[2:]-1)**2)


def testLinkagePlotting():
    
    n = 4
    mini = False
    if mini:
        pop = 8
        batch = 1
        lr = 0.01
        verbose = True
        topprop = 0.5
        maxevals = 50
    elif False:
        pop = 300
        batch = 30
        lr = 0.01
        verbose = False
        topprop = 0.2
        maxevals = 1500
    else:
        pop = 4
        batch = 200
        lr = 0.5
        verbose = False
        topprop = 0.5
        maxevals = 20
    
    #f = OppositeFunction(SphereFunction(n, xopt = [1]*n))
    f = OppositeFunction(LinkedFunction(n))
    
    # a value that cannot be reached
    f.desiredValue = 0
    res = zeros((n,n))
    
    for dummy in range(batch):
        f.reset()
        E = LiMaG(f, learningRate = lr, useMatrixForCrossover = False, multiParents = False,
                  mutationStdDev = 0.0, topproportion = topprop, verbose = verbose,
                  popsize = pop, maxEvals = maxevals, 
                  fitnessSmoothing = False)
        print E.optimize()   
        print len(f.xlist)
        res += E.lm
        
    res /= batch
    # now plot it's linkage matrix as a colormap
    c = ColorMap(res, minvalue = 0, maxvalue = 1, pixelspervalue = 600/n)
    #c.save('../temp/linkageplot.png')
    c.show()
    
def plotFunction():
    p = FitnessPlotter(LinkedFunction, is3d = True)
    p.plotAll()
    
    
if __name__ == '__main__':
    #plotFunction()
    testLinkagePlotting()
    
