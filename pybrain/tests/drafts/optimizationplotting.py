from pybrain.tools.plotting import FitnessPlotter
from pybrain.rl import NaturalEvolutionStrategies
import pybrain.rl.environments.functions as testfunctions


def testOptimizationPlot():
    f = testfunctions.RosenbrockFunction(2)
    E = NaturalEvolutionStrategies(f, x0 = [-2.7, 2.5], lambd = 50, returnall = True, lr = 0.005, ranking = 'smooth',
                               maxEvals = 5000)

    dummy, xs, sigmas = E.optimize()
    p4 = FitnessPlotter(f, xmin = -3, xmax = 3, ymin = -3, ymax = 3)
    #p4.addSamples(xs, color = 'c')
    p4.addCovEllipse(sigmas[0],  xs[0],  color = 'k')
    p4.addCovEllipse(sigmas[20], xs[20], color = 'k')
    p4.addCovEllipse(sigmas[40], xs[40], color = 'k')
    p4.addCovEllipse(sigmas[60], xs[60], color = 'k')
    p4.addCovEllipse(sigmas[80], xs[80], color = 'k')
    p4.plotAll(levels = 100)
                
    
if __name__ == '__main__':
    testOptimizationPlot()