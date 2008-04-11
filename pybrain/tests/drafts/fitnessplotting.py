from scipy import cov, mean, array
from pybrain.tools.plotting import FitnessPlotter
import pybrain.rl.environments.functions as testfunctions


# function
f1 = lambda x,y: 1-x**2+y**3

# data points
s = array([[0.1,0.4],[-0.5, -0.1],[-0.1, 0.2],[-0.6, 0.3],[-0.5, 0.5]])

# covariance ellipse
e = cov(s.T)
c = [mean(s[:,0]), mean(s[:,1])]


def testFitnessPlots(is3d = True):
    p1 = FitnessPlotter(f1, is3d = is3d)
    p1.addSamples(s)
    p1.addCovEllipse(e, c)
    p1.plotAll(popup = False)
    
    p2 = FitnessPlotter(testfunctions.CigarFunction, ymin = -0.005, ymax = 0.005, is3d = is3d)
    p2.plotAll(popup = False)
    
    p3 = FitnessPlotter(testfunctions.RastriginFunction, xmin = -5, xmax = 5, ymin = -5, ymax = 5, is3d = is3d)
    p3.plotAll(popup = False)
    
    p4 = FitnessPlotter(testfunctions.WeierstrassFunction, xmin = -.2, xmax = .5, ymin = -.2, ymax = .5, is3d = is3d)
    p4.plotAll(popup = False)
    
    p5 = FitnessPlotter(testfunctions.Schwefel_2_13Function, xmin = -5, xmax = 5, ymin = -5, ymax = 5, is3d = is3d)
    p5.plotAll(levels = 70)
    

if __name__ == '__main__':
    testFitnessPlots()