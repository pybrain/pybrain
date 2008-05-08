__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de; Christian Osendorfer, osendorf@in.tum.de'

from scipy import *
from scipy.linalg import cholesky, solve, inv
from scipy.linalg.basic import norm
from pybrain.datasets import SupervisedDataSet

# for plotting
from pylab import *
from matplotlib import axes3d as a3

class GaussianProcess:

    def __init__(self, indim, start, stop, step):
        self.indim = indim
        self.testx = self._buildGrid(start, stop, step)
        self.theta = (1.0, 1.0)
        self.calculated = True
        self.pred_mean = zeros(len(self.testx))
        self.pred_cov = eye(len(self.testx))
    
    def _kernel(self, a, b):
        """ kernel function, here RBF kernel """
        return self.theta[0]*exp(-0.5*norm(a-b, 2)**2/self.theta[1])

    def _buildGrid(self, start, stop, step):
        """ returns a mgrid type of array for 'dim' dimensions """
        dimstr = 'start:stop:step, '*self.indim
        return eval('c_[map(ravel, mgrid['+dimstr+'])]').T

    def _buildCov(self, a, b):
        K = zeros((len(a), len(b)), float)
        for i in range(len(a)):
            for j in range(len(b)):
                K[i,j] = self._kernel(a[i,:], b[j,:])
        return K
            
    def trainOnDataset(self, dataset):
        """ takes a SequentialDataSet with indim input dimension and scalar target """
        assert (dataset.getDimension('input') == self.indim)
        assert (dataset.getDimension('target') == 1)
         
        self.trainx = dataset.getField('input')
        self.trainy = ravel(dataset.getField('target'))
        self.noise = array([0.00001]*len(self.trainx))
        self.calculated = False
        
    def _calculate(self):
        # build covariance matrices
        train_train = self._buildCov(self.trainx, self.trainx)
        train_test = self._buildCov(self.trainx, self.testx)
        test_train = train_test.T
        test_test = self._buildCov(self.testx, self.testx)

        # calculate predictive mean and covariance
        K = train_train + self.noise*eye(len(self.trainx))
        self.pred_mean = dot(test_train, solve(K, self.trainy, sym_pos=True))
        self.pred_cov = test_test - dot(test_train, dot(inv(K), train_test))
        self.calculated = True
    
    def draw(self):
        if not self.calculated:
            self._calculate()
        
        return self.pred_mean + random.multivariate_normal(zeros(len(self.testx)), self.pred_cov)
        
    def plotCurves(self):
        if not self.calculated:
            self._calculate()
        
        if self.indim == 1:
            figure()
            hold(True)
            # plot drawn curves (gray)
            for i in range(50):
                plot(self.testx, self.pred_mean + random.multivariate_normal(zeros(len(self.testx)), self.pred_cov), color='gray')
            # plot mean (black)
            plot(self.testx, self.pred_mean, 'k', linewidth=2)
            # plot variance (semi-transp)
            fillx = r_[ravel(self.testx), ravel(self.testx[::-1])]
            filly = r_[self.pred_mean+2*diag(self.pred_cov), self.pred_mean[::-1]-2*diag(self.pred_cov)[::-1]]
            fill(fillx, filly, facecolor='blue', edgecolor='white', alpha=0.3)
            title('1D Gaussian Process with 50 samples, mean and variance')
            
        elif self.indim == 2:
            fig = figure()
            ax = a3.Axes3D(fig)
            # plot mean
            (x, y, z) = map(lambda m: m.reshape(sqrt(len(m)), sqrt(len(m))), (self.testx[:,0], self.testx[:,1], self.pred_mean))
            ax.plot_wireframe(x, y, z, colors='gray')
        
        else: print "plotting only supported for indim=1 or indim=2."


if __name__ == '__main__':
    # --- example on how to use the GP in 1 dimension
    ds = SupervisedDataSet(1,1)
    
    x = mgrid[0:5:4j]
    y = cos(x)

    for i,j in zip(x, y):
        ds.addSample([i], [j])
    
    gp = GaussianProcess(indim=1, start=0, stop=5, step=0.1)    
    gp.trainOnDataset(ds)
    gp.plotCurves() 
        
    # --- example on how to use the GP in 2 dimensions
    ds = SupervisedDataSet(2,1)
    
    x,y = mgrid[0:5:4j, 0:5:4j]
    z = cos(x)*sin(y)
    (x, y, z) = map(ravel, [x, y, z])

    for i,j,k in zip(x, y, z):
        ds.addSample([i, j], [k])
    
    gp = GaussianProcess(indim=2, start=0, stop=5, step=0.2)    
    gp.trainOnDataset(ds)
    gp.plotCurves() 
    
    show()