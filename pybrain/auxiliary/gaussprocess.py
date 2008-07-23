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
        self.mean = 0
        self.start = start
        self.stop = stop
        self.step = step
        self.indim = indim
        self.trainx = zeros((0, indim), float)
        self.trainy = zeros((0), float)
        self.noise = zeros((0), float)
        self.testx = self._buildGrid(start, stop, step)
        self.calculated = True
        self.pred_mean = zeros(len(self.testx))
        self.pred_cov = eye(len(self.testx))
        self.autonoise = False
        self.hyper = (0.5, 2.0, 0.01)
    
    def _kernel(self, a, b):
        """ kernel function, here RBF kernel """
        (l, sigma_f, sigma_n) = self.hyper
        # CHECKME: multiply with I at the end? should return scalar though...
        return sigma_f**2*exp(-1.0/(2*l**2)*norm(a-b, 2)**2 + sigma_n) #*eye(self.indim))

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
    
    def reset(self):
        self.trainx = zeros((0, self.indim), float)
        self.trainy = zeros((0), float)
        self.noise = zeros((0), float)
        self.pred_mean = zeros(len(self.testx))
        self.pred_cov = eye(len(self.testx))
              
    def trainOnDataset(self, dataset):
        """ takes a SequentialDataSet with indim input dimension and scalar target """
        assert (dataset.getDimension('input') == self.indim)
        assert (dataset.getDimension('target') == 1)
         
        self.trainx = dataset.getField('input')
        self.trainy = ravel(dataset.getField('target'))
        self.noise = array([0.001]*len(self.trainx))
        # print self.trainx, self.trainy
        self.calculated = False
        
    def addDataset(self, dataset):
        """ adds the points from the dataset to the training set """
        assert (dataset.getDimension('input') == self.indim)
        assert (dataset.getDimension('target') == 1)
        
        self.trainx = r_[self.trainx, dataset.getField('input')]
        self.trainy = r_[self.trainy, ravel(dataset.getField('target'))]
        self.noise = array([0.001]*len(self.trainx))
        self.calculated = False
        
    def addSample(self, train, target):
        self.trainx = r_[self.trainx, asarray([train])]
        self.trainy = r_[self.trainy, asarray(target)]
        self.noise = r_[self.noise, array([0.001])]
        self.calculated = False
        
    def _calculate(self):
        # build covariance matrices
        train_train = self._buildCov(self.trainx, self.trainx)
        train_test = self._buildCov(self.trainx, self.testx)
        test_train = train_test.T
        test_test = self._buildCov(self.testx, self.testx)

        # calculate predictive mean and covariance
        K = train_train + self.noise*eye(len(self.trainx))
        
        if self.autonoise:
            # calculate average neighboring distance for auto-noise
            avgdist = 0
            sort_trainx = sort(self.trainx)
            for i,d in enumerate(sort_trainx):
                if i == 0:
                    continue
                avgdist += d - sort_trainx[i-1]
            avgdist /= len(sort_trainx)-1
            print avgdist
            # sort(self.trainx)
        
            # add auto-noise from neighbouring samples (not standard gp)  
            for i in range(len(self.trainx)):
                for j in range(len(self.trainx)):
                    if norm(self.trainx[i] - self.trainx[j]) > avgdist:
                        continue
                
                    d = norm(self.trainy[i] - self.trainy[j]) / (exp(norm(self.trainx[i] - self.trainx[j])))
                    K[i,i] += d
        
        self.pred_mean = self.mean+dot(test_train, solve(K, self.trainy-self.mean, sym_pos=False))
        self.pred_cov = test_test - dot(test_train, dot(inv(K), train_test))
        self.calculated = True
    
    def draw(self):
        if not self.calculated:
            self._calculate()
        
        return self.pred_mean + random.multivariate_normal(zeros(len(self.testx)), self.pred_cov)
    
    def plotCurves(self, showSamples=False):
        if not self.calculated:
            self._calculate()
        
        if self.indim == 1:
            clf()
            hold(True)
            if showSamples:
                # plot samples (gray)
                for i in range(5):
                    plot(self.testx, self.pred_mean + random.multivariate_normal(zeros(len(self.testx)), self.pred_cov), color='gray')
            
            # plot training set
            plot(self.trainx, self.trainy, 'bx')
            # plot mean (black)
            plot(self.testx, self.pred_mean, 'b', linewidth=1)
            # plot variance (semi-transp)
            fillx = r_[ravel(self.testx), ravel(self.testx[::-1])]
            filly = r_[self.pred_mean+2*diag(self.pred_cov), self.pred_mean[::-1]-2*diag(self.pred_cov)[::-1]]
            fill(fillx, filly, facecolor='gray', edgecolor='white', alpha=0.3)
            title('1D Gaussian Process with mean and variance')
            
        elif self.indim == 2:
            fig = figure()
            ax = a3.Axes3D(fig)
            
            # plot training set
            ax.plot3D(ravel(self.trainx[:,0]), ravel(self.trainx[:,1]), ravel(self.trainy), 'ro')
            
            # plot mean
            (x, y, z) = map(lambda m: m.reshape(sqrt(len(m)), sqrt(len(m))), (self.testx[:,0], self.testx[:,1], self.pred_mean))
            ax.plot_wireframe(x, y, z, colors='gray')
        
        else: print "plotting only supported for indim=1 or indim=2."
    

if __name__ == '__main__':
    # --- example on how to use the GP in 1 dimension
    ds = SupervisedDataSet(1, 1)
    gp = GaussianProcess(indim=1, start=-3, stop=3, step=0.05)    

    x = mgrid[-3:3:0.2]
    y = 0.1*x**2 + x + 1
    z = sin(x) + 0.5*cos(y)
    
    ds.addSample(-2.5, -1)
    ds.addSample(-1.0, 3)
    gp.mean = 0
    
    # new feature "autonoise" adds uncertainty to data depending on
    # it's distance to other points in the dataset. not tested much yet.
    # gp.autonoise = True
    
    gp.trainOnDataset(ds) 
    gp.plotCurves(showSamples=True) 
     
     
            
    # --- example on how to use the GP in 2 dimensions
    
    ds = SupervisedDataSet(2,1)
    gp = GaussianProcess(indim=2, start=0, stop=5, step=0.2)    
    
    x,y = mgrid[0:5:4j, 0:5:4j]
    z = cos(x)*sin(y)
    (x, y, z) = map(ravel, [x, y, z])
    
    for i,j,k in zip(x, y, z):
        ds.addSample([i, j], [k])
    
    gp.trainOnDataset(ds)
    gp.plotCurves() 

    show()