from __future__ import print_function

__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de; Christian Osendorfer, osendorf@in.tum.de'


from scipy import r_, exp, zeros, eye, array, asarray, random, ravel, diag, sqrt, sin, cos, sort, mgrid, dot, floor
from scipy import c_ #@UnusedImport
from scipy.linalg import solve, inv
from pybrain.datasets import SupervisedDataSet
from scipy.linalg import norm


class GaussianProcess:
    """ This class represents a basic n-dimensional Gaussian Process. The implementation
        follows the book 'Gaussian Processes for Machine Learning' by Carl E. Rasmussen
        (an online version is available at: http://www.gaussianprocess.org/gpml/chapters/).
        The hyper parameters of the GP can be adjusted by setting the self.hyper varible,
        which must be a tuple of size 3.
    """

    def __init__(self, indim, start=0, stop=1, step=0.1):
        """ initializes the gaussian process object.

            :arg indim: input dimension
            :key start: start of interval for sampling the GP.
            :key stop: stop of interval for sampling the GP.
            :key step: stepsize for sampling interval.
            :note: start, stop, step can either be scalars or tuples of size 'indim'.
        """
        self.mean = 0
        self.start = start
        self.stop = stop
        self.step = step
        self.indim = indim
        self.trainx = zeros((0, indim), float)
        self.trainy = zeros((0), float)
        self.noise = zeros((0), float)
        self.testx = self._buildGrid()
        self.calculated = True
        self.pred_mean = zeros(len(self.testx))
        self.pred_cov = eye(len(self.testx))
        self.autonoise = False
        self.hyper = (0.5, 2.0, 0.1)

    def _kernel(self, a, b):
        """ kernel function, here RBF kernel """
        (l, sigma_f, _sigma_n) = self.hyper
        r = sigma_f ** 2 * exp(-1.0 / (2 * l ** 2) * norm(a - b, 2) ** 2)
        # if a == b:
        #   r += sigma_n**2
        return r

    def _buildGrid(self):
        (start, stop, step) = (self.start, self.stop, self.step)
        """ returns a mgrid type of array for 'dim' dimensions """
        if isinstance(start, (int, float, complex)):
            dimstr = 'start:stop:step, '*self.indim
        else:
            assert len(start) == len(stop) == len(step)
            dimstr = ["start[%i]:stop[%i]:step[%i], " % (i, i, i) for i in range(len(start))]
            dimstr = ''.join(dimstr)
        return eval('c_[map(ravel, mgrid[' + dimstr + '])]').T

    def _buildCov(self, a, b):
        K = zeros((len(a), len(b)), float)
        for i in range(len(a)):
            for j in range(len(b)):
                K[i, j] = self._kernel(a[i, :], b[j, :])
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
        self.noise = array([0.001] * len(self.trainx))
        # print(self.trainx, self.trainy)
        self.calculated = False

    def addDataset(self, dataset):
        """ adds the points from the dataset to the training set """
        assert (dataset.getDimension('input') == self.indim)
        assert (dataset.getDimension('target') == 1)

        self.trainx = r_[self.trainx, dataset.getField('input')]
        self.trainy = r_[self.trainy, ravel(dataset.getField('target'))]
        self.noise = array([0.001] * len(self.trainx))
        self.calculated = False

    def addSample(self, train, target):
        self.trainx = r_[self.trainx, asarray([train])]
        self.trainy = r_[self.trainy, asarray(target)]
        self.noise = r_[self.noise, array([0.001])]
        self.calculated = False

    def testOnArray(self, arr):
        self.testx = arr
        self._calculate()
        return self.pred_mean

    def _calculate(self):
        # calculate only of necessary
        if len(self.trainx) == 0:
            return

        # build covariance matrices
        train_train = self._buildCov(self.trainx, self.trainx)
        train_test = self._buildCov(self.trainx, self.testx)
        test_train = train_test.T
        test_test = self._buildCov(self.testx, self.testx)

        # calculate predictive mean and covariance
        K = train_train + self.noise * eye(len(self.trainx))

        if self.autonoise:
            # calculate average neighboring distance for auto-noise
            avgdist = 0
            sort_trainx = sort(self.trainx)
            for i, d in enumerate(sort_trainx):
                if i == 0:
                    continue
                avgdist += d - sort_trainx[i - 1]
            avgdist /= len(sort_trainx) - 1
            # sort(self.trainx)

            # add auto-noise from neighbouring samples (not standard gp)
            for i in range(len(self.trainx)):
                for j in range(len(self.trainx)):
                    if norm(self.trainx[i] - self.trainx[j]) > avgdist:
                        continue

                    d = norm(self.trainy[i] - self.trainy[j]) / (exp(norm(self.trainx[i] - self.trainx[j])))
                    K[i, i] += d

        self.pred_mean = self.mean + dot(test_train, solve(K, self.trainy - self.mean, sym_pos=0))
        self.pred_cov = test_test - dot(test_train, dot(inv(K), train_test))
        self.calculated = True

    def draw(self):
        if not self.calculated:
            self._calculate()

        return self.pred_mean + random.multivariate_normal(zeros(len(self.testx)), self.pred_cov)

    def plotCurves(self, showSamples=False, force2D=True):
        from pylab import clf, hold, plot, fill, title, gcf, pcolor, gray

        if not self.calculated:
            self._calculate()

        if self.indim == 1:
            clf()
            hold(True)
            if showSamples:
                # plot samples (gray)
                for _ in range(5):
                    plot(self.testx, self.pred_mean + random.multivariate_normal(zeros(len(self.testx)), self.pred_cov), color='gray')

            # plot training set
            plot(self.trainx, self.trainy, 'bx')
            # plot mean (blue)
            plot(self.testx, self.pred_mean, 'b', linewidth=1)
            # plot variance (as "polygon" going from left to right for upper half and back for lower half)
            fillx = r_[ravel(self.testx), ravel(self.testx[::-1])]
            filly = r_[self.pred_mean + 2 * diag(self.pred_cov), self.pred_mean[::-1] - 2 * diag(self.pred_cov)[::-1]]
            fill(fillx, filly, facecolor='gray', edgecolor='white', alpha=0.3)
            title('1D Gaussian Process with mean and variance')

        elif self.indim == 2 and not force2D:
            from matplotlib import axes3d as a3

            fig = gcf()
            fig.clear()
            ax = a3.Axes3D(fig) #@UndefinedVariable

            # plot training set
            ax.plot3D(ravel(self.trainx[:, 0]), ravel(self.trainx[:, 1]), ravel(self.trainy), 'ro')

            # plot mean
            (x, y, z) = [m.reshape(sqrt(len(m)), sqrt(len(m))) for m in (self.testx[:, 0], self.testx[:, 1], self.pred_mean)]
            ax.plot_wireframe(x, y, z, colors='gray')
            return ax

        elif self.indim == 2 and force2D:
            # plot mean on pcolor map
            gray()
            # (x, y, z) = map(lambda m: m.reshape(sqrt(len(m)), sqrt(len(m))), (self.testx[:,0], self.testx[:,1], self.pred_mean))
            m = floor(sqrt(len(self.pred_mean)))
            pcolor(self.pred_mean.reshape(m, m)[::-1, :])

        else: print("plotting only supported for indim=1 or indim=2.")


if __name__ == '__main__':

    from pylab import figure, show

    # --- example on how to use the GP in 1 dimension
    ds = SupervisedDataSet(1, 1)
    gp = GaussianProcess(indim=1, start= -3, stop=3, step=0.05)
    figure()

    x = mgrid[-3:3:0.2]
    y = 0.1 * x ** 2 + x + 1
    z = sin(x) + 0.5 * cos(y)

    ds.addSample(-2.5, -1)
    ds.addSample(-1.0, 3)
    gp.mean = 0

    # new feature "autonoise" adds uncertainty to data depending on
    # it's distance to other points in the dataset. not tested much yet.
    # gp.autonoise = True

    gp.trainOnDataset(ds)
    gp.plotCurves(showSamples=True)

    # you can also test the gp on single points, but this deletes the
    # original testing grid. it can be restored with a call to _buildGrid()
    print((gp.testOnArray(array([[0.4]]))))


    # --- example on how to use the GP in 2 dimensions

    ds = SupervisedDataSet(2, 1)
    gp = GaussianProcess(indim=2, start=0, stop=5, step=0.2)
    figure()

    x, y = mgrid[0:5:4j, 0:5:4j]
    z = cos(x) * sin(y)
    (x, y, z) = list(map(ravel, [x, y, z]))

    for i, j, k in zip(x, y, z):
        ds.addSample([i, j], [k])

    gp.trainOnDataset(ds)
    gp.plotCurves()

    show()
