__author__ = "Martin Felder, felder@in.tum.de"


try:
    from svm import svm_model, svm_parameter, svm_problem, cross_validation #@UnresolvedImport
    from svm import C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR #@UnresolvedImport @UnusedImport
    from svm import LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED #@UnresolvedImport @UnusedImport
except ImportError:
    raise ImportError("Cannot find LIBSVM installation. Make sure svm.py and svmc.* are in the PYTHONPATH!")

from numpy import * #@UnusedWildImport
import logging


class SVMTrainer(object):
    """A class performing supervised learning of a DataSet by an SVM unit. See 
    the remarks on :class:`SVMUnit` above. This whole class is a bit of a hack,
    and provided mostly for convenience of comparisons."""
    
    def __init__(self, svmunit, dataset, modelfile=None, plot=False):
        """ Initialize data and unit to be trained, and load the model, if 
        provided.
        
        The passed `svmunit` has to be an object of class :class:`SVMUnit` 
        that is going to be trained on the :class:`ClassificationDataSet` object
        dataset. 
        Compared to FNN training we do not use a test data set, instead 5-fold 
        cross-validation is performed if needed.
        
        If `modelfile` is provided, this model is loaded instead of training.
        If `plot` is True, a grid search is performed and the resulting pattern
        is plotted."""
        self.svm = svmunit
        self.ds = dataset
        self.svmtarget = dataset['target'].flatten()
        self.plot = plot
        self.searchlog = 'gridsearch_results.txt'
        # set default parameters for training
        self.params = {
            'kernel_type':RBF
            }

        if modelfile is not None:  self.load(modelfile)
        
        
    def train(self, search=False, **kwargs):
        """ Train the SVM on the dataset. For RBF kernels (the default), an optional meta-parameter search can be performed.

        :key search: optional name of grid search class to use for RBF kernels: 'GridSearch' or 'GridSearchDOE' 
        :key log2g: base 2 log of the RBF width parameter
        :key log2C: base 2 log of the slack parameter
        :key searchlog: filename into which to dump the search log
        :key others: ...are passed through to the grid search and/or libsvm 
        """
        
        self.setParams(**kwargs)
        problem = svm_problem(self.ds['target'].flatten(), self.ds['input'].tolist())
        if search:
            # this is a bit of a hack...
            model = eval(search + "(problem, self.svmtarget, cmin=[0,-7],cmax=[25,1], cstep=[0.5,0.2],plotflag=self.plot,searchlog=self.searchlog,**self.params)")
        else:
            param = svm_parameter(**self.params)
            model = svm_model(problem, param)
            logging.info("Training completed with parameters:")
            logging.info(repr(param))

        self.svm.setModel(model)
        
        
    def save(self, filename):
        """ save the trained SVM """
        self.svm.saveModel(filename)
        
    
    def load(self, filename):
        """ no training at all - just load the SVM model from a file """
        self.svm.loadModel(filename)
    
    def setParams(self, **kwargs):
        """ Set parameters for SVM training. Apart from the ones below, you can use all parameters 
        defined for the LIBSVM svm_model class, see their documentation.

        :key searchlog: Save a list of coordinates and the achieved CV accuracy to this file."""
        if kwargs.has_key('weight'):
            self.params['nr_weight'] = len(kwargs['weight'])
        if kwargs.has_key('log2C'):
            self.params['C'] = 2 ** kwargs['log2C']
            kwargs.pop('log2C')
        if kwargs.has_key('log2g'):
            self.params['gamma'] = 2 ** kwargs['log2g']
            kwargs.pop('log2g')
        if kwargs.has_key('searchlog'):
            self.searchlog = kwargs['searchlog']
            kwargs.pop('searchlog')
        self.params.update(kwargs)

        
        
class GridSearch(svm_model):
    """Helper class used by :class:`SVMTrainer` to perform an exhaustive grid search, and plot the
    resulting accuracy surface, if desired. Adapted from the LIBSVM python toolkit."""
    
    allPts = []
    allScores = []
    
    def __init__(self, problem, targets, cmin, cmax, cstep=None, crossval=5,
                 plotflag=False, maxdepth=8, searchlog='gridsearch_results.txt', **params):
        """ Set up (log) grid search over the two RBF kernel parameters C and gamma.

        :arg problem: the LIBSVM svm_problem to be optimized, ie. the input and target data
        :arg targets: unfortunately, the targets used in the problem definition have to be given again here
        :arg cmin: lower left corner of the log2C/log2gamma window to search
        :arg cmax: upper right corner of the log2C/log2gamma window to search
        :key cstep: step width for log2C and log2gamma (ignored for DOE search)
        :key crossval: split dataset into this many parts for cross-validation
        :key plotflag: if True, plot the error surface contour (regular) or search pattern (DOE)
        :key maxdepth: maximum window bisection depth (DOE only)
        :key searchlog: Save a list of coordinates and the achieved CV accuracy to this file
        :key others: ...are passed through to the cross_validation method of LIBSVM
        """
        self.nPars = len(cmin)
        self.usermin = cmin
        self.usermax = cmax
        self.userstep = cstep
        self.crossval = crossval
        self.plotflag = plotflag
        self.maxdepth = maxdepth  # number of zoom-in steps (DOE search only!)
        
        # set default parameters for training
        self.params = params
        
        if self.plotflag:
            import pylab as p 
            p.ion()
            p.figure(figsize=[12, 8])
        
        assert isinstance(problem, svm_problem)
        self.problem = problem
        self.targets = targets
        
        self.resfile = open(searchlog, 'w')

        # do the parameter searching
        param = self.search()
        
        if self.plotflag: 
            p.ioff()
            p.show()
        
        self.resfile.close()
        svm_model.__init__(self, problem, param)
        
    def setParams(self, **kwargs):
        """ set parameters for SVM training """
        if kwargs.has_key('weight'):
            self.params['nr_weight'] = len(kwargs['weight'])
        self.params.update(kwargs)
    
    def search(self):
        """ iterate successive parameter grid refinement and evaluation; adapted from LIBSVM grid search tool """
        jobs = self.calculate_jobs()
        scores = []
        for line in jobs:
            for (c, g) in line:
                # run cross-validation for this point
                self.setParams(C=2 ** c, gamma=2 ** g)
                param = svm_parameter(**self.params)
                cvresult = array(cross_validation(self.problem, param, self.crossval))
                corr, = where(cvresult == self.targets)
                res = (c, g, float(corr.size) / self.targets.size)                
                scores.append(res)
                self._save_points(res)
            self._redraw(scores)
        scores = array(scores)
        best = scores[scores[:, 0].argmax(), 1:]
        self.setParams(C=2 ** best[0], gamma=2 ** best[1])
        logging.info("best log2C=%12.7g, log2g=%11.7g " % (best[0], best[1]))
        param = svm_parameter(**self.params)
        return param
    
        
    def _permute_sequence(self, seq):
        """ helper function to create a nice sequence of refined regular grids; from LIBSVM grid search tool """
        n = len(seq)
        if n <= 1: return seq
    
        mid = int(n / 2)
        left = self._permute_sequence(seq[:mid])
        right = self._permute_sequence(seq[mid + 1:])
    
        ret = [seq[mid]]
        while left or right:
            if left: ret.append(left.pop(0))
            if right: ret.append(right.pop(0))
    
        return ret

    def _range_f(self, begin, end, step):
        """ like range, but works on non-integer too; from LIBSVM grid search tool """ 
        seq = []
        while 1:
            if step > 0 and begin > end: break
            if step < 0 and begin < end: break
            seq.append(begin)
            begin = begin + step
        return seq

    def calculate_jobs(self):
        """ like range, but works on non-integer too; from LIBSVM grid search tool """ 
        c_seq = self._permute_sequence(self._range_f(self.usermin[0], self.usermax[0], self.userstep[0]))
        g_seq = self._permute_sequence(self._range_f(self.usermin[1], self.usermax[1], self.userstep[1]))
        nr_c = float(len(c_seq))
        nr_g = float(len(g_seq))
        global total_points
        total_points = (nr_g + 1) * (nr_g)
        i = 0
        j = 0
        jobs = []
    
        while i < nr_c or j < nr_g:
            if i / nr_c < j / nr_g:
                # increase C resolution
                line = []
                for k in range(0, j):
                    line.append((c_seq[i], g_seq[k]))
                i = i + 1
                jobs.append(line)
            else:
                # increase g resolution
                line = []
                for k in range(0, i):
                    line.append((c_seq[k], g_seq[j]))
                j = j + 1
                jobs.append(line)
        return jobs

    def _save_points(self, res):
        """ save the list of points and corresponding scores into a file """
        self.resfile.write("%g, %g, %g\n" % res)
        logging.info("log2C=%g, log2g=%g, res=%g" % res)
        self.resfile.flush()
        
    def _redraw(self, db, tofile=0, eta=None):
        """ redraw the updated grid interactively """
        import pylab as p 
        if len(db) <= 3 or not self.plotflag: return
        #begin_level = round(max(map(lambda(x):x[2],db))) - 3
        #step_size = 0.25
        nContours = 25
        #suffix = ''
        #if eta is not None:
        #    suffix = " (ETA: %5.2f min)" % eta
        def cmp (x, y):
            if x[0] < y[0]: return - 1
            if x[0] > y[0]: return 1
            if x[1] > y[1]: return - 1
            if x[1] < y[1]: return 1
            return 0
        db.sort(cmp)
        dbarr = p.asarray(db)
        # reconstruct grid: array is ordered along first and second dimension
        x = dbarr[:, 0]
        dimy = len(x[x == x[0]])
        dimx = x.size / dimy
        print('plotting: ', dimx, dimy)
        x = x.reshape(dimx, dimy)
        y = dbarr[:, 1]
        y = y.reshape(dimx, dimy)
        z = dbarr[:, 2].reshape(dimx, dimy)
    
        # plot using manual double buffer
        p.ioff()
        p.clf()
        p.contourf(x, y, z, nContours)
        p.hsv()
        p.colorbar()
        p.xlim(self.usermin[0], self.usermax[0])
        p.ylim(self.usermin[1], self.usermax[1])
        p.xlabel(r'$\rm{log}_2(C)$')
        p.ylabel(r'$\rm{log}_2(\gamma)$')
        p.ion()
        p.draw_if_interactive()

        

class GridSearchDOE(GridSearch):
    """ Same as GridSearch, but implements a design-of-experiments based search pattern, as
    described by C. Staelin, http://www.hpl.hp.com/techreports/2002/HPL-2002-354R1.pdf """

    # DOE pattern; the last 5 points do not need to be calculated when refining the grid 
    doepat = array([[0.5, 1], [0.25, 0.75], [0.75, 0.75], [0, 0.5], [1, 0.5], \
              [0.25, 0.25], [0.75, 0.25], [0.5, 0], [0, 1], [1, 1], [0.5, 0.5], [0, 0], [1, 0]])
    nPts = 13
    depth = 0
    
    def search(self, cmin=None, cmax=None):
        """ iterate parameter grid refinement and evaluation recursively """
        if self.depth > self.maxdepth:
            # maximum search depth reached - finish up
            best = self.allPts[self.allScores.argmax(), :]
            logging.info("best log2C=%12.7g, log2g=%11.7g " % (best[0], best[1]))
            self.setParams(C=2 ** best[0], gamma=2 ** best[1])
            param = svm_parameter(**self.params)
            logging.info("Grid search completed! Final parameters:")
            logging.info(repr(param))
            return param
        
        # generate DOE gridpoints using current range
        if cmin is None:
            # use initial values, if none given
            cmin = array(self.usermin)
            cmax = array(self.usermax)
        points = self.refineGrid(cmin, cmax)
        
        # calculate scores for all grid points using n-fold cross-validation
        scores = []
        isnew = array([True] * self.nPts)
        for i in range(self.nPts):
            idx = self._findIndex(points[i, :])
            if idx >= 0:
                # point already exists
                isnew[i] = False
                scores.append(self.allScores[idx])
            else: 
                # new point, run cross-validation
                self.setParams(C=2 ** points[i, 0], gamma=2 ** points[i, 1])
                param = svm_parameter(**self.params)
                cvresult = array(cross_validation(self.problem, param, self.crossval))
                # save cross validation result as "% correct" 
                corr, = where(cvresult == self.targets)
                corr = float(corr.size) / self.targets.size
                scores.append(corr)
                self._save_points((points[i, 0], points[i, 1], corr))

        scores = array(scores)
        
        # find max and new ranges by halving the old ones, whereby
        # entire search region must lie within original search range
        newctr = points[scores.argmax(), :].copy()
        newdiff = (cmax - cmin) / 4.0
        for i in range(self.nPars):
            newctr[i] = min([max([newctr[i], self.usermin[i] + newdiff[i]]), self.usermax[i] - newdiff[i]])
        cmin = newctr - newdiff
        cmax = newctr + newdiff
        logging.info("depth:\t%3d\tcrange:\t%g\tscore:\t%g" % (self.depth, cmax[0] - cmin[0], scores.max()))
        
        # append points and scores to the full list
        if self.depth == 0:
            self.allPts = points[isnew, :].copy()
            self.allScores = scores[isnew].copy()
        else:
            self.allPts = append(self.allPts, points[isnew, :], axis=0)
            self.allScores = append(self.allScores, scores[isnew], axis=0)
        
        if self.plotflag:
            import pylab as p 
            if self.depth == 0:
                self.oPlot = p.plot(self.allPts[:, 0], self.allPts[:, 1], 'o')[0]
            # insert new data into plot
            self.oPlot.set_data(self.allPts[:, 0], self.allPts[:, 1])
            p.draw()
        
        # recursively call ourselves
        self.depth += 1    
        return self.search(cmin, cmax)
    
    
    def refineGrid(self, cmin, cmax):
        """ given grid boundaries, generate the corresponding DOE pattern from template"""
        diff = array((cmax - cmin).tolist()*self.nPts).reshape(self.nPts, self.nPars)
        return self.doepat * diff + array(cmin.tolist()*self.nPts).reshape(self.nPts, self.nPars)
    
    def _findIndex(self, point):
        """ determines whether given point already exists in list of all calculated points.
        raises exception if more than one point is found, returns -1 if no point is found """
        if self.depth == 0: return - 1
        check = self.allPts[:, 0] == point[0]
        for i in range(1, point.size):
            check = check & (self.allPts[:, i] == point[i])
        idx, = where(check)
        if idx.size == 0:
            return - 1
        elif idx.size > 1:
            logging.error("Something went wrong - found more than one matching point!")
            logging.error(str(point))
            logging.error(str(self.allPts))
            raise
        else:
            return idx[0]


