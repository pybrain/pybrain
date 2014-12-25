from __future__ import print_function

__author__ = 'Michael Isik'

from pybrain.tools.validation import CrossValidator
from numpy import linspace, append, ones, zeros, array, where, apply_along_axis
import copy

class GridSearch2D:
    """ Abstract class providing a method for searching optimal metaparmeters
        of a training algorithm.

        It is especially designed for searching two metaparameters, which
        explains the "2D" in "GridSearch2D". To use it, one must create a
        subclass and implement the _validate() method.
        See GridSearchCostGamma for an example.

        On construction, the minima, maxima and granularity of the search space
        must be defined. Then a list of jobs are calculated. Each job stands for
        a metaparameter setting, that will be validated in advance.
        The special job execution order makes it possible to visualize the
        progress of the Gridsearch at several intermediate steps.
    """
    def __init__(self, min_params, max_params, n_steps=7, **kwargs):
        """ :key min_params: Tuple of two elements specifying the minima
                               of the two metaparameters
            :key max_params: Tuple of two elements specifying the minima
                               of the two metaparameters
            :key max_param:  Tuple of two elements, specifying the number of
                               steps between the minimum and maximum of each search
                               dimension. Alternative, specify a scalar to set
                               the same granularity for each dimension.
            :key **kwargs:   See setArgs()
        """
        assert len(min_params) == len(max_params)

        self._min_params = array(min_params, float)
        self._max_params = array(max_params, float)
        self._n_dim = len(min_params)
        self._n_steps = append([], n_steps) * ones(self._n_dim)
        self._range = self._max_params - self._min_params
        self._performances = {}
        self._verbosity = 0
        self.setArgs(**kwargs)

    def setArgs(self, **kwargs):
        """ :key **kwargs:
                verbosity : set verbosity
        """
        for key, value in list(kwargs.items()):
            if key in ("verbose", "verbosity", "ver", "v"):
                self._verbosity = value

    def getPerformances(self):
        """ Returns the performances calculated so far. They are stored inside
            a dictionary, mapping jobs to performances. A job is a tuple of
            metaparameters.
        """
        return copy.copy(self._performances)

    def search(self):
        """ The main search method, that validates all calculated metaparameter
            settings (=jobs) by calling the abstract _validate() method.
            After enough new jobs were validated in order to visualize a grid,
            the _onStep() callback method is called.
        """
        jobs = self._calculateJobs()
        perfs = self._performances
        for line in jobs:
            for params in line:
                perf = self._validate(params)
                perfs[params] = perf
                if self._verbosity > 0:
                    print(("validated:", params, " performance = ", perf))

            self._onStep()

        max_idx = array(list(perfs.values())).argmax()
        return list(perfs.keys())[max_idx]

    def _validate(self, params):
        """ Abstract validation method. Should validate the supplied metaparameters,
            and return this performance on the learning task.
        """
        raise NotImplementedError()

    def _onStep(self):
        """ Callback function, that gets called after a gridvisualization
            could be updated because of new performance values.
            Overwrite this function to make use of it.
        """
        pass

    def _calculateJobs(self):
        """ Calculate and return the metaparameter settings to be validated (=jobs).
        """
        ndim = len(self._min_params)
        linspaces = []
        for i in range(ndim):
            linspaces.append(
                self._permuteSequence(
                    list(linspace(self._min_params[i], self._max_params[i], self._n_steps[i]))))
#        print(linspaces; exit(0))
#        linspaces = array(linspaces,float)
        nr_c = len(linspaces[0])
        nr_g = len(linspaces[1])
        i = 0
        j = 0
        jobs = []

        while i < nr_c or j < nr_g:
            if i / float(nr_c) < j / float(nr_g):
                line = []
                for k in range(0, j):
                    line.append((linspaces[0][i], linspaces[1][k]))
                i += 1
                jobs.append(line)
            else:
                line = []
                for k in range(0, i):
                    line.append((linspaces[0][k], linspaces[1][j]))
                j += 1
                jobs.append(line)
        return jobs
#        return grid


    def _permuteSequence(self, seq):
        """ Helper function for calculating the job list
        """
        n = len(seq)
        if n <= 1: return seq

        mid = int(n / 2)
        left = self._permuteSequence(seq[:mid])
        right = self._permuteSequence(seq[mid + 1:])

        ret = [seq[mid]]
        while left or right:
            if left:  ret.append(left.pop(0))
            if right: ret.append(right.pop(0))

        return ret




class GridSearchDOE:
    """ Abstract class providing a method for searching optimal metaparmeters
        of a training algorithm after the DOE principle.
        Read: "Parameter selection for support vector machines"
              Carl Staelin, Senior Member IEEE
        for more information

        To use this class, one must create a subclass and implement the
        _validate() method. See GridSearchDOECostGamma for an example.
    """
    _doe_pat = array([ [ -1.0 , +1.0 ]                , [ 0.0, +1.0 ]                , [ +1.0 , +1.0 ] ,
                                       [ -0.5, +0.5 ] , [ +0.5, +0.5 ] ,
                       [ -1.0 , 0.0 ]                , [ 0.0, 0.0 ]                , [ +1.0 , 0.0 ] ,
                                       [ -0.5, -0.5 ] , [ +0.5, -0.5 ] ,
                       [ -1.0 , -1.0 ]                , [ 0.0, -1.0 ]                , [ +1.0 , -1.0 ]   ])

    def __init__(self, min_params, max_params, n_iterations=5, **kwargs):
        """ See GridSearch.init()
        """
        assert len(min_params) == len(max_params)
        self._min_params = array(min_params, float)
        self._max_params = array(max_params, float)
        self._n_iterations = n_iterations
        self._refine_factor = 2.
        self._range = self._max_params - self._min_params
        self._performances = {}
        self._verbosity = 0
        self.setArgs(**kwargs)

    def setArgs(self, **kwargs):
        """ :key **kwargs:
                verbosity : set verbosity
        """
        for key, value in list(kwargs.items()):
            if key in ("verbose", "ver", "v"):
                self._verbosity = value

    def search(self):
        """ The main search method, that validates all calculated metaparameter
            settings by calling the abstract _validate() method.
        """
        self._n_params = len(self._min_params)

        center = self._min_params + self._range / 2.
        for level in range(self._n_iterations):
            grid = self._calcGrid(center, level)
            local_perf = apply_along_axis(self._validateWrapper, 1, grid)

            max_idx = local_perf.argmax()
            center = grid[max_idx]
            if self._verbosity > 0:
                print()
                print(("Found maximum at:", center, "   performance = ", local_perf[max_idx]))
                print()

        return center


    def _validateWrapper(self, params):
        """ Helper function that wraps the _validate() method.
        """
        perf = self._validate(params)
        if self._verbosity > 0:
            print(("validated:", params, " performance = ", perf))
        self._performances[tuple(params)] = perf
        return perf

    def _calcGrid(self, center, level):
        """ Calculate the next grid to validate.

            :arg center: The central position of the grid
            :arg level:  The iteration number
        """
        local_range = self._range / (self._refine_factor ** level)
        scale = local_range / 2
        translation = center
        grid = self._doe_pat * scale + translation

        grid = self._moveGridIntoBounds(grid)

        return grid

    def _moveGridIntoBounds(self, grid):
        """ If the calculated grid is out of bounds,
            this method moves it back inside, and returns the new grid.
        """
        grid = array(grid)
        local_min_params = grid.min(axis=0)
        local_max_params = grid.max(axis=0)
        tosmall_idxs, = where(local_min_params < self._min_params)
        togreat_idxs, = where(local_max_params > self._max_params)
        translation = zeros(self._n_params)
        for idx in tosmall_idxs:
            translation[idx] = self._min_params[idx] - local_min_params[idx]
        for idx in togreat_idxs:
            translation[idx] = self._max_params[idx] - local_max_params[idx]
        grid += translation
        return grid

    def _validate(self, params):
        """ Abstract validation method. Should validate the supplied metaparameters,
            and return this performance on the learning task.
        """
        raise NotImplementedError()


class GridSearchCostGamma(GridSearch2D):
    """ GridSearch class, that searches for the optimal cost and gamma values
        of a support vector machine. See SVMTrainer and the SVM module for
        more information on these metaparameters.
        The parameters are searched in log2-space. Crossvalidation is used
        to determine the performance values.
    """
    def __init__(self, trainer, dataset, min_params=[-5, -15], max_params=[15, 3], n_steps=7, **kwargs):
        """ The parameter boundaries are specified in log2-space.

            :arg trainer: The SVM trainer including the SVM module.
                            (Could be any kind of trainer and module)
            :arg dataset: Dataset used for crossvalidation
        """
        GridSearch2D.__init__(self, min_params, max_params, n_steps)
        self._trainer = trainer
        self._dataset = dataset

        self._validator_kwargs = {}
        self._n_folds = 5
        self.setArgs(**kwargs)


    def setArgs(self, **kwargs):
        """ :key **kwargs:
                nfolds    : Number of folds of crossvalidation
                max_epochs: Maximum number of epochs for training
                verbosity : set verbosity
        """
        for key, value in list(kwargs.items()):
            if key in ("folds", "nfolds"):
                self._n_folds = int(value)
            elif key in ("max_epochs"):
                self._validator_kwargs['max_epochs'] = value
            elif key in ("verbose", "ver", "v"):
                self._verbosity = value
            else:
                GridSearch2D.setArgs(self, **{key:value})

    def _validate(self, params):
        """ The overridden validate function, that uses cross-validation in order
            to determine the params' performance value.
        """
        trainer = self._getTrainerForParams(params)
        return CrossValidator(trainer, self._dataset, self._n_folds, **self._validator_kwargs).validate()

    def _getTrainerForParams(self, params):
        """ Returns a trainer, loaded with the supplied metaparameters.
        """
        trainer = copy.deepcopy(self._trainer)
        trainer.setArgs(cost=2 ** params[0], gamma=2 ** params[1], ver=0)
        return trainer




class GridSearchDOECostGamma(GridSearchDOE):
    """ Same as GridSearchCostGamma, except, that it uses the Design of Experiment (DOE)
        algorithm.
    """
    def __init__(self, trainer, dataset, min_params=[-5, -15], max_params=[15, 3], n_iterations=5, **kwargs):
        """ See GridSearchCostGamma and GridSearchDOE """
        GridSearchDOE.__init__(self, min_params, max_params, n_iterations)
        assert len(min_params) == 2
        self._trainer = trainer
        self._dataset = dataset

        self._n_folds = 5
        self._validator_kwargs = {}
        self.setArgs(**kwargs)

    def setArgs(self, **kwargs):
        """ See GridSearchCostGamma """
        for key, value in list(kwargs.items()):
            if key in ("folds", "nfolds"):
                self._n_folds = int(value)
            elif key in ("max_epochs"):
                self._validator_kwargs['max_epochs'] = value
            else:
                GridSearchDOE.setArgs(self, **{key:value})


    def _validate(self, params):
        """ See GridSearchCostGamma """
        glob_idx = tuple(params)
        perf = self._performances

        if glob_idx not in perf:
            trainer = self._getTrainerForParams(params)
            local_perf = CrossValidator(trainer, self._dataset, self._n_folds, **self._validator_kwargs).validate()
            perf[glob_idx] = local_perf
        else:
            local_perf = perf[glob_idx]
        return local_perf


    def _getTrainerForParams(self, params):
        """ See GridSearchCostGamma """
        trainer = copy.deepcopy(self._trainer)
        trainer.setArgs(cost=2 ** params[0], gamma=2 ** params[1], ver=0)
        return trainer





