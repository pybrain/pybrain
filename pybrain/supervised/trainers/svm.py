__author__ = 'Michael Isik'


import numpy
from numpy            import array, zeros, empty, where, float, float64, int, Infinity
from trainer          import Trainer


class SVMTrainer(Trainer):
    """ The trainer class for regular SVM modules.

        During training, the multiplier stored inside the alpha array are
        adapted in order to find a solution for following optimization problem:

            min_alpha f(alpha) = 0.5 * alpha' * Q * alpha - sum(alpha)

        Following constraints must not be violated:

            (1) 0 <= alpha_i <= C
            (2) y' * alpha = 0

        With:
             - Q[i][j] = Y[i] * Y[j] * k( X[i], X[j] )
             - X is the array of input vectors in the training set
             - Y is the array of target values (-1, 1) of the training set
             - k is the selected kernel function
             - C is the cost value that acts as upper bound for alpha multiplier

        The threshold value beta, which is used for calculating the functional
        margin ( see SVM.functionalMargin() ) can be generated at any time
        given alpha, X and Y.
    """

    _LOWER_BOUND = -1
    _FREE        =  0
    _UPPER_BOUND =  1
    _EPS         = 10. ** - 3
    _TAU         = 10. ** -12

    def __init__(self, module, dataset=None, **kwargs):
        """ Initializes the Trainer.
            @param **kwargs: see setParams()
        """
        Trainer.__init__(self,module)

        self._C_desc = 1.0
        self._C = []
        self._verbosity = 0.
        self.setParams(**kwargs)

        self._i = 0
        self._j = 0
        self.stepno = -1

        if dataset: self.setData(dataset)
        self._fully_initialized = False
        self._refreshC()

    def setParams(self,**kwargs):
        """ Set trainer's parameters
            @param **kwargs: allowed keys:
                - cost: Set's the cost value for C-SVMs. The
        """
        for key, value in kwargs.items():
            if key in ("cost"):
                self._C_desc = value
                self._refreshC()
            elif key in ("verbose", "ver", "v"):
                self._verbosity = value
            else:
                self.module.setParams(**{key:value})

#        assert not ( isinstance(self._C_desc, dict) and not self.module._one_against_one )
#       assert self._C > 0

    def setData(self,dataset):
        """ Set training data by supplying an instance of SupervisedDataSet. """
        self.module._setData(dataset)
        self._refreshC()


    def _refreshC(self):
        l = len(self.module._kernel._X)
        if l <= 0: return

        self._C = empty(l,float)
        if isinstance(self._C_desc, dict):
            assert set(self.module._classes).issubset( set(self._C_desc.keys()) )
            m = self.module
            Y = m._kernel._Y
            for key,value in self._C_desc.items():
                rawout = m.classToRawOutput(key)
                if rawout != None:
                    w, = where( Y == rawout )
                    assert value > 0
                    self._C[w] = float(value)
        else:
            assert float(self._C_desc) > 0
            self._C[:] = float(self._C_desc)


    def _onStep(self):
        """ Callback function that gets called once before each training step.
            Override it in order to make use of it.
        """


    def train(self):
        """ The main train method. Optimizes the multiplier stored in alpha.

            In each iteration of the main training loop two multiplier are selected
            by the selectWorkingSet() method and adapted in order to minimize the
            objective function f(alpha) while preserving the constraints.
            All other multiplier are held fixed.
        """
        while self.step(False):
            pass

        self.updateBeta()

        if self._verbosity > 0:
            print "iteration:", self.stepno, "   free =", self._free_num, "  lb =", self._lb_num, "  ub =", self._ub_num, "objective = ", self.calculateObjectiveValue()
            print("\n=== Training finished")
        # calculate objective value
        #dbg( "objective = ", self.calculateObjectiveValue() )

    def trainEpochs(self,epochs=0):
        if epochs>0:
            for i in range(epochs):
                if not self.step(False): break
            self.updateBeta()
        else:
            self.train()
        if self._verbosity > 0:
            print "iteration:", self.stepno, "   free =", self._free_num, "  lb =", self._lb_num, "  ub =", self._ub_num, "objective = ", self.calculateObjectiveValue()
            print("\n=== Training finished")



    def _initialize(self):
        # kernel stuff
        kernel = self.module._kernel
        l      = kernel.l
        Y      = kernel._Y
        QD     = kernel._QD


        # initialize alpha array
        self.module._alpha = zeros(l,dtype=float64)
        alpha = self.module._alpha

        # precalculated conditions and alpha status
        self._cond_pos_Y               = ( Y ==  1 )
        self._cond_neg_Y               = ( Y == -1 )
        self._cond_not_lower           = empty( l, dtype=bool )
        self._cond_not_upper           = empty( l, dtype=bool )
        self._cond_not_upper_and_pos_Y = empty( l, dtype=bool )
        self._cond_not_lower_and_pos_Y = empty( l, dtype=bool )
        self._cond_not_upper_and_neg_Y = empty( l, dtype=bool )
        self._cond_not_lower_and_neg_Y = empty( l, dtype=bool )
        self._alpha_status   = array([-10 for i in range(l)],dtype=int) # initialize with undefined value (-10)
        self._lb_num   = 0
        self._free_num = 0
        self._ub_num   = 0
        for i in range(l): self._updateAlphaStatus(i)
        UPPER_BOUND = self._UPPER_BOUND
        LOWER_BOUND = self._LOWER_BOUND
        FREE        = self._FREE

        # set active_size and active_set
        self._active_size = l
        active_size       = self._active_size
        self._active_set  = array(range(active_size),dtype=int)
        active_set        = self._active_set


        # initialize gradient arrays
        self._G     = array([-1 for i in range(l)],dtype=float)
        self._G_bar = zeros(l,dtype=float)
        G     = self._G
        G_bar = self._G_bar

        # todo: not fully optimized yet
        for i in where(self._cond_not_lower)[0]:
            Q_i = kernel.getQRow(i)
            G  += alpha[i] * Q_i
            if self._isUpperBound(i):
                for j in range(l):
                    G_bar[j] += self._getC(i) * Q_i[j]
        self._fully_initialized = True

    



    def step(self, updatebeta=True):
#        return self.stepp()
        
        if not self._fully_initialized:
            self._initialize()

        # kernel stuff
        kernel = self.module._kernel
        l      = kernel.l
        Y      = kernel._Y
        QD     = kernel._QD
        alpha = self.module._alpha
        UPPER_BOUND = self._UPPER_BOUND
        LOWER_BOUND = self._LOWER_BOUND
        FREE        = self._FREE
        active_size       = self._active_size
        active_set        = self._active_set
        G     = self._G
        G_bar = self._G_bar





        # a new step begins
        self.stepno += 1
        if not (self.stepno % 100) :
            if self._verbosity > 0:
                print "iteration:", self.stepno, "   free =", self._free_num, "  lb =", self._lb_num, "  ub =", self._ub_num, "  objective =", self.calculateObjectiveValue()

        # determine working set
        i,j = self._selectWorkingSet()
#        i,j = selectWorkingSet(self)
        if i<0: return False

#        ni,nj = selectWorkingSet(self)
#        print
#        print "original: ", i, j
#        print "new     : ", ni,nj

        self._i = i
        self._j = j

        self._onStep()

        # fetch some values
        Q_i = self.module._kernel.getQRow(i)
        Q_j = self.module._kernel.getQRow(j)
        C_i = self._getC(i)
        C_j = self._getC(j)

        old_alpha_i = alpha[i]
        old_alpha_j = alpha[j]

        # optimize alpha values of working set
        if Y[i] != Y[j]:
            a = Q_i[i] + Q_j[j] + 2*Q_i[j]
            if a <= 0:
                a = self._TAU
            delta = ( -G[i] - G[j] ) / a
            diff  = alpha[i] - alpha[j]
            alpha[i] += delta
            alpha[j] += delta

            if diff > 0:
                if  alpha[j] < 0:
                    alpha[j] = 0
                    alpha[i] = diff
            else:
                if  alpha[i] < 0:
                    alpha[i] = 0
                    alpha[j] = -diff
            if diff > C_i - C_j:
                if  alpha[i] > C_i:
                    alpha[i] = C_i
                    alpha[j] = C_i - diff
            else:
                if  alpha[j] > C_j:
                    alpha[j] = C_j
                    alpha[i] = C_j + diff
        else:
            a = Q_i[i] + Q_j[j] - 2*Q_i[j]
            if a <= 0:
                a = self._TAU
            delta = ( G[i] - G[j] ) / a
            sum_  = alpha[i] + alpha[j]
            alpha[i] -= delta
            alpha[j] += delta

            if sum_ > C_i:
                if  alpha[i] > C_i:
                    alpha[i] = C_i
                    alpha[j] = sum_ - C_i
            else:
                if  alpha[j] < 0:
                    alpha[j] = 0
                    alpha[i] = sum_
            if sum_ > C_j:
                if  alpha[j] > C_j:
                    alpha[j] = C_j
                    alpha[i] = sum_ - C_j
            else:
                if  alpha[i] < 0:
                    alpha[i] = 0
                    alpha[j] = sum_

        # update gradient
        delta_alpha_i = alpha[i] - old_alpha_i
        delta_alpha_j = alpha[j] - old_alpha_j

        G[0:active_size] += Q_i[0:active_size] * delta_alpha_i+Q_j[0:active_size] * delta_alpha_j

        # update alpha status and G_bar
        i_is_upper_bound_old = self._isUpperBound(i)
        j_is_upper_bound_old = self._isUpperBound(j)
        self._updateAlphaStatus(i)
        self._updateAlphaStatus(j)


        if i_is_upper_bound_old != self._isUpperBound(i):
            Q_i = kernel.getQRow(i)
            if i_is_upper_bound_old: G_bar -= C_i * Q_i
            else:                    G_bar += C_i * Q_i
        if j_is_upper_bound_old != self._isUpperBound(i):
            Q_j = kernel.getQRow(j)
            if j_is_upper_bound_old: G_bar -= C_j * Q_j
            else:                    G_bar += C_j * Q_j


        if updatebeta:
            # update the beta threshold value
            self.updateBeta()


        return True


    def _selectWorkingSet(self):
        """ Selects two multiplier that should be optimized.

            Returns two indices for the alpha array. The corresponding alpha
            values will be optimized due to further processing by the training
            loop. The indices are choosen by the maximal violating pair strategy
        """
        # set local variables
        Y  = self.module._kernel._Y
        G  = self._G
        QD = self.module._kernel._QD
        active_size  = self._active_size
        UPPER_BOUND  = self._UPPER_BOUND
        LOWER_BOUND  = self._LOWER_BOUND
        alpha_status = self._alpha_status


        # === find G_max1 and G_max_idx (=i)
        G_max11    = -Infinity
        G_max_idx1 = -1
        where1,    = where( self._cond_not_upper_and_pos_Y )
        if len(where1):
            G_max_idx1 = numpy.argmin( G[where1] )
            G_max_idx1 = where1[G_max_idx1]
            G_max11    = -G[G_max_idx1]

        G_max12    = -Infinity
        G_max_idx2 = -1
        where1,    = where( self._cond_not_lower_and_neg_Y )
        if len(where1):
            G_max_idx2 = numpy.argmax(G[where1])
            G_max_idx2 = where1[G_max_idx2]
            G_max12     = G[G_max_idx2]




        if  G_max11  >= G_max12:
            G_max1    = G_max11
            G_max_idx = G_max_idx1
        else:
            G_max1    = G_max12
            G_max_idx = G_max_idx2

        i = G_max_idx



        Q_i = None
        if i != -1: Q_i = self.module._kernel.getQRow(i)

        # === find G_max2 and G_min_idx(=j)

        # find G_min_idx1 obj_diff_min1 and G_max21
        G_min_idx1    = -1
        G_max21       = -Infinity
        obj_diff_min1 = Infinity

        where1, = where(self._cond_not_lower_and_pos_Y)

        if len(where1):
            G_filt    = G[where1]
            G_max21   = G_filt.max()

            grad_diff = G_max1 + G_filt

            where2, = where(grad_diff > 0)
            where1_where2 = where1[where2]
            a = Q_i[i] + QD[where1_where2] - 2. * Y[i] * Q_i[where1_where2]

            where3a, = where( a >  0 )
            where3b, = where( a <= 0 )

            obj_diff = empty(len(a),dtype=float)
            obj_diff[where3a] = numpy.negative( grad_diff[where2[where3a]] ** 2 ) / a[where3a]
            obj_diff[where3b] = numpy.negative( grad_diff[where2[where3a]] ** 2 ) / self._TAU

            if len(obj_diff):
                obj_diff_min_idx1 = numpy.argmin(obj_diff)
                obj_diff_min1     = obj_diff[obj_diff_min_idx1]
                G_min_idx1        = where1[ where2[ obj_diff_min_idx1 ] ]




        # find G_min_idx2 obj_diff_min2 and G_max22
        G_min_idx2    = -1
        G_max22       = -Infinity
        obj_diff_min2 = Infinity

        where1, = where(self._cond_not_upper_and_neg_Y)

        if len(where1):
            G_filt    =  G[where1]
            G_max22   = -G_filt.min()

            grad_diff = G_max1 - G_filt

            where2,       = where(grad_diff > 0)
            where1_where2 = where1[where2]

            a = Q_i[i] + QD[where1_where2] + 2. * Y[i] * Q_i[where1_where2]

            where3a, = where(a >  0)
            where3b, = where(a <= 0)

            obj_diff = empty(len(a),dtype=float)
            obj_diff[where3a] = numpy.negative( grad_diff[where2[where3a]] ** 2 ) / a[where3a]
            obj_diff[where3b] = numpy.negative( grad_diff[where2[where3b]] ** 2 ) / self._TAU

            if len(obj_diff):
                obj_diff_min_idx2 = numpy.argmin(obj_diff)
                obj_diff_min2     = obj_diff[obj_diff_min_idx2]
                G_min_idx2        = where1[ where2[ obj_diff_min_idx2 ] ]


        # determine mins and maxs
        if obj_diff_min1 <= obj_diff_min2:
            obj_diff_min  = obj_diff_min1
            G_min_idx     = G_min_idx1
        else:
            obj_diff_min  = obj_diff_min2
            G_min_idx     = G_min_idx2

        G_max2 = max(G_max21, G_max22)

        if G_max1+G_max2 < self._EPS: return -1,-1

        return G_max_idx,G_min_idx




    def _updateAlphaStatus(self,i):
        """ Updates the value of alpha_status array at index i.
            The alpha_status array holds the states of all multipliers.

            A multiplier a can be in one of three states:
              LOWER_BOUND :  if a == 0
              FREE        :  if 0 < a < C
              UPPER_BOUND :  if a == C
        """
        alpha_i      = self.module._alpha[i]
        alpha_status = self._alpha_status
        UPPER_BOUND  = self._UPPER_BOUND
        LOWER_BOUND  = self._LOWER_BOUND
        FREE         = self._FREE

        alpha_status_i_old  = alpha_status[i]
        if   alpha_i >= self._getC(i) : alpha_status[i] = UPPER_BOUND
        elif alpha_i <= 0             : alpha_status[i] = LOWER_BOUND
        else                          : alpha_status[i] = FREE
        alpha_status_i      = alpha_status[i]

        if alpha_status_i_old  != alpha_status_i:
            if   alpha_status_i_old == LOWER_BOUND : self._lb_num   -= 1
            elif alpha_status_i_old == FREE        : self._free_num -= 1
            elif alpha_status_i_old == UPPER_BOUND : self._ub_num   -= 1

            if   alpha_status_i     == LOWER_BOUND : self._lb_num   += 1
            elif alpha_status_i     == FREE        : self._free_num += 1
            elif alpha_status_i     == UPPER_BOUND : self._ub_num   += 1

            self._cond_not_lower[i]           = ( alpha_status_i != LOWER_BOUND          )
            self._cond_not_upper[i]           = ( alpha_status_i != UPPER_BOUND          )
            self._cond_not_upper_and_pos_Y[i] = ( self._cond_not_upper[i] & self._cond_pos_Y[i] )
            self._cond_not_lower_and_pos_Y[i] = ( self._cond_not_lower[i] & self._cond_pos_Y[i] )
            self._cond_not_upper_and_neg_Y[i] = ( self._cond_not_upper[i] & self._cond_neg_Y[i] )
            self._cond_not_lower_and_neg_Y[i] = ( self._cond_not_lower[i] & self._cond_neg_Y[i] )


    def _isUpperBound(self,i):
        """ Returns wheter alpha[i] is upper bound. """
        return self._alpha_status[i] == self._UPPER_BOUND

    def _isLowerBound(self,i):
        """ Returns wheter alpha[i] is lower bound. """
        return self._alpha_status[i] == self._LOWER_BOUND

    def _isFree(self,i):
        """ Returns wheter alpha[i] is free. """
        return self._alpha_status[i] == self._FREE

    def _getC(self,i):
        # todo: implement different C's for each class
#        return self.module._kernel._Y[i]
#        print self._C
#        print i
        return self._C[i]


    def calculateObjectiveValue(self):
        """ Calculates and returns the Value of the Objective Function:
                f(alpha) = 0.5 * alpha' * Q * alpha - sum(alpha)
        """
        v = 0
        alpha = self.module._alpha
        G     = self._G
        v = sum( alpha * ( G - 1 ) ) / 2
        return v



    def updateBeta(self):
        """ Writes the updated threshold value to the module. """
        self.module._beta = self.calculateBeta()

    def calculateBeta(self):
        """ Calculates and returns the threshold value beta.

            The threshold value is not trained directly by the training loop,
            but instead can be calculated for a given set of alpha values at any time.
            It definitly must be calculated at the end of the training process,
            after alpha has converged.

            The alpha values just define the orientation of the separating hyperplane
            in feature space. In order to move this hyperplane away from the point
            of origin, we need to add the threshold beta.

            See SVM.functionalMargin() for more information about beta.
        """
        Y = self.module._kernel._Y
        UPPER_BOUND  = self._UPPER_BOUND
        LOWER_BOUND  = self._LOWER_BOUND
        FREE         = self._FREE
        alpha_status = self._alpha_status
        actsz        = self._active_size


        active_alpha_status = alpha_status[0:actsz]
        where_f,  = where( active_alpha_status == FREE )

        YG       = Y[0:actsz] * self._G[0:actsz]
        nr_free  = len(    where_f  )
        sum_free = sum( YG[where_f] )

        if nr_free > 0:
            r =  sum_free   / nr_free
        else:
            where_ub_neg_or_lb_pos, = where( ( ( active_alpha_status == UPPER_BOUND ) & self._cond_neg_Y[0:actsz] ) | ( (active_alpha_status == LOWER_BOUND ) & self._cond_pos_Y[0:actsz] ) )
            where_ub_pos_or_lb_neg, = where( ( ( active_alpha_status == UPPER_BOUND ) & self._cond_pos_Y[0:actsz] ) | ( (active_alpha_status == LOWER_BOUND ) & self._cond_neg_Y[0:actsz] ) )
            ub = Infinity
            lb = -Infinity
            if len( where_ub_neg_or_lb_pos ): ub = min( YG[where_ub_neg_or_lb_pos] )
            if len( where_ub_pos_or_lb_neg ): lb = max( YG[where_ub_pos_or_lb_neg] )
            r = ( ub + lb ) / 2

        return r



    def calculateW(self):
        """ Calculates and returns w. This is only possible for kernels with an
            explicit feature function (named phi).

            w is the an orthogonal vector to the separating hyperplane.
            The (functional) distance of a point x to the decision boundary
            could be calculated by:   < w', phi(x) > - beta

            w can just be calculated, if the feature function phi(x) is explicitly
            defined through the kernel.
        """
        assert( self.module._kernel.phi != None )
        phi     = self.module._kernel.phi
        X       = self.module._kernel._X
        Y       = self.module._kernel._Y
        alpha   = self.module._alpha

        tmp = ((alpha * Y) * array( [phi(x) for x in X], dtype=float ).T).T
        w = numpy.sum( tmp, axis=0 )

        return w








