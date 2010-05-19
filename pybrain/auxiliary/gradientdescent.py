__author__ = ('Thomas Rueckstiess, ruecksti@in.tum.de'
              'Justin Bayer, bayer.justin@googlemail.com')


from scipy import zeros, asarray, sign, array, cov, dot, clip, ndarray
from scipy.linalg import inv


class GradientDescent(object):

    def __init__(self):
        """ initialize algorithms with standard parameters (typical values given in parentheses)"""

        # --- BackProp parameters ---
        # learning rate (0.1-0.001, down to 1e-7 for RNNs)
        self.alpha = 0.1

        # alpha decay (0.999; 1.0 = disabled)
        self.alphadecay = 1.0

        # momentum parameters (0.1 or 0.9)
        self.momentum = 0.0
        self.momentumvector = None

        # --- RProp parameters ---
        self.rprop = False
        # maximum step width (1 - 20)
        self.deltamax = 5.0
        # minimum step width (0.01 - 1e-6)
        self.deltamin = 0.01
        # the remaining parameters do not normally need to be changed
        self.deltanull = 0.1
        self.etaplus = 1.2
        self.etaminus = 0.5
        self.lastgradient = None

    def init(self, values):
        """ call this to initialize data structures *after* algorithm to use
        has been selected

        :arg values: the list (or array) of parameters to perform gradient descent on
                       (will be copied, original not modified)
        """
        assert isinstance(values, ndarray)
        self.values = values.copy()
        if self.rprop:
            self.lastgradient = zeros(len(values), dtype='float64')
            self.rprop_theta = self.lastgradient + self.deltanull
            self.momentumvector = None
        else:
            self.lastgradient = None
            self.momentumvector = zeros(len(values))

    def __call__(self, gradient, error=None):
        """ calculates parameter change based on given gradient and returns updated parameters """
        # check if gradient has correct dimensionality, then make array """
        assert len(gradient) == len(self.values)
        gradient_arr = asarray(gradient)

        if self.rprop:
            rprop_theta = self.rprop_theta

            # update parameters
            self.values += sign(gradient_arr) * rprop_theta

            # update rprop meta parameters
            dirSwitch = self.lastgradient * gradient_arr
            rprop_theta[dirSwitch > 0] *= self.etaplus
            idx =  dirSwitch < 0
            rprop_theta[idx] *= self.etaminus
            gradient_arr[idx] = 0

            # upper and lower bound for both matrices
            rprop_theta = rprop_theta.clip(min=self.deltamin, max=self.deltamax)

            # save current gradients to compare with in next time step
            self.lastgradient = gradient_arr.copy()

            self.rprop_theta = rprop_theta

        else:
            # update momentum vector (momentum = 0 clears it)
            self.momentumvector *= self.momentum

            # update parameters (including momentum)
            self.momentumvector += self.alpha * gradient_arr
            self.alpha *= self.alphadecay

            # update parameters
            self.values += self.momentumvector

        return self.values

    descent = __call__


class NaturalGradient(object):

    def __init__(self, samplesize):
        # Counter after how many samples a new gradient estimate will be
        # returned.
        self.samplesize = samplesize
        # Samples of the gradient are held in this datastructure.
        self.samples = []

    def init(self, values):
        self.values = values.copy()

    def __call__(self, gradient, error=None):
        # Append a copy to make sure this one is not changed after by the
        # client.
        self.samples.append(array(gradient))
        # Return None if no new estimate is being given.
        if len(self.samples) < self.samplesize:
            return None
        # After all the samples have been put into a single array, we can
        # delete them.
        gradientarray = array(self.samples).T
        inv_covar = inv(cov(gradientarray))
        self.values += dot(inv_covar, gradientarray.sum(axis=1))
        return self.values


class IRpropPlus(object):

    def __init__(self, upfactor=1.1, downfactor=0.9, bound=0.5):
        self.upfactor = upfactor
        self.downfactor = downfactor
        if not bound > 0:
            raise ValueError("bound greater than 0 needed.")

    def init(self, values):
        self.values = values.copy()
        self.prev_values = values.copy()
        self.more_prev_values = values.copy()
        self.previous_gradient = zeros(values.shape)
        self.step = zeros(values.shape)
        self.previous_error = float("-inf")

    def __call__(self, gradient, error):
        products = self.previous_gradient * gradient
        signs = sign(gradient)

        # For positive gradient parts.
        positive = (products > 0).astype('int8')
        pos_step = self.step * self.upfactor * positive
        clip(pos_step, -self.bound, self.bound)
        pos_update = self.values - signs * pos_step

        # For negative gradient parts.
        negative = (products < 0).astype('int8')
        neg_step = self.step * self.downfactor * negative
        clip(neg_step, -self.bound, self.bound)
        if error <= self.previous_error:
            # If the error has decreased, do nothing.
            neg_update = zeros(gradient.shape)
        else:
            # If it has increased, move back 2 steps.
            neg_update = self.more_prev_values
        # Set all negative gradients to zero for the next step.
        gradient *= positive

        # Bookkeeping.
        self.previous_gradient = gradient
        self.more_prev_values = self.prev_values
        self.prev_values = self.values.copy()
        self.previous_error = error

        # Updates.
        self.step[:] = pos_step + neg_step
        self.values[:] = positive * pos_update + negative * neg_update

        return self.values
