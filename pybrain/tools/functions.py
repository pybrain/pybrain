__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import array, exp, tanh, clip, log, dot, sqrt, power, pi, tan, diag, rand, real_if_close
from scipy.linalg import inv, det, svd, logm, expm2


def semilinear(x):
    """ This function ensures that the values of the array are always positive. It is
        x+1 for x=>0 and exp(x) for x<0. """
    try:
        # assume x is a numpy array
        shape = x.shape
        x.flatten()
        x = x.tolist()
    except AttributeError:
        # no, it wasn't: build shape from length of list
        shape = (1, len(x))
    def f(val):
        if val < 0:
            # exponential function for x<0
            return safeExp(val)
        else:
            # linear function for x>=0
            return val + 1.0
    return array(map(f, x)).reshape(shape)


def semilinearPrime(x):
    """ This function is the first derivative of the semilinear function (above).
        It is needed for the backward pass of the module. """
    try:
        # assume x is a numpy array
        shape = x.shape
        x.flatten()
        x = x.tolist()
    except AttributeError:
        # no, it wasn't: build shape from length of list
        shape = (1, len(x))
    def f(val):
        if val < 0:
            # exponential function for x<0
            return safeExp(val)
        else:
            # linear function for x>=0
            return 1.0
    return array(map(f, x)).reshape(shape)


def safeExp(x):
    """ Bounded range for the exponential function (won't produce inf or NaN). """
    return exp(clip(x, -500, 500))


def sigmoid(x):
    """ Logistic sigmoid function. """
    return 1. / (1. + safeExp(-x))


def sigmoidPrime(x):
    """ Derivative of logistic sigmoid. """
    tmp = sigmoid(x)
    return tmp * (1 - tmp)


def tanhPrime(x):
    """ Derivative of tanh. """
    tmp = tanh(x)
    return 1 - tmp * tmp


def ranking(R):
    """ Produces a linear ranking of the values in R. """
    l = sorted(list(enumerate(R)), cmp=lambda a, b: cmp(a[1], b[1]))
    l = sorted(list(enumerate(l)), cmp=lambda a, b: cmp(a[1], b[1]))
    return array(map(lambda kv: kv[0], l))


def expln(x):
    """ This continuous function ensures that the values of the array are always positive.
        It is ln(x+1)+1 for x >= 0 and exp(x) for x < 0. """
    def f(val):
        if val < 0:
            # exponential function for x < 0
            return exp(val)
        else:
            # natural log function for x >= 0
            return log(val + 1.0) + 1
    try:
        result = array(map(f, x))
    except TypeError:
        result = array(f(x))

    return result


def explnPrime(x):
    """ This function is the first derivative of the expln function (above).
        It is needed for the backward pass of the module. """
    def f(val):
        if val < 0:
            # exponential function for x<0
            return exp(val)
        else:
            # linear function for x>=0
            return 1.0 / (val + 1.0)
    try:
        result = array(map(f, x))
    except TypeError:
        result = array(f(x))

    return result


def multivariateNormalPdf(z, x, sigma):
    """ The pdf of a multivariate normal distribution (not in scipy).
    The sample z and the mean x should be 1-dim-arrays, and sigma a square 2-dim-array. """
    assert len(z.shape) == 1 and len(x.shape) == 1 and len(x) == len(z) and sigma.shape == (len(x), len(z))
    tmp = -0.5 * dot(dot((z - x), inv(sigma)), (z - x))
    res = (1. / power(2.0 * pi, len(z) / 2.)) * (1. / sqrt(det(sigma))) * exp(tmp)
    return res


def simpleMultivariateNormalPdf(z, detFactorSigma):
    """ Assuming z has been transformed to a mean of zero and an identity matrix of covariances.
    Needs to provide the determinant of the factorized (real) covariance matrix. """
    dim = len(z)
    return exp(-0.5 * dot(z, z)) / (power(2.0 * pi, dim / 2.) * detFactorSigma)


def multivariateCauchy(mu, sigma, onlyDiagonal=True):
    """ Generates a sample according to a given multivariate Cauchy distribution. """
    if not onlyDiagonal:
        u, s, d = svd(sigma)
        coeffs = sqrt(s)
    else:
        coeffs = diag(sigma)
    r = rand(len(mu))
    res = coeffs * tan(pi * (r - 0.5))
    if not onlyDiagonal:
        res = dot(d, dot(res, u))
    return res + mu


def approxChiFunction(dim):
    """ Returns Chi (expectation of the length of a normal random vector)
    approximation according to: Ostermeier 1997. """
    dim = float(dim)
    return sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))


def sqrtm(M):
    """ Returns the symmetric semi-definite positive square root of a matrix. """
    r = real_if_close(expm2(0.5 * logm(M)), 1e-8)
    return (r + r.T) / 2

