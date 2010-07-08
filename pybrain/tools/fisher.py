__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.utilities import blockCombine
from scipy import mat, dot, outer
from scipy.linalg import inv, cholesky



def calcFisherInformation(sigma, invSigma=None, factorSigma=None):
    """ Compute the exact Fisher Information Matrix of a Gaussian distribution,
    given its covariance matrix.
    Returns a list of the diagonal blocks. """
    if invSigma == None:
        invSigma = inv(sigma)
    if factorSigma == None:
        factorSigma = cholesky(sigma)
    dim = sigma.shape[0]
    fim = [invSigma]
    for k in range(dim):
        D = invSigma[k:, k:].copy()
        D[0, 0] += factorSigma[k, k] ** -2
        fim.append(D)
    return fim



def calcInvFisher(sigma, invSigma=None, factorSigma=None):
    """ Efficiently compute the exact inverse of the FIM of a Gaussian.
    Returns a list of the diagonal blocks. """
    if invSigma == None:
        invSigma = inv(sigma)
    if factorSigma == None:
        factorSigma = cholesky(sigma)
    dim = sigma.shape[0]

    invF = [mat(1 / (invSigma[-1, -1] + factorSigma[-1, -1] ** -2))]
    invD = 1 / invSigma[-1, -1]
    for k in reversed(range(dim - 1)):
        v = invSigma[k + 1:, k]
        w = invSigma[k, k]
        wr = w + factorSigma[k, k] ** -2
        u = dot(invD, v)
        s = dot(v, u)
        q = 1 / (w - s)
        qr = 1 / (wr - s)
        t = -(1 + q * s) / w
        tr = -(1 + qr * s) / wr
        invF.append(blockCombine([[qr, tr * u], [mat(tr * u).T, invD + qr * outer(u, u)]]))
        invD = blockCombine([[q , t * u], [mat(t * u).T, invD + q * outer(u, u)]])

    invF.append(sigma)
    invF.reverse()
    return invF

