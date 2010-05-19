# -*- coding: utf-8 -*-


"""Module that contains functionality for calculating the principal components
of a dataset."""


__author__ = 'Justin S Bayer, bayerj@in.tum.de'


from scipy import asmatrix, cov
from scipy.linalg import inv, eig
from numpy.random import standard_normal


def reduceDim(data, dim, func='pca'):
    """Reduce the dimension of datapoints to dim via principal component
    analysis.

    A matrix of shape (n, d) specifies n points of dimension d.
    """
    try:
        pcaFunc = globals()[func]
    except KeyError:
        raise ValueError('Unknown function to calc principal components')
    pc = pcaFunc(data, dim)
    return (pc * asmatrix(makeCentered(data)).T).T


def makeCentered(data):
    """Move the mean of the data matrix into the origin.

    Rows are perceived as datapoints.
    """
    return data - data.mean(axis=0)


def pca(data, dim):
    """ Return the first dim principal components as colums of a matrix.

    Every row of the matrix resembles a point in the data space.
    """

    assert dim <= data.shape[1], \
        "dim must be less or equal than the original dimension"

    # We have to make a copy of the original data and substract the mean
    # of every entry
    data = makeCentered(data)
    cm = cov(data.T)

    # OPT only calculate the dim first eigenvectors here
    # The following calculation may seem a bit "weird" but also correct to me.
    # The eigenvectors with the dim highest eigenvalues have to be selected
    # We keep track of the indexes via enumerate to restore the right ordering
    # later.
    eigval, eigvec = eig(cm)
    eigval = [(val, ind) for ind, val  in enumerate(eigval)]
    eigval.sort()
    eigval[:-dim] = []  # remove all but the highest dim elements

    # now we have to bring them back in the right order
    eig_indexes = [(ind, val) for val, ind in eigval]
    eig_indexes.sort(reverse=True)
    eig_indexes = [ind for ind, val in eig_indexes]

    return eigvec.take(eig_indexes, 1).T


def pPca(data, dim):
    """Return a matrix which contains the first `dim` dimensions principal
    components of data.

    data is a matrix which's rows correspond to datapoints. Implementation of
    the 'probabilistic PCA' algorithm.
    """
    num = data.shape[1]
    data = asmatrix(makeCentered(data))
    # Pick a random reduction
    W = asmatrix(standard_normal((num, dim)))
    # Save for convergence check
    W_ = W[:]
    while True:
        E = inv(W.T * W) * W.T * data.T
        W, W_ = data.T * E.T * inv(E * E.T), W
        if abs(W - W_).max() < 0.001:
            break
    return W.T
