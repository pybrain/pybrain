__author__ = 'Tom Schaul, tom@idsia.ch'

from random import uniform
from scipy import array
from scipy.stats.distributions import norm


def importanceMixing(oldpoints, oldpdf, newpdf, newdistr, forcedRefresh = 0.01):
    """ Implements importance mixing. Given a set of points, an old and a new pdf-function for them
    and a generator function for new points, it produces a list of indices of the old points to be reused and a list of new points.
    Parameter (optional): forced refresh rate.
    """
    reuseindices = []
    batch = len(oldpoints)
    for i, sample in enumerate(oldpoints):
        r = uniform(0, 1)
        if r < (1-forcedRefresh) * newpdf(sample) / oldpdf(sample):
            reuseindices.append(i)
        # never use only old samples
        if batch - len(reuseindices) <= max(1, batch * forcedRefresh):
            break
    newpoints = []
    # add the remaining ones
    while len(reuseindices)+len(newpoints) < batch:
        r = uniform(0, 1)
        sample = newdistr()
        if r < forcedRefresh:
            newpoints.append(sample)
        else:
            if r < 1 - oldpdf(sample)/newpdf(sample):
                newpoints.append(sample)
    return reuseindices, newpoints


def testImportanceMixing(popsize = 5000, forcedRefresh = 0.0):
    import pylab
    distr1 = norm()
    distr2 = norm(loc = 1.5)
    p1 = distr1.rvs(popsize)
    inds, np = importanceMixing(p1, distr1.pdf, distr2.pdf, lambda: distr2.rvs()[0], forcedRefresh)
    reuse = [p1[i] for i in inds]
    p2 = reuse + np
    p2b = distr2.rvs(popsize)
    pylab.hist(array([p2, p2b]).T,
               20, normed=1, histtype='bar')
    pylab.show()


if __name__ == '__main__':
    testImportanceMixing()