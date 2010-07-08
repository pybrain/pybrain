#######################################################################
# k-means++
#
# this is a k-means clustering algorithm that selects its
# initial cluster centers in a smart way to speed up convergence.
# see: Arthur, D. and Vassilvitskii, S. "k-means++: the advantages
# of careful seeding". ACM-SIAM symposium on Discrete algorithms. 2007
#
# Implementation from Yong Sun's website
# http://blogs.sun.com/yongsun/entry/k_means_and_k_means
#######################################################################

__author__ = "Thomas Rueckstiess, ruecksti@in.tum.de"

from scipy.cluster.vq import kmeans2
from scipy import random, array
from scipy.linalg import norm

def kinit(X, k):
    'init k seeds according to kmeans++'
    n = X.shape[0]

    'choose the 1st seed randomly, and store D(x)^2 in D[]'
    centers = [X[random.randint(n)]]
    D = [norm(x - centers[0]) ** 2 for x in X]

    for _ in range(k - 1):
        bestDsum = bestIdx = -1

        for i in range(n):
            'Dsum = sum_{x in X} min(D(x)^2,||x-xi||^2)'
            Dsum = reduce(lambda x, y:x + y,
                          (min(D[j], norm(X[j] - X[i]) ** 2) for j in xrange(n)))

            if bestDsum < 0 or Dsum < bestDsum:
                bestDsum, bestIdx = Dsum, i

        centers.append (X[bestIdx])
        D = [min(D[i], norm(X[i] - X[bestIdx]) ** 2) for i in xrange(n)]

    return array (centers)

def kmeanspp(Y, k):
    return kmeans2(Y, kinit(Y, k), minit='points')

