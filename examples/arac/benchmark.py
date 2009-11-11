__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'

try:
    import arac
except:
    raise Exception("This example requires the optional dependency 'arac' (for fast networks) to run.")

import time

import scipy

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import SigmoidLayer


def timeit(net, outer=1000, inner=1):
    inpt = scipy.empty(net.indim)
    err = scipy.empty(net.outdim)
    start = time.time()
    for _ in xrange(outer):
        for _ in xrange(inner):
            net.activate(inpt)
        for _ in xrange(inner):
            net.backActivate(err)
    net.reset()
    net.resetDerivatives()
    stop = time.time()
    return stop - start

configs = [
  ([2, 3, 1], {'hiddenclass': SigmoidLayer}, 10000, 1),
  ([20, 30, 10], {'hiddenclass': SigmoidLayer}, 5, 100),
  ([20, 30, 10], {'hiddenclass': SigmoidLayer}, 1000, 1),
  ([784, 500, 10], {'hiddenclass': SigmoidLayer}, 200, 1),
  ([784, 500, 2000, 2000, 10], {'hiddenclass': SigmoidLayer}, 100, 1),
]



for args, kwargs, outer, inner in configs:
    # normal networks
    if inner != 1:
        kwargs['recurrent'] = True
    print "=" * 20
    print args
    print kwargs
    net = buildNetwork(*args, **kwargs)
    normaltime = timeit(net, outer, inner)
    kwargs['fast'] = True
    fnet = buildNetwork(*args, **kwargs)
    fasttime = timeit(fnet, outer, inner)
    print "Normal time: %.5f" % (normaltime / outer / inner)
    print "Fast Time:  %.5f" % (fasttime / outer / inner)
    print "Speedup: %.2f" % (normaltime / fasttime)

