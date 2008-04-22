from pybrain import LSTMLayer, LinearLayer, buildNetwork
from pybrain.datasets import SequentialDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tests.helpers import sortedProfiling

from scipy import randn

seqlen = 10
seqs = 1

# TODO: bugs with more lstm cells !?

n = buildNetwork(10, 5000, 10)#, hiddenclass = LSTMLayer)

ds = SequentialDataSet(n.indim, n.outdim)
for dummy in range(seqs):
    ds.newSequence()
    for dummy in range(seqlen):
        ds.addSample(randn(n.indim), randn(n.outdim))
t = BackpropTrainer(n, ds)

print 'weights:', n.paramdim, '  seqlen:', seqlen, '  seqs:', seqs

sortedProfiling('t.trainEpochs(10)')
