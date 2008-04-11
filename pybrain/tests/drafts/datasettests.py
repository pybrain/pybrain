from scipy import array
from pybrain.datasets import SequentialDataSet, SupervisedDataSet


def testSequential():
    ds = SequentialDataSet(1, 4)
    ds.addSample(array([1.34]), [[0.2, 0.4, 1.0, -0.3]])
    ds.resize(10)    
    ds.newSequence()
    ds.addSample(2, [[0.0, -5, -3, 1.1]])
    ds.newSequence()
    ds.addSample(0, [0.0, -5, -3, 1.1])

    ds.vectorFormat = '1d'
    ds.reset()

    for i in range(ds.getNumSequences()):
        print "sequence", i, ":", ds.getSequenceLength(i), ds.getSequence(i)
        

def testSupervised():
    i = array([[1,2,3],[4,5,6]])
    t = array([[2],[3]])
    tl = [2,3]
    ds = SupervisedDataSet(i,t)
    print str(ds)
    ds = SupervisedDataSet(i,tl)
    print str(ds)


if __name__ == "__main__":
    testSequential()
    testSupervised()