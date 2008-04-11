from pybrain.datasets import SequentialDataSet
from profile import run

def main():
    ds = SequentialDataSet(3, 1)
    for i in range(100):
        ds.newSequence()
        for j in range(100):
            ds.addSample([1, 2, 3], [4])
         
run('main()')




