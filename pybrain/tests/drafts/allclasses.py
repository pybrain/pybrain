__author__ = 'Tom Schaul, tom@idsia.ch'

import pybrain
import inspect


def printAllPybrainImports():
    tmp = sorted(vars(pybrain).items())
    print '  Imported classes:'
    for k, v in tmp:
        if inspect.isclass(v):
            print k
    print 
    
    print '  Imported functions:'
    for k, v in tmp:
        if inspect.isfunction(v):
            print k


        
if __name__ == '__main__':
    printAllPybrainImports()