""" Implementation of all the benchmark functions used in the 2010 GECCO workshop BBOB
(Black-Box Optimization Benchmarking). 

Note: f_opt is fixed to 0 for all.

"""

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.rl.environments.functions.unimodal import * #@UnusedWildImport
from pybrain.rl.environments.functions.transformations import BBOBTransformationFunction
from pybrain.rl.environments.functions.multimodal import * #@UnusedWildImport



# --- separable ---

def bbob_f1(dim):
    return BBOBTransformationFunction(SphereFunction(dim))

def bbob_f2(dim):
    return BBOBTransformationFunction(ElliFunction(dim),
                                      oscillate=True)

def bbob_f3(dim):
    return BBOBTransformationFunction(RastriginFunction(dim), 
                                      oscillate=True,
                                      asymmetry=0.2)

def bbob_f4(dim):
    return BBOBTransformationFunction(BucheRastriginFunction(dim), 
                                      oscillate=True,
                                      penalized=100)   

def bbob_f5(dim):
    return BBOBTransformationFunction(BoundedLinear(dim), 
                                      translate=False)   


# --- moderate conditioning ---

def bbob_f6(dim):
    return BBOBTransformationFunction(AttractiveSectorFunction(dim), 
                                      rotate=True, 
                                      translate=False)

def bbob_f7(dim):
    return BBOBTransformationFunction(StepElliFunction(dim), 
                                      conditioning=10,
                                      penalized=1,
                                      rotate=True)

def bbob_f8(dim):
    return BBOBTransformationFunction(RosenbrockFunction(dim))

def bbob_f9(dim):
    return BBOBTransformationFunction(RosenbrockFunction(dim), 
                                      rotate=True)


# --- unimodal, high conditioning ---

def bbob_f10(dim):
    return BBOBTransformationFunction(ElliFunction(dim), 
                                      oscillate=True,
                                      rotate=True)

def bbob_f11(dim):
    return BBOBTransformationFunction(TabletFunction(dim), 
                                      oscillate=True,
                                      rotate=True)

def bbob_f12(dim):
    return BBOBTransformationFunction(CigarFunction(dim), 
                                      asymmetry=0.5,
                                      rotate=True)
    
def bbob_f13(dim):
    return BBOBTransformationFunction(SharpRFunctionBis(dim), 
                                      conditioning=10,
                                      rotate=True)

def bbob_f14(dim):
    return BBOBTransformationFunction(DiffPowFunction(dim, a=4), 
                                      rotate=True)


# --- multi-modal with global structure ---

def bbob_f15(dim):
    return BBOBTransformationFunction(RastriginFunction(dim), 
                                      conditioning=10,
                                      oscillate=True,
                                      asymmetry=0.2,
                                      rotate=True)

def bbob_f16(dim):
    return BBOBTransformationFunction(WeierstrassFunction(dim, kmax=11), 
                                      conditioning=0.01,
                                      oscillate=True,
                                      rotate=True)
    
    
def bbob_f17(dim):
    return BBOBTransformationFunction(SchaffersF7Function(dim), 
                                      conditioning=10,
                                      asymmetry=0.5,
                                      penalized=10,
                                      rotate=True)

def bbob_f18(dim):
    return BBOBTransformationFunction(SchaffersF7Function(dim), 
                                      conditioning=1000,
                                      asymmetry=0.5,
                                      penalized=10,
                                      rotate=True)

def bbob_f19(dim):
    return BBOBTransformationFunction(GriewankRosenbrockFunction(dim), 
                                      rotate=True)


# --- multi-modal with weak global structure ---    

def bbob_f20(dim):
    return BBOBTransformationFunction(Schwefel20Function(dim), 
                                      translate=False)
    
def bbob_f21(dim):
    return BBOBTransformationFunction(GallagherGauss101MeFunction(dim), 
                                      translate=False)
    
def bbob_f22(dim):
    return BBOBTransformationFunction(GallagherGauss21HiFunction(dim), 
                                      translate=False)

def bbob_f23(dim):
    return BBOBTransformationFunction(KatsuuraFunction(dim), 
                                      rotate=True,
                                      conditioning=100)

def bbob_f24(dim):
    return BBOBTransformationFunction(LunacekBiRastriginFunction(dim), 
                                      translate=False)
    

# all of them
bbob_collection = [#bbob_f1, bbob_f2, bbob_f3, #bbob_f4, 
                   #bbob_f5, 
                   #bbob_f6, bbob_f7, bbob_f8, bbob_f9, bbob_f10, 
                   #bbob_f11, bbob_f12, bbob_f13, bbob_f14, bbob_f15, 
                   #bbob_f16, 
                   #bbob_f17, 
                   #bbob_f18, 
                   #bbob_f19, bbob_f20, 
                   bbob_f21, bbob_f22, bbob_f23, bbob_f24]
