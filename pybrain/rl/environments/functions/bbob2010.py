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
bbob_collection = [bbob_f1, bbob_f2, bbob_f3, bbob_f4, 
                   bbob_f5, 
                   bbob_f6, bbob_f7, bbob_f8, bbob_f9, bbob_f10, 
                   bbob_f11, bbob_f12, bbob_f13, bbob_f14, bbob_f15, 
                   bbob_f16, 
                   bbob_f17, 
                   bbob_f18, 
                   bbob_f19, bbob_f20, 
                   bbob_f21, bbob_f22, bbob_f23, bbob_f24]

#moderate noise
def bbob_f101(dim):
    return BBOBTransformationFunction(SphereFunction(dim),
                                      gnoise=0.01,
                                      penalized=1)
def bbob_f102(dim):
    return BBOBTransformationFunction(SphereFunction(dim),
                                      unoise=0.01,
                                      penalized=1)  
def bbob_f103(dim):
    return BBOBTransformationFunction(SphereFunction(dim),
                                      cnoise=(0.01,0.05),
                                      penalized=1)
    
def bbob_f104(dim):
    return BBOBTransformationFunction(RosenbrockFunction(dim),
                                      gnoise=0.01,
                                      penalized=1)  
def bbob_f105(dim):
    return BBOBTransformationFunction(RosenbrockFunction(dim),
                                      unoise=0.01,
                                      penalized=1)   
def bbob_f106(dim):
    return BBOBTransformationFunction(RosenbrockFunction(dim),
                                      cnoise=(0.01,0.05),
                                      penalized=1)
    
# severe noise
def bbob_f107(dim):
    return BBOBTransformationFunction(SphereFunction(dim),
                                      gnoise=1,
                                      penalized=1)    
def bbob_f108(dim):
    return BBOBTransformationFunction(SphereFunction(dim),
                                      unoise=1,
                                      penalized=1)    
def bbob_f109(dim):
    return BBOBTransformationFunction(SphereFunction(dim),
                                      cnoise=(1,0.2),
                                      penalized=1)
    
def bbob_f110(dim):
    return BBOBTransformationFunction(RosenbrockFunction(dim),
                                      gnoise=1,
                                      penalized=1)    
def bbob_f111(dim):
    return BBOBTransformationFunction(RosenbrockFunction(dim),
                                      unoise=1,
                                      penalized=1)    
def bbob_f112(dim):
    return BBOBTransformationFunction(RosenbrockFunction(dim),
                                      cnoise=(1,0.2),
                                      penalized=1)
    
def bbob_f113(dim):
    return BBOBTransformationFunction(StepElliFunction(dim), 
                                      conditioning=10,
                                      penalized=1,
                                      rotate=True,
                                      gnoise=1)        
def bbob_f114(dim):
    return BBOBTransformationFunction(StepElliFunction(dim), 
                                      conditioning=10,
                                      penalized=1,
                                      rotate=True,
                                      unoise=1)        
def bbob_f115(dim):
    return BBOBTransformationFunction(StepElliFunction(dim), 
                                      conditioning=10,
                                      penalized=1,
                                      rotate=True,
                                      cnoise=(1,0.2)) 
    
def bbob_f116(dim):
    return BBOBTransformationFunction(ElliFunction(dim, a=100),                                       
                                      oscillate=True, 
                                      penalized=1,                                   
                                      rotate=True,
                                      gnoise=1)        
def bbob_f117(dim):
    return BBOBTransformationFunction(ElliFunction(dim, a=100),                                       
                                      oscillate=True, 
                                      penalized=1,
                                      rotate=True,
                                      unoise=1)        
def bbob_f118(dim):
    return BBOBTransformationFunction(ElliFunction(dim, a=100),                                       
                                      oscillate=True, 
                                      penalized=1,
                                      rotate=True,
                                      cnoise=(1,0.2))   
    
def bbob_f119(dim):
    return BBOBTransformationFunction(DiffPowFunction(dim),  
                                      penalized=1,                                   
                                      rotate=True,
                                      gnoise=1)        
def bbob_f120(dim):
    return BBOBTransformationFunction(DiffPowFunction(dim),  
                                      penalized=1,
                                      rotate=True,
                                      unoise=1)        
def bbob_f121(dim):
    return BBOBTransformationFunction(DiffPowFunction(dim),  
                                      penalized=1,
                                      rotate=True,
                                      cnoise=(1,0.2))    


# multi-modal with severe noise
def bbob_f122(dim):
    return BBOBTransformationFunction(SchaffersF7Function(dim), 
                                      conditioning=10,
                                      asymmetry=0.5,
                                      penalized=1,                                   
                                      rotate=True,
                                      gnoise=1)        
def bbob_f123(dim):
    return BBOBTransformationFunction(SchaffersF7Function(dim), 
                                      conditioning=10,
                                      asymmetry=0.5,
                                      penalized=1,
                                      rotate=True,
                                      unoise=1)        
def bbob_f124(dim):
    return BBOBTransformationFunction(SchaffersF7Function(dim), 
                                      conditioning=10,
                                      asymmetry=0.5,
                                      penalized=1,
                                      rotate=True,
                                      cnoise=(1,0.2)) 
    
def bbob_f125(dim):
    return BBOBTransformationFunction(GriewankRosenbrockFunction(dim),  
                                      penalized=1,                                   
                                      rotate=True,
                                      gnoise=1)        
def bbob_f126(dim):
    return BBOBTransformationFunction(GriewankRosenbrockFunction(dim),  
                                      penalized=1,
                                      rotate=True,
                                      unoise=1)        
def bbob_f127(dim):
    return BBOBTransformationFunction(GriewankRosenbrockFunction(dim),  
                                      penalized=1,
                                      rotate=True,
                                      cnoise=(1,0.2)) 
    
def bbob_f128(dim):
    return BBOBTransformationFunction(GallagherGauss101MeFunction(dim), 
                                      translate=False,
                                      penalized=1,                                   
                                      gnoise=1)        
def bbob_f129(dim):
    return BBOBTransformationFunction(GallagherGauss101MeFunction(dim), 
                                      translate=False,
                                      penalized=1,
                                      unoise=1)        
def bbob_f130(dim):
    return BBOBTransformationFunction(GallagherGauss101MeFunction(dim), 
                                      translate=False,
                                      penalized=1,
                                      cnoise=(1,0.2)) 
    
    

bbob_noise_collection = [bbob_f101, bbob_f102, bbob_f103, 
                         bbob_f104, bbob_f105, bbob_f106, 
                         bbob_f107, bbob_f108, bbob_f109, 
                         bbob_f110, bbob_f111, bbob_f112, 
                         bbob_f113, bbob_f114, bbob_f115, 
                         bbob_f116, bbob_f117, bbob_f118, 
                         bbob_f119, bbob_f120, bbob_f121, 
                         bbob_f122, bbob_f123, bbob_f124,
                         bbob_f125, bbob_f126, bbob_f127, 
                         bbob_f128, bbob_f129, bbob_f130
                         ]
