""" Implementation of all the benchmark functions used in the 2010 GECCO workshop BBOB
(Black-Box Optimization Benchmarking). 

Note: f_opt is fixed to 0 for all.

"""
from pybrain.rl.environments.functions.unimodal import SphereFunction,\
    ElliFunction
from pybrain.rl.environments.functions.transformations import TranslateFunction,\
    OscillatingFunction, AsymmetrizedFunction
from pybrain.rl.environments.functions.multimodal import RastriginFunction

__author__ = 'Tom Schaul, tom@idsia.ch'


def bbob_f1(dim):
    return TranslateFunction(SphereFunction(dim))

def bbob_f2(dim):
    return OscillatingFunction(TranslateFunction(ElliFunction(dim)))

def bbob_f3(dim):
    return AsymmetrizedFunction(OscillatingFunction(TranslateFunction(RastriginFunction(dim))), 0.2)

def bbob_f4(dim):
    return TranslateFunction(SphereFunction(dim))

def bbob_f5(dim):
    return TranslateFunction(SphereFunction(dim))