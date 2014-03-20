__author__ = 'Tom Schaul, tom@idsia.ch'

""" 
Doing RL when an environment model (transition matrices and rewards) are available,
and the states are observed by a feature vector for each state:

 - a feature map (fMap) is a 2D array of features, one row per state.

(otherwise same representation as for in policyiteration.py)

Here we provide some algorithms when the values are estimated linearly from the features.

"""

# TODO: optimize LSTDQ by vectorization

import random
from scipy import ravel, zeros, outer, dot, tile, transpose, tensordot
from scipy.linalg import lstsq
from numpy.matlib import repmat


from pybrain.rl.learners.modelbased.policyiteration import randomPolicy, greedyQPolicy, collapsedTransitions, policyIteration


def trueFeatureStats(T, R, fMap, discountFactor, stateProp=1, MAT_LIMIT=1e8):
    """ Gather the statistics needed for LSTD,
    assuming infinite data (true probabilities).
    Option: if stateProp is  < 1, then only a proportion of all 
    states will be seen as starting state for transitions """
    dim = len(fMap)
    numStates = len(T)
    statMatrix = zeros((dim, dim))
    statResidual = zeros(dim)
    ss = range(numStates)
    repVersion = False
    
    if stateProp < 1:
        ss = random.sample(ss, int(numStates * stateProp))
    elif dim * numStates**2 < MAT_LIMIT:
        repVersion = True
    
    # two variants, depending on how large we can afford our matrices to become.        
    if repVersion:    
        tmp1 = tile(fMap, (numStates,1,1))
        tmp2 = transpose(tmp1, (2,1,0))
        tmp3 = tmp2 - discountFactor * tmp1            
        tmp4 = tile(T, (dim,1,1))
        tmp4 *= transpose(tmp1, (1,2,0))
        statMatrix = tensordot(tmp3, tmp4, axes=[[0,2], [1,2]]).T
        statResidual = dot(R, dot(fMap, T).T)
    else:
        for sto in ss:
            tmp = fMap - discountFactor * repmat(fMap[:, sto], numStates, 1).T
            tmp2 = fMap * repmat(T[:, sto], dim, 1)
            statMatrix += dot(tmp2, tmp.T)             
            statResidual += R[sto] * dot(fMap, T[:, sto])
    return statMatrix, statResidual


def LSTD_values(T, R, fMap, discountFactor, **kwargs):
    """ Least-squares temporal difference algorithm. """
    statMatrix, statResidual = trueFeatureStats(T, R, fMap, discountFactor,**kwargs)
    weights = lstsq(statMatrix, statResidual)[0]
    return dot(weights, fMap)


def LSTD_Qvalues(Ts, policy, R, fMap, discountFactor):
    """ LSTDQ is like LSTD, but with features replicated 
    once for each possible action.
    
    Returns Q-values in a 2D array. """
    numA = len(Ts)
    dim = len(Ts[0])
    numF = len(fMap)
    fMapRep = zeros((numF * numA, dim * numA))
    for a in range(numA):
        fMapRep[numF * a:numF * (a + 1), dim * a:dim * (a + 1)] = fMap

    statMatrix = zeros((numF * numA, numF * numA))
    statResidual = zeros(numF * numA)
    for sto in range(dim):
        r = R[sto]
        fto = zeros(numF * numA)
        for nextA in range(numA):
            fto += fMapRep[:, sto + nextA * dim] * policy[sto][nextA]
        for sfrom in range(dim):
            for a in range(numA):
                ffrom = fMapRep[:, sfrom + a * dim]
                prob = Ts[a][sfrom, sto]
                statMatrix += outer(ffrom, ffrom - discountFactor * fto) * prob
                statResidual += ffrom * r * prob

    Qs = zeros((dim, numA))
    w = lstsq(statMatrix, statResidual)[0]
    for a in range(numA):
        Qs[:,a] = dot(w[numF*a:numF*(a+1)], fMap)
    return Qs


def LSPI_policy(fMap, Ts, R, discountFactor, initpolicy=None, maxIters=20):
    """ LSPI is like policy iteration, but Q-values are estimated based 
    on the feature map. 
    Returns the best policy found. """
    if initpolicy is None:
        policy, _ = randomPolicy(Ts) 
    else:
        policy = initpolicy
    
    while maxIters > 0:
        Qs = LSTD_Qvalues(Ts, policy, R, fMap, discountFactor)
        newpolicy = greedyQPolicy(Qs)
        if sum(ravel(abs(newpolicy - policy))) < 1e-3:
            return policy, collapsedTransitions(Ts, policy)
        policy = newpolicy
        maxIters -= 1
    return policy, collapsedTransitions(Ts, policy)


def LSTD_PI_policy(fMap, Ts, R, discountFactor, initpolicy=None, maxIters=20):
    """ Alternative version of LSPI using value functions
    instead of state-action values as intermediate.
    """
    def veval(T):
        return LSTD_values(T, R, fMap, discountFactor)
    return policyIteration(Ts, R, discountFactor, VEvaluator=veval, 
                           initpolicy=initpolicy, maxIters=maxIters)
