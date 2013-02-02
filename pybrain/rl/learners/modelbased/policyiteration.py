__author__ = 'Tom Schaul, tom@idsia.ch'

""" 
Doing RL when an environment model (transition matrices and rewards) are available.


Representation:
 - a policy is a 2D-array of probabilities, 
    one row per state (summing to 1), one column per action.
 
 - a transition matrix (T) maps from originating states to destination states 
    (probabilities in each row sum to 1).
 
 - a reward vector (R) maps each state to the reward value obtained when entering (or staying in) a state.
 
 - a feature map (fMap) is a 2D array of features, one row per state.
 
 - a task model is defined by a list of transition matrices (Ts), one per action, a 
    reward vector R, a discountFactor
    
Note: a task model combined with a policy is again a transition matrix ("collapsed" dynamics).
        
 - a value function (V) is a vector of expected discounted rewards (one per state).
 
 - a set of state-action values (Qs) is a 2D array, one row per action.
 
"""

# TODO: we may use an alternative, more efficient representation if all actions are deterministic 
# TODO: it may be worth considering a sparse representation of T matrices.
# TODO: optimize some of this code with vectorization
    

from scipy import dot, zeros, zeros_like, ones, mean, array, ravel, rand
from numpy.matlib import repmat

from pybrain.utilities import all_argmax


def trueValues(T, R, discountFactor):
    """ Compute the true discounted value function for each state,
    given a policy (encoded as collapsed transition matrix). """
    assert discountFactor < 1
    distr = T.copy()
    res = dot(T, R)
    for i in range(1, int(10 / (1. - discountFactor))):
        distr = dot(distr, T)
        res += (discountFactor ** i) * dot(distr, R)
    return res


def trueQValues(Ts, R, discountFactor, policy):
    """ The true Q-values, given a model and a policy. """
    T = collapsedTransitions(Ts, policy)
    V = trueValues(T, R, discountFactor)
    Vnext = V*discountFactor+R
    numA = len(Ts)
    dim = len(R)
    Qs = zeros((dim, numA))
    for si in range(dim):
        for a in range(numA):
            Qs[si, a] = dot(Ts[a][si], Vnext)   
    return Qs


def collapsedTransitions(Ts, policy):
    """ Collapses a list of transition matrices (one per action) and a list 
        of action probability vectors into a single transition matrix."""
    res = zeros_like(Ts[0])
    dim = len(Ts[0])
    for ai, ap in enumerate(policy.T):
        res += Ts[ai] * repmat(ap, dim, 1).T
    return res


def greedyPolicy(Ts, R, discountFactor, V):
    """ Find the greedy policy, (soft tie-breaking)
    given a value function and full transition model. """
    dim = len(V)
    numA = len(Ts)
    Vnext = V*discountFactor+R
    policy = zeros((dim, numA))
    for si in range(dim):
        actions = all_argmax([dot(T[si, :], Vnext) for T in Ts])
        for a in actions:
            policy[si, a] = 1. / len(actions)        
    return policy, collapsedTransitions(Ts, policy)    


def greedyQPolicy(Qs):
    """ Find the greedy deterministic policy, 
    given the Q-values. """
    dim = len(Qs)
    numA = len(Qs[0])
    policy = zeros((dim, numA))
    for si in range(dim):
        actions = all_argmax(Qs[si])
        for a in actions:
            policy[si, a] = 1. / len(actions)    
    return policy


def randomPolicy(Ts):
    """ Each action is equally likely. """
    numA = len(Ts)
    dim = len(Ts[0])
    return ones((dim, numA)) / float(numA), mean(array(Ts), axis=0)


def randomDeterministic(Ts):
    """ Pick a random deterministic action for each state. """
    numA = len(Ts)
    dim = len(Ts[0])
    choices = (rand(dim) * numA).astype(int)
    policy = zeros((dim, numA))
    for si, a in choices:
        policy[si, a] = 1
    return policy, collapsedTransitions(Ts, policy)


def policyIteration(Ts, R, discountFactor, VEvaluator=None, initpolicy=None, maxIters=20):
    """ Given transition matrices (one per action),
    produce the optimal policy, using the policy iteration algorithm.
    
    A custom function that maps policies to value functions can be provided. """
    if initpolicy is None:
        policy, T = randomPolicy(Ts) 
    else:
        policy = initpolicy
        T = collapsedTransitions(Ts, policy)
        
    if VEvaluator is None:
        VEvaluator = lambda T: trueValues(T, R, discountFactor)
    
    while maxIters > 0:
        V = VEvaluator(T)
        newpolicy, T = greedyPolicy(Ts, R, discountFactor, V)
        # if the probabilities are not changing more than by 0.001, we're done.
        if sum(ravel(abs(newpolicy - policy))) < 1e-3:
            return policy, T
        policy = newpolicy
        maxIters -= 1
    return policy, T

