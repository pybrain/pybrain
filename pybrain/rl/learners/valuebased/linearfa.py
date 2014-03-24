__author__ = 'Tom Schaul, tom@idsia.ch'

""" RL with linear function approximation. 

In part inspired by pseudo-code from Szepesvari's 'Algorithms for RL' (2010)
"""

from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner
from scipy import zeros, dot, outer, exp, clip, ravel, ones, rand, array, randn
from scipy.linalg import pinv2
from pybrain.utilities import r_argmax, fListToString, setAllArgs
import unittest
from random import choice, randint

def rv_dot(x, y):
    return dot(ravel(x), ravel(y))


class LinearFALearner(ValueBasedLearner):
    """ A reinforcement learner using linear function approximation (FA),
    on the states (the actions remain discrete/tabular).
     
    The weights (theta) are arranged in a 2D-array: state-features by actions.
    
    The 'state' field in the ReinforcementDataSet is to be interpreted as state-features now.
    
    Superclass for all the actual algorithms.
    """
    
    learningRate = 0.5      # aka alpha: make sure this is being decreased by calls from the learning agent!
    learningRateDecay = 100 # aka n_0, but counting decay-calls
    
    randomInit = True
    
    rewardDiscount = 0.99 # aka gamma
    
    batchMode = False
    passNextAction = False # for the _updateWeights method    
    
    def __init__(self, num_actions, num_features, **kwargs):
        ValueBasedLearner.__init__(self)
        setAllArgs(self, kwargs)
        self.explorer = None        
        self.num_actions = num_actions
        self.num_features = num_features
        if self.randomInit:
            self._theta = randn(self.num_actions, self.num_features) / 10.
        else:
            self._theta = zeros((self.num_actions, self.num_features))
        self._additionalInit()
        self._behaviorPolicy = self._boltzmannPolicy
        self.reset()
        
    def _additionalInit(self):
        pass
        
    def _qValues(self, state):
        """ Return vector of q-values for all actions, 
        given the state(-features). """
        return dot(self._theta, state)
    
    def _greedyAction(self, state):
        return r_argmax(self._qValues(state))
    
    def _greedyPolicy(self, state):
        tmp = zeros(self.num_actions)
        tmp[self._greedyAction(state)] = 1
        return tmp
    
    def _boltzmannPolicy(self, state, temperature=1.):
        tmp = self._qValues(state)
        return LinearFALearner._boltzmannProbs(tmp, temperature)
    
    @staticmethod
    def _boltzmannProbs(qvalues, temperature=1.):
        if temperature == 0:
            tmp = zeros(len(qvalues))        
            tmp[r_argmax(qvalues)] = 1.
        else:
            tmp = qvalues / temperature            
            tmp -= max(tmp)        
            tmp = exp(clip(tmp, -20, 0))
        return tmp / sum(tmp)
        
    def reset(self):        
        ValueBasedLearner.reset(self)        
        self._callcount = 0
        self.newEpisode()
    
    def newEpisode(self):  
        ValueBasedLearner.newEpisode(self)      
        self._callcount += 1
        self.learningRate *= ((self.learningRateDecay + self._callcount) 
                              / (self.learningRateDecay + self._callcount + 1.))

    
class Q_LinFA(LinearFALearner):
    """ Standard Q-learning with linear FA. """
    
    def _updateWeights(self, state, action, reward, next_state):
        """ state and next_state are vectors, action is an integer. """
        td_error = reward + self.rewardDiscount * max(dot(self._theta, next_state)) - dot(self._theta[action], state) 
        #print(action, reward, td_error,self._theta[action], state, dot(self._theta[action], state))
        #print(self.learningRate * td_error * state)
        #print()
        self._theta[action] += self.learningRate * td_error * state 
          

class QLambda_LinFA(LinearFALearner):
    """ Q-lambda with linear FA. """
    
    _lambda = 0.9
    
    def newEpisode(self):
        """ Reset eligibilities after each episode. """
        LinearFALearner.newEpisode(self)
        self._etraces = zeros((self.num_actions, self.num_features))
        
    def _updateEtraces(self, state, action, responsibility=1.):
        self._etraces *= self.rewardDiscount * self._lambda * responsibility
        self._etraces[action] += state 
            
    def _updateWeights(self, state, action, reward, next_state):
        """ state and next_state are vectors, action is an integer. """
        self._updateEtraces(state, action)
        td_error = reward + self.rewardDiscount * max(dot(self._theta, next_state)) - dot(self._theta[action], state)
        self._theta += self.learningRate * td_error * self._etraces  
        
    
class SARSALambda_LinFA(QLambda_LinFA):
    
    passNextAction = True
    
    def _updateWeights(self, state, action, reward, next_state, next_action):
        """ state and next_state are vectors, action is an integer. """
        td_error = reward + self.rewardDiscount * dot(self._theta[next_action], next_state) - dot(self._theta[action], state)
        self._updateEtraces(state, action)
        self._theta += self.learningRate * td_error * self._etraces  
        
        
class LSTDQLambda(QLambda_LinFA):
    """ Least-squares Q(lambda)"""
        
    def _additionalInit(self):
        phi_size = self.num_actions * self.num_features
        self._A = zeros((phi_size, phi_size))
        self._b = zeros(phi_size)                           
    
    def _updateWeights(self, state, action, reward, next_state, learned_policy=None):
        """ Policy is a function that returns a probability vector for all actions, 
        given the current state(-features). """
        if learned_policy is None:
            learned_policy = self._greedyPolicy
        
        self._updateEtraces(state, action)
        
        phi = zeros((self.num_actions, self.num_features))
        phi[action] += state        
        phi_n = outer(learned_policy(next_state), next_state)
        
        self._A += outer(ravel(self._etraces), ravel(phi - self.rewardDiscount * phi_n))
        self._b += reward * ravel(self._etraces)       
        
        self._theta = dot(pinv2(self._A), self._b).reshape(self.num_actions, self.num_features)
        
        
class LSPILambda(LSTDQLambda):
    """ Least-squares policy iteration (incomplete). """        
    # TODO: batch version: iterate until policy converges
        
        
class LSPI(LinearFALearner):
    """ LSPI without eligibility traces. (Mark's version) """
    
    exploring = False
    explorationReward = 1.
    
    passNextAction = True    
    
    lazyInversions = 20
    
    def _additionalInit(self):
        phi_size = self.num_actions * self.num_features
        if self.randomInit:
            self._A = randn(phi_size, phi_size) / 100.
            self._b = randn(phi_size) / 100.
        else:
            self._A = zeros((phi_size, phi_size))
            self._b = zeros(phi_size)          
        self._untouched = ones(phi_size, dtype=bool)
        self._count = 0
    
    def _updateWeights(self, state, action, reward, next_state, next_action):
        phi = zeros((self.num_actions, self.num_features))
        phi[action] += state        
        phi = ravel(phi)
        
        phi_n = zeros((self.num_actions, self.num_features))
        phi_n[next_action] += next_state
        phi_n = ravel(phi_n)
        
        self._A += outer(phi, phi - self.rewardDiscount * phi_n)
        self._b += reward * phi
        
        
        if self.lazyInversions is None or self._count % self.lazyInversions == 0:        
            if self.exploring:
                # add something to all the entries that are untouched
                self._untouched &= (phi == 0)
                res = dot(pinv2(self._A), self._b + self.explorationReward * self._untouched)
            else:
                res = dot(pinv2(self._A), self._b)
            self._theta = res.reshape(self.num_actions, self.num_features)    
            
        self._count += 1
        
        
class GQLambda(QLambda_LinFA):
    """ From Maei/Sutton 2010, with additional info from Adam White. """
    
    sec_learningRate = 1.
    
    def _additionalInit(self):
        self._sec_weights = zeros((self.num_actions, self.num_features)) # w        
    
    def _updateWeights(self, state, action, reward, next_state, behavior_policy=None, learned_policy=None, outcome_reward=None):
        if learned_policy is None:
            learned_policy = self._greedyPolicy
        if behavior_policy is None:
            behavior_policy = self._behaviorPolicy
        
        responsibility = learned_policy(state)[action] / behavior_policy(state)[action]
        self._updateEtraces(state, action, responsibility)
        
        phi_bar = outer(learned_policy(next_state), next_state)
        
        td_error = reward + self.rewardDiscount * rv_dot(self._theta, phi_bar) - dot(self._theta[action], state)
        if outcome_reward is not None:
            td_error += (1 - self.rewardDiscount) * outcome_reward
            
        self._theta += self.learningRate * (td_error * self._etraces - 
                                            self.rewardDiscount * (1 - self._lambda) * rv_dot(self._sec_weights, self._etraces) * phi_bar)
        
        phi = zeros((self.num_actions, self.num_features))
        phi[action] += state
        
        self._sec_weights += self.learningRate * self.sec_learningRate * (td_error * self._etraces - rv_dot(self._sec_weights, phi) * phi)
    


class LearningTester(unittest.TestCase):
    """ Some simple test cases. 
        Note: this can still fail from time to time!"""
        
    verbose = False
    
    algos = [Q_LinFA, QLambda_LinFA, SARSALambda_LinFA, LSPI, LSTDQLambda, GQLambda]    
    need_next_action = [LSPI, SARSALambda_LinFA]
    
    def testSingleStateFullDiscounted(self):
        r = self.runSequences(num_actions=4, num_features=3, num_states=1, num_interactions=500,
                              gamma=0, lr=0.25)
        if self.verbose:
            for x, l in r:
                print(x)
                for a in l:
                    print(fListToString(a[0], 2)        )
        for _, l in r:        
            self.assertAlmostEquals(min(l[0][0]), 1, places=0) 
            self.assertAlmostEquals(max(l[0][0]), 1, places=0) 
            self.assertAlmostEquals(2 * min(l[1][0]), 1, places=0) 
            self.assertAlmostEquals(2 * max(l[1][0]), 1, places=0) 
            self.assertAlmostEquals(min(l[2][0]), 0, places=0) 
            self.assertAlmostEquals(max(l[2][0]), len(l[2][0]) - 1, places=0) 
            self.assertAlmostEquals(min(l[3][0]), max(l[3][0]), places=0)                 
    
    def testSingleState(self):
        r = self.runSequences(num_actions=3, num_features=2, num_states=1, num_interactions=1000,
                              lr=0.2, _lambda=0.5, gamma=0.5)
        if self.verbose:
            for x, l in r:
                print(x)
                for a in l:
                    print(fListToString(a[0], 2)        )
        for _, l in r:
            self.assertAlmostEquals(min(l[0][0]), max(l[0][0]), places=0) 
            self.assertAlmostEquals(min(l[1][0]), max(l[1][0]), places=0)
            self.assertAlmostEquals(min(l[2][0]) + len(l[2][0]) - 1, max(l[2][0]), places=0)             
            self.assertAlmostEquals(min(l[3][0]), max(l[3][0]), places=0)                         
                        
    def testSingleAction(self):        
        r = self.runSequences(num_actions=1, r_states=map(array, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
                              num_interactions=1000, lr=0.1, _lambda=0.5, gamma=0.5)
        if self.verbose:
            for x, l in r:
                print(x)
                for a in l:
                    print(fListToString(a, 2)        )
        for _, l in r:
            self.assertAlmostEquals(min(l[0]), max(l[0]), places=0) 
            self.assertAlmostEquals(min(l[1]), max(l[1]), places=0)
            self.assertAlmostEquals(min(l[2]), max(l[2]), places=0)             
            self.assertAlmostEquals(max(l[3]) - 1, min(l[3]), places=0) 
            
    def testSimple(self):        
        r = self.runSequences(num_actions=3, num_features=5, num_states=4, num_interactions=2000,
                              lr=0.1, _lambda=0.5, gamma=0.5)        
        if self.verbose:
            for x, l in r:
                print(x)
                for a in l:
                    print(fListToString(a[0], 2)        )
        for _, l in r:
            self.assertAlmostEquals(min(l[0][0]), max(l[0][0]), places=0) 
            self.assertAlmostEquals(min(l[1][0]), max(l[1][0]), places=0)
            self.assertAlmostEquals(min(l[2][0]) + len(l[2][0]) - 1, max(l[2][0]), places=0)             
            self.assertAlmostEquals(min(l[3][0]), max(l[3][0]), places=0) 
            
    def runSequences(self, num_actions=1, num_features=1, num_states=1,
                     num_interactions=10000, gamma=None, _lambda=None, lr=None, r_states=None):
        if r_states is None:
            r_states = [rand(num_features) for _ in range(num_states)]
        else:
            num_features = len(r_states[0])
            num_states = len(r_states)
        state_seq = [choice(r_states) for  _ in range(num_interactions)]
        action_seq = [randint(0, num_actions - 1) for  _ in range(num_interactions)]
        rewards = [ones(num_interactions), rand(num_interactions), action_seq, [s[0] for s in state_seq]]
        datas = [zip(state_seq, action_seq, r) for r in rewards]        
        res = []        
        for algo in self.algos:
            res.append((algo.__name__, []))
            for d in datas:
                l = algo(num_actions, num_features)
                if gamma is not None:       
                    l.rewardDiscount = gamma
                if _lambda is not None:
                    l._lambda = _lambda
                if lr is not None:
                    l.learningRate = lr                     
                self.trainWith(l, d)
                res[-1][-1].append([dot(l._theta, s) for s in r_states])
        return res
            
    def trainWith(self, algo, data):
        last_s = None
        last_a = None
        last_r = None
        for s, a, r in data:
            if last_s is not None:
                if algo.__class__ in self.need_next_action:
                    algo._updateWeights(last_s, last_a, last_r, s, a)
                else:
                    algo._updateWeights(last_s, last_a, last_r, s)
            last_s = s
            last_a = a
            last_r = r        

if __name__ == '__main__':
    unittest.main()
