#@Pydev CodeAnalysisIgnore

__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import array, sin, cos, randn
import logging

from pybrain.rl.environments.episodic import EpisodicTask


try:
    import cartpolewrap as impl
except ImportError as e:
    logging.error("FastCartPoleTask is wrapping C code that needs to be compiled - it's simple: run .../cartpolecompile.py")
    raise e



class FastCartPoleTask(EpisodicTask):
    """ A Python wrapper of the standard C implementation of the pole-balancing task, directly using the
    reference code of Faustino Gomez. """

    indim = 1

    desiredValue = 100000

    # additional random observations
    extraRandoms = 0

    __single = None
    def __init__(self, numPoles=1, markov=True, verbose=False,
                 extraObservations=False, extraRandoms=0, maxSteps=100000):
        """ @extraObservations: if this flag is true, the observations include the Cartesian coordinates
        of the pole(s).
        """
        if self.__single != None:
            raise Exception('Singleton class - there is already an instance around', self.__single)
        self.__single = self
        impl.init(markov, numPoles, maxSteps)
        self.markov = markov
        self.numPoles = numPoles
        self.verbose = verbose
        self.extraObservations = extraObservations
        self.extraRandoms = extraRandoms
        self.desiredValue = maxSteps
        self.reset()

    def __str__(self):
        s = 'Cart-Pole-Balancing-Task, '
        if self.markov:
            s += 'markovian'
        else:
            s += 'non-markovian'
        s += ', with '
        if self.numPoles == 1:
            s += 'a single pole'
        else:
            s += str(self.numPoles) + ' poles'
        if self.extraObservations:
            s += ' and additional observations (cartesian coordinates of tip of pole(s))'
        if self.extraRandoms > 0:
            s += ' and ' + str(self.extraRandoms) + ' additional random observations'
        return s

    def reset(self):
        if self.verbose:
            print('** reset **')
        self.cumreward = 0
        impl.res()

    @property
    def outdim(self):
        res = 1 + self.numPoles
        if self.markov:
            res *= 2
        if self.extraObservations:
            res += 2 * self.numPoles
        res += self.extraRandoms
        return res

    def getReward(self):
        r = 1. + impl.getR()
        if self.verbose:
            print((' +r', r,))
        return r

    def isFinished(self):
        if self.verbose:
            print(('  -finished?', impl.isFinished()))
        return impl.isFinished()

    def getObservation(self):
        obs = array(impl.getObs())
        if self.verbose:
            print(('obs', obs))
        obs.resize(self.outdim)
        if self.extraObservations:
            cartpos = obs[-1]
            if self.markov:
                angle1 = obs[1]
            else:
                angle1 = obs[0]
            obs[-1 + self.extraRandoms] = 0.1 * cos(angle1) + cartpos
            obs[-2 + self.extraRandoms] = 0.1 * sin(angle1) + cartpos
            if self.numPoles == 2:
                if self.markov:
                    angle2 = obs[3]
                else:
                    angle2 = obs[1]
                obs[-3 + self.extraRandoms] = 0.05 * cos(angle2) + cartpos
                obs[-4 + self.extraRandoms] = 0.05 * sin(angle2) + cartpos

        if self.extraRandoms > 0:
            obs[-self.extraRandoms:] = randn(self.extraRandoms)

        if self.verbose:
            print(('obs', obs))
        return obs

    def performAction(self, action):
        if self.verbose:
            print(('act', action))
        impl.performAction(action[0])
        self.addReward()




