__author__ = 'Tom Schaul, tom@idsia.ch'


from scipy import array, sin, cos
import logging

from pybrain.rl import EpisodicTask

try: 
    import cartpolewrap as impl
except ImportError, e:
    logging.error("CartPoleTask is wrapping C code that needs to be compiled - it's simple: run .../cartpolecompile.py")
    raise e



class CartPoleTask(EpisodicTask):
    """ A Python wrapper of the standard C implentation of the pole-balancing task, directly using the 
    reference code of Faustino Gomez. """
    
    __single = None
    def __init__(self, numPoles = 1, markov = True, verbose = False, extraObservations = False):
        """ @extraObservations: if this flag is true, the observations include the cartesian coordinates 
        of the pole(s).
        """
        if self.__single != None:
            raise Exception('Singleton class - there is already an instance around', self.__single)
        self.__single = self
        impl.init(markov, numPoles)
        self.markov = markov
        self.numPoles = numPoles
        self.verbose = verbose
        self.extraObs = extraObservations
        self.reset()

    def reset(self):
        if self.verbose:
            print '** reset **'
        self.cumreward = 0     
        impl.res()        

    def getOutDim(self):
        res = 1+self.numPoles
        if self.markov:
            res *= 2
        if self.extraObs:
            res += 2*self.numPoles
        return res
        
    def getInDim(self):
        return 1

    def getReward(self):
        r = 1.+impl.getR()
        if self.verbose:
            print ' +r', r,
        return r
        
    def isFinished(self):
        if self.verbose:
            print '  -finished?', impl.isFinished()
        return impl.isFinished() 
    
    def getObservation(self):
        obs = array(impl.getObs())
        if self.verbose:
            print 'obs', obs
        if self.extraObs:
            obs.resize(self.getOutDim())
            if self.markov:
                angle1 = obs[1]
            else:
                angle1 = obs[0]
            obs[-1] = 0.1*cos(angle1)
            obs[-2] = 0.1*sin(angle1)            
            if self.numPoles == 2:
                if self.markov:
                    angle2 = obs[3]
                else:
                    angle2 = obs[1]
                obs[-3] = 0.1*cos(angle2)
                obs[-4] = 0.1*sin(angle2)
        if self.verbose:
            print 'obs', obs
        return obs
        
    def performAction(self, action):
        if self.verbose:
            print 'act', action
        impl.performAction(action[0])
        self.addReward()
        
if __name__ == '__main__':
    from pybrain.rl import EpisodicExperiment
    from pybrain.rl.agents import FlatNetworkAgent
    x = CartPoleTask()
    a = FlatNetworkAgent(x.getOutDim(), x.getInDim())
    e = EpisodicExperiment(x, a)
    e.doEpisodes(2)
    