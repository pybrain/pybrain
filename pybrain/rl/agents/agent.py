from pybrain.utilities import abstractMethod

class Agent(object):
    """ An agent is an entity capable of producing actions, based on previous observations.
        Generally it will also learn from experience. It can interact directly with a Task. 
    """
    
    def integrateObservation(self, obs):
        """ integrate the current observation of the environment.
            @param obs: The last observation returned from the environment
            @type obs: by default, this is assumed to be a numpy array of doubles
        """
        pass
        
    def getAction(self):
        """ return a chosen action.
            @rtype: by default, this is assumed to ba a numpy array of doubles.
            @note: This method is abstract and needs to be implemented.
        """
        abstractMethod()
        
    def giveReward(self, r):
        """ reward or punish the agent.
            @param r: reward, if C{r} is positive, punishment if C{r} is negative
            @type r: double            
        """             
        pass
