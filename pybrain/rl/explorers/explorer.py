__author__ = "Thomas Rueckstiess, ruecksti@in.tum.de"


from pybrain.structure.modules.module import Module


class Explorer(Module):
    """ An Explorer object is used in Agents, receives the current state
        and action (from the controller Module) and returns an explorative
        action that is executed instead the given action.
        
        Continous explorer will produce continous action states, discrete
        once discrete actions accordingly. 
        
        Explorer                        action    episodic?
        =============================== ========= =========
        NormalExplorer                  continous no
        StateDependentExplorer          continous yes
        BoltzmannExplorer               discrete  no
        EpsilonGreedyExplorer           discrete  no
        DiscreteStateDependentExplorer  discrete  yes
        

        Explorer has to be added to the learner before adding the learner
        to the LearningAgent.

        For Example::

            controller = ActionValueNetwork(2, 100)
            learner = SARSA()
            learner.explorer = NormalExplorer(1, 0.1)
            self.learning_agent = LearningAgent(controller, learner)
    """

    def activate(self, state, action):
        """ The super class commonly ignores the state and simply passes the
            action through the module. implement _forwardImplementation()
            in subclasses.
        """
        return Module.activate(self, action)


    def newEpisode(self):
        """ Inform the explorer about the start of a new episode. """
        pass
