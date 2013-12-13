__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner


class Q(ValueBasedLearner):

    offPolicy = True
    batchMode = True

    def __init__(self, alpha=0.5, gamma=0.99):
        ValueBasedLearner.__init__(self)

        self.alpha = alpha
        self.gamma = gamma

        self.laststate = None
        self.lastaction = None

    def learn(self):
        """ Learn on the current dataset, either for many timesteps and
            even episodes (batchMode = True) or for a single timestep
            (batchMode = False). Batch mode is possible, because Q-Learning
            is an off-policy method.

            In batchMode, the algorithm goes through all the samples in the
            history and performs an update on each of them. if batchMode is
            False, only the last data sample is considered. The user himself
            has to make sure to keep the dataset consistent with the agent's
            history.
        """
        if self.batchMode:
            samples = self.dataset
        else:
            samples = [[self.dataset.getSample()]]

        for seq in samples:
            # information from the previous episode (sequence)
            # should not influence the training on this episode
            self.laststate = None
            self.lastaction = None
            self.lastreward = None

            for state, action, reward in seq:

                state = int(state)
                action = int(action)

                # first learning call has no last state: skip
                if self.laststate == None:
                    self.lastaction = action
                    self.laststate = state
                    self.lastreward = reward
                    continue

                qvalue = self.module.getValue(self.laststate, self.lastaction)
                maxnext = self.module.getValue(state, self.module.getMaxAction(state))
                self.module.updateValue(self.laststate, self.lastaction, qvalue + self.alpha * (self.lastreward + self.gamma * maxnext - qvalue))

                # move state to oldstate
                self.laststate = state
                self.lastaction = action
                self.lastreward = reward

