__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'

from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner


class QLambda(ValueBasedLearner):
    """ Q-lambda is a variation of Q-learning that uses an eligibility trace. """

    offPolicy = True
    batchMode = False

    def __init__(self, alpha=0.5, gamma=0.99, qlambda=0.9):
        ValueBasedLearner.__init__(self)

        self.alpha = alpha
        self.gamma = gamma
        self.qlambda = qlambda

        self.laststate = None
        self.lastaction = None


    def learn(self):
        states = self.dataset['state']
        actions = self.dataset['action']
        rewards = self.dataset['reward']

        for i in range(states.shape[0] - 1, 0, -1):
            lbda = self.qlambda ** (states.shape[0] - 1 - i)
            # if eligibility trace gets too long, break
            if lbda < 0.0001:
                break

            state = int(states[i])
            laststate = int(states[i - 1])
            # action = int(actions[i])
            lastaction = int(actions[i - 1])
            lastreward = int(rewards[i - 1])

            qvalue = self.module.getValue(laststate, lastaction)
            maxnext = self.module.getValue(state, self.module.getMaxAction(state))
            self.module.updateValue(laststate, lastaction, qvalue + self.alpha * lbda * (lastreward + self.gamma * maxnext - qvalue))
