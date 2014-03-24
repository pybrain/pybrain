__author__ = 'Tom Schaul, tom@idsia.ch and Daan Wiertra, daan@idsia.ch'

from scipy import zeros, array, mean, randn, exp, dot, argmax

from pybrain.datasets import ReinforcementDataSet, ImportanceDataSet, SequentialDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.utilities import drawIndex
from pybrain.rl.learners.directsearch.directsearch import DirectSearchLearner


# TODO: greedy runs: start once in every possible starting state!
# TODO: supervised: train-set, test-set, early stopping -> actual convergence!


class RWR(DirectSearchLearner):
    """ Reward-weighted regression.

    The algorithm is currently limited to discrete-action episodic tasks, subclasses of POMDPTasks.
    """

    # parameters
    batchSize = 20

    # feedback settings
    verbose = True
    greedyRuns = 20
    supervisedPlotting = False

    # settings for the supervised training
    learningRate = 0.005
    momentum = 0.9
    maxEpochs = 20
    validationProportion = 0.33
    continueEpochs = 2

    # parameters for the variation that uses a value function
    # TODO: split into 2 classes.
    valueLearningRate = None
    valueMomentum = None
    #valueTrainEpochs = 5
    resetAllWeights = False
    netweights = 0.01

    def __init__(self, net, task, valueNetwork=None, **args):
        self.net = net
        self.task = task
        self.setArgs(**args)
        if self.valueLearningRate == None:
            self.valueLearningRate = self.learningRate
        if self.valueMomentum == None:
            self.valueMomentum = self.momentum
        if self.supervisedPlotting:
            from pylab import ion
            ion()

        # adaptive temperature:
        self.tau = 1.

        # prepare the datasets to be used
        self.weightedDs = ImportanceDataSet(self.task.outdim, self.task.indim)
        self.rawDs = ReinforcementDataSet(self.task.outdim, self.task.indim)
        self.valueDs = SequentialDataSet(self.task.outdim, 1)

        # prepare the supervised trainers
        self.bp = BackpropTrainer(self.net, self.weightedDs, self.learningRate,
                                  self.momentum, verbose=False,
                                  batchlearning=True)

        # CHECKME: outsource
        self.vnet = valueNetwork
        if valueNetwork != None:
            self.vbp = BackpropTrainer(self.vnet, self.valueDs, self.valueLearningRate,
                                       self.valueMomentum, verbose=self.verbose)

        # keep information:
        self.totalSteps = 0
        self.totalEpisodes = 0

    def shapingFunction(self, R):
        return exp(self.tau * R)

    def updateTau(self, R, U):
        self.tau = sum(U) / dot((R - self.task.minReward), U)

    def reset(self):
        self.weightedDs.clear()
        self.valueDs.clear()
        self.rawDs.clear()
        self.bp.momentumvector *= 0.0
        if self.vnet != None:
            self.vbp.momentumvector *= 0.0
            if self.resetAllWeights:
                self.vnet.params[:] = randn(len(self.vnet.params)) * self.netweights

    def greedyEpisode(self):
        """ run one episode with greedy decisions, return the list of rewards recieved."""
        rewards = []
        self.task.reset()
        self.net.reset()
        while not self.task.isFinished():
            obs = self.task.getObservation()
            act = self.net.activate(obs)
            chosen = argmax(act)
            self.task.performAction(chosen)
            reward = self.task.getReward()
            rewards.append(reward)
        return rewards

    def learn(self, batches):
        self.greedyAvg = []
        self.rewardAvg = []
        self.lengthAvg = []
        self.initr0Avg = []
        for b in range(batches):
            if self.verbose:
                print
                print('Batch', b + 1)
            self.reset()
            self.learnOneBatch()
            self.totalEpisodes += self.batchSize

            # greedy measure (avg over some greedy runs)
            rws = 0.
            for dummy in range(self.greedyRuns):
                tmp = self.greedyEpisode()
                rws += (sum(tmp) / float(len(tmp)))
            self.greedyAvg.append(rws / self.greedyRuns)
            if self.verbose:
                print('::', round(rws / self.greedyRuns, 5), '::')

    def learnOneBatch(self):
        # collect a batch of runs as experience
        r0s = []
        lens = []
        avgReward = 0.
        for dummy in range(self.batchSize):
            self.rawDs.newSequence()
            self.valueDs.newSequence()
            self.task.reset()
            self.net.reset()
            acts, obss, rewards = [], [], []
            while not self.task.isFinished():
                obs = self.task.getObservation()
                act = self.net.activate(obs)
                chosen = drawIndex(act)
                self.task.performAction(chosen)
                reward = self.task.getReward()
                obss.append(obs)
                y = zeros(len(act))
                y[chosen] = 1
                acts.append(y)
                rewards.append(reward)
            avgReward += sum(rewards) / float(len(rewards))

            # compute the returns from the list of rewards
            current = 0
            returns = []
            for r in reversed(rewards):
                current *= self.task.discount
                current += r
                returns.append(current)
            returns.reverse()
            for i in range(len(obss)):
                self.rawDs.addSample(obss[i], acts[i], returns[i])
                self.valueDs.addSample(obss[i], returns[i])
            r0s.append(returns[0])
            lens.append(len(returns))

        r0s = array(r0s)
        self.totalSteps += sum(lens)
        avgLen = sum(lens) / float(self.batchSize)
        avgR0 = mean(r0s)
        avgReward /= self.batchSize
        if self.verbose:
            print('***', round(avgLen, 3), '***', '(avg init exp. return:', round(avgR0, 5), ')',)
            print('avg reward', round(avgReward, 5), '(tau:', round(self.tau, 3), ')')
            print(lens)
        # storage:
        self.rewardAvg.append(avgReward)
        self.lengthAvg.append(avgLen)
        self.initr0Avg.append(avgR0)


#        if self.vnet == None:
#            # case 1: no value estimator:

        # prepare the dataset for training the acting network
        shaped = self.shapingFunction(r0s)
        self.updateTau(r0s, shaped)
        shaped /= max(shaped)
        for i, seq in enumerate(self.rawDs):
            self.weightedDs.newSequence()
            for sample in seq:
                obs, act, dummy = sample
                self.weightedDs.addSample(obs, act, shaped[i])

#        else:
#            # case 2: value estimator:
#
#
#            # train the value estimating network
#            if self.verbose: print('Old value error:  ', self.vbp.testOnData())
#            self.vbp.trainEpochs(self.valueTrainEpochs)
#            if self.verbose: print('New value error:  ', self.vbp.testOnData())
#
#            # produce the values and analyze
#            rminusvs = []
#            sizes = []
#            for i, seq in enumerate(self.valueDs):
#                self.vnet.reset()
#                seq = list(seq)
#                for sample in seq:
#                    obs, ret = sample
#                    val = self.vnet.activate(obs)
#                    rminusvs.append(ret-val)
#                    sizes.append(len(seq))
#
#            rminusvs = array(rminusvs)
#            shapedRminusv = self.shapingFunction(rminusvs)
#            # CHECKME: here?
#            self.updateTau(rminusvs, shapedRminusv)
#            shapedRminusv /= array(sizes)
#            shapedRminusv /= max(shapedRminusv)
#
#            # prepare the dataset for training the acting network
#            rvindex = 0
#            for i, seq in enumerate(self.rawDs):
#                self.weightedDs.newSequence()
#                self.vnet.reset()
#                for sample in seq:
#                    obs, act, ret = sample
#                    self.weightedDs.addSample(obs, act, shapedRminusv[rvindex])
#                    rvindex += 1

        # train the acting network
        tmp1, tmp2 = self.bp.trainUntilConvergence(maxEpochs=self.maxEpochs,
                                                   validationProportion=self.validationProportion,
                                                   continueEpochs=self.continueEpochs,
                                                   verbose=self.verbose)
        if self.supervisedPlotting:
            from pylab import plot, legend, figure, clf, draw
            figure(1)
            clf()
            plot(tmp1, label='train')
            plot(tmp2, label='valid')
            legend()
            draw()

        return avgLen, avgR0


