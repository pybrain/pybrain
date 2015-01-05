from __future__ import print_function

__author__ = 'Daan Wierstra and Tom Schaul'

from scipy import dot, argmax
from random import shuffle
from math import isnan
from pybrain.supervised.trainers.trainer import Trainer
from pybrain.utilities import fListToString
from pybrain.auxiliary import GradientDescent


class BackpropTrainer(Trainer):
    """Trainer that trains the parameters of a module according to a
    supervised dataset (potentially sequential) by backpropagating the errors
    (through time)."""

    def __init__(self, module, dataset=None, learningrate=0.01, lrdecay=1.0,
                 momentum=0., verbose=False, batchlearning=False,
                 weightdecay=0.):
        """Create a BackpropTrainer to train the specified `module` on the
        specified `dataset`.

        The learning rate gives the ratio of which parameters are changed into
        the direction of the gradient. The learning rate decreases by `lrdecay`,
        which is used to to multiply the learning rate after each training
        step. The parameters are also adjusted with respect to `momentum`, which
        is the ratio by which the gradient of the last timestep is used.

        If `batchlearning` is set, the parameters are updated only at the end of
        each epoch. Default is False.

        `weightdecay` corresponds to the weightdecay rate, where 0 is no weight
        decay at all.
        """
        Trainer.__init__(self, module)
        self.setData(dataset)
        self.verbose = verbose
        self.batchlearning = batchlearning
        self.weightdecay = weightdecay
        self.epoch = 0
        self.totalepochs = 0
        # set up gradient descender
        self.descent = GradientDescent()
        self.descent.alpha = learningrate
        self.descent.momentum = momentum
        self.descent.alphadecay = lrdecay
        self.descent.init(module.params)

    def train(self):
        """Train the associated module for one epoch."""
        assert len(self.ds) > 0, "Dataset cannot be empty."
        self.module.resetDerivatives()
        errors = 0
        ponderation = 0.
        shuffledSequences = []
        for seq in self.ds._provideSequences():
            shuffledSequences.append(seq)
        shuffle(shuffledSequences)
        for seq in shuffledSequences:
            e, p = self._calcDerivs(seq)
            errors += e
            ponderation += p
            if not self.batchlearning:
                gradient = self.module.derivs - self.weightdecay * self.module.params
                new = self.descent(gradient, errors)
                if new is not None:
                    self.module.params[:] = new
                self.module.resetDerivatives()

        if self.verbose:
            print("Total error: {z: .12g}".format(z=errors / ponderation))
        if self.batchlearning:
            self.module._setParameters(self.descent(self.module.derivs))
        self.epoch += 1
        self.totalepochs += 1
        return errors / ponderation


    def _calcDerivs(self, seq):
        """Calculate error function and backpropagate output errors to yield
        the gradient."""
        self.module.reset()
        for sample in seq:
            self.module.activate(sample[0])
        error = 0
        ponderation = 0.
        for offset, sample in reversed(list(enumerate(seq))):
            # need to make a distinction here between datasets containing
            # importance, and others
            target = sample[1]
            outerr = target - self.module.outputbuffer[offset]
            if len(sample) > 2:
                importance = sample[2]
                error += 0.5 * dot(importance, outerr ** 2)
                ponderation += sum(importance)
                self.module.backActivate(outerr * importance)
            else:
                error += 0.5 * sum(outerr ** 2)
                ponderation += len(target)
                # FIXME: the next line keeps arac from producing NaNs. I don't
                # know why that is, but somehow the __str__ method of the
                # ndarray class fixes something,
                str(outerr)
                self.module.backActivate(outerr)

        return error, ponderation

    def _checkGradient(self, dataset=None, silent=False):
        """Numeric check of the computed gradient for debugging purposes."""
        if dataset:
            self.setData(dataset)
        res = []
        for seq in self.ds._provideSequences():
            self.module.resetDerivatives()
            self._calcDerivs(seq)
            e = 1e-6
            analyticalDerivs = self.module.derivs.copy()
            numericalDerivs = []
            for p in range(self.module.paramdim):
                storedoldval = self.module.params[p]
                self.module.params[p] += e
                righterror, dummy = self._calcDerivs(seq)
                self.module.params[p] -= 2 * e
                lefterror, dummy = self._calcDerivs(seq)
                approxderiv = (righterror - lefterror) / (2 * e)
                self.module.params[p] = storedoldval
                numericalDerivs.append(approxderiv)
            r = list(zip(analyticalDerivs, numericalDerivs))
            res.append(r)
            if not silent:
                print(r)
        return res

    def testOnData(self, dataset=None, verbose=False):
        """Compute the MSE of the module performance on the given dataset.

        If no dataset is supplied, the one passed upon Trainer initialization is
        used."""
        if dataset == None:
            dataset = self.ds
        dataset.reset()
        if verbose:
            print('\nTesting on data:')
        errors = []
        importances = []
        ponderatedErrors = []
        for seq in dataset._provideSequences():
            self.module.reset()
            e, i = dataset._evaluateSequence(self.module.activate, seq, verbose)
            importances.append(i)
            errors.append(e)
            ponderatedErrors.append(e / i)
        if verbose:
            print(('All errors:', ponderatedErrors))
        assert sum(importances) > 0
        avgErr = sum(errors) / sum(importances)
        if verbose:
            print(('Average error:', avgErr))
            print(('Max error:', max(ponderatedErrors), 'Median error:',
                   sorted(ponderatedErrors)[len(errors) / 2]))
        return avgErr

    def testOnClassData(self, dataset=None, verbose=False,
                        return_targets=False):
        """Return winner-takes-all classification output on a given dataset.

        If no dataset is given, the dataset passed during Trainer
        initialization is used. If return_targets is set, also return
        corresponding target classes.
        """
        if dataset == None:
            dataset = self.ds
        dataset.reset()
        out = []
        targ = []
        for seq in dataset._provideSequences():
            self.module.reset()
            for input, target in seq:
                res = self.module.activate(input)
                out.append(argmax(res))
                targ.append(argmax(target))
        if return_targets:
            return out, targ
        else:
            return out

    def trainUntilConvergence(self, dataset=None, maxEpochs=None, verbose=None,
                              continueEpochs=10, validationProportion=0.25,
                              trainingData=None, validationData=None,
                              convergence_threshold=10):
        """Train the module on the dataset until it converges.

        Return the module with the parameters that gave the minimal validation
        error.

        If no dataset is given, the dataset passed during Trainer
        initialization is used. validationProportion is the ratio of the dataset
        that is used for the validation dataset.
        
        If the training and validation data is already set, the splitPropotion is ignored

        If maxEpochs is given, at most that many epochs
        are trained. Each time validation error hits a minimum, try for
        continueEpochs epochs to find a better one."""
        epochs = 0
        if dataset is None:
            dataset = self.ds
        if verbose is None:
            verbose = self.verbose
        if trainingData is None or validationData is None:
            # Split the dataset randomly: validationProportion of the samples for
            # validation.
            trainingData, validationData = (
                dataset.splitWithProportion(1 - validationProportion))
        if not (len(trainingData) > 0 and len(validationData)):
            raise ValueError("Provided dataset too small to be split into training " +
                             "and validation sets with proportion " + str(validationProportion))
        self.ds = trainingData
        bestweights = self.module.params.copy()
        bestverr = self.testOnData(validationData)
        bestepoch = 0
        self.trainingErrors = []
        self.validationErrors = [bestverr]
        while True:
            trainingError = self.train()
            validationError = self.testOnData(validationData)
            if isnan(trainingError) or isnan(validationError):
                raise Exception("Training produced NaN results")
            self.trainingErrors.append(trainingError)
            self.validationErrors.append(validationError)
            if epochs == 0 or self.validationErrors[-1] < bestverr:
                # one update is always done
                bestverr = self.validationErrors[-1]
                bestweights = self.module.params.copy()
                bestepoch = epochs

            if maxEpochs != None and epochs >= maxEpochs:
                self.module.params[:] = bestweights
                break
            epochs += 1

            if len(self.validationErrors) >= continueEpochs * 2:
                # have the validation errors started going up again?
                # compare the average of the last few to the previous few
                old = self.validationErrors[-continueEpochs * 2:-continueEpochs]
                new = self.validationErrors[-continueEpochs:]
                if min(new) > max(old):
                    self.module.params[:] = bestweights
                    break
                lastnew = round(new[-1], convergence_threshold)
                if sum(round(y, convergence_threshold) - lastnew for y in new) == 0:
                    self.module.params[:] = bestweights
                    break
        #self.trainingErrors.append(self.testOnData(trainingData))
        self.ds = dataset
        if verbose:
            print(('train-errors:', fListToString(self.trainingErrors, 6)))
            print(('valid-errors:', fListToString(self.validationErrors, 6)))
        return self.trainingErrors[:bestepoch], self.validationErrors[:1 + bestepoch]
