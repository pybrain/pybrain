# $Id$
__author__ = 'Daan Wierstra and Tom Schaul'

from scipy import zeros, dot, ones, argmax
from random import shuffle

from trainer import Trainer
from pybrain.utilities import fListToString
from pybrain.auxiliary import GradientDescent


class BackpropTrainer(Trainer):
    """ Train the parameters of a module according to a supervised dataset (potentially sequential)
        by backpropagating the errors (through time). """
        
    def __init__(self, module, dataset = None, learningrate = 0.01, lrdecay=1.0, momentum = 0., 
                 verbose = False, batchlearning = False, weightdecay = 0.):
        """ @param module: the module whose parameters should be trained. 
            @param learningrate: learning rate (default: 0.01)
            @param lrdecay: learning rate decay (default: 1.0 = none)
            @param momentum: momentum (default: 0)
            @param weightdecay: weight decay rate (default: 0 = none). 
            Caveat: To ease comparison between different setups, weight decay does not alter the reported
            errors, merely the weight updates.
        """
        Trainer.__init__(self, module)
        self.setData(dataset)
        self.verbose = verbose
        self.batchlearning = batchlearning
        self.epoch = 0
        self.totalepochs = 0
        # set up gradient descender
        self.descent = GradientDescent()
        self.descent.alpha = learningrate
        self.descent.momentum = momentum
        self.descent.alphadecay = lrdecay
        self.descent.weightdecay = weightdecay
        self.descent.init(module.params)
        
        
    def train(self):
        """ @param batchlearning: should the parameters be updated only at the end of the epoch? """
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
                self.module._setParameters(self.descent(self.module.derivs))
                self.module.resetDerivatives()
        if self.verbose:
            print "Total error:", errors/ponderation
        if self.batchlearning:
            self.module._setParameters(self.descent(self.module.derivs))
        self.epoch += 1
        self.totalepochs += 1
        return errors/ponderation
        
    
    def _calcDerivs(self, seq):
        self.module.reset()        
        for time, sample in enumerate(seq):
            input = sample[0]      
            self.module.inputbuffer[time] = input
            self.module.forward()
        error = 0
        ponderation = 0.
        for time, sample in reversed(list(enumerate(seq))):
            if len(sample) > 2:
                dummy, target, importance = sample
            else:
                dummy, target = sample
                importance = ones(len(target))              
            outerr = target - self.module.outputbuffer[time]
            self.module.outputerror[time] = outerr*importance
            error += 0.5 * dot(importance, outerr**2)
            ponderation += sum(importance)
            self.module.backward()
        return error, ponderation
            
    def _checkGradient(self, dataset = None, silent = False):
        """ Numeric check of the computed gradient. To be used for debugging purposes. """
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
                self.module.params[p] -= 2*e
                lefterror, dummy = self._calcDerivs(seq)
                approxderiv = (righterror-lefterror)/(2*e)
                self.module.params[p] = storedoldval
                numericalDerivs.append(approxderiv)
            r = zip(analyticalDerivs, numericalDerivs)
            res.append(r)
            if not silent:
                print r
        return res
    
    def testOnData(self, dataset = None, verbose = False):
        """ Compute the MSE of the module performance on the given dataset.
        @param dataset: by default the one previously used by the trainer """
        if dataset == None:
            dataset = self.ds
        dataset.reset()
        if verbose:
            print '\nTesting on data:'
        errors = []
        importances = []
        ponderatedErrors = []
        for seq in dataset._provideSequences():
            self.module.reset()
            e, i = dataset._evaluateSequence(self.module.activate, seq, verbose)
            importances.append(i)
            errors.append(e)
            ponderatedErrors.append(e/i)
        if verbose:
            print 'All errors:', ponderatedErrors
        assert sum(importances) > 0
        avgErr = sum(errors)/sum(importances)
        if verbose:
            print 'Average error:', avgErr
            print 'Max error:', max(ponderatedErrors), 'Median error:', sorted(ponderatedErrors)[len(errors)/2]
        return avgErr
                
    def testOnClassData(self, dataset = None, verbose = False, return_targets=False):
        """ Return winner-takes-all classification output on given data data set
        FIXME: provisional; should also compute cross-class entropy and class errors
        @param dataset: by default the one previously used by the trainer """        
        if dataset == None:
            dataset = self.ds
        dataset.reset()
        if verbose:
            print '\nTesting on class data:'
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
        
    def trainUntilConvergence(self, alldata = None, maxEpochs = None, verbose = None,
                              continueEpochs = 1, validationProportion = 0.33):
        """ Early-Stopping
        Train on the trainingData, until the validation error stops going down (for continueEpochs epochs). 
        Return the module with the parameters that gave the minimal validation error. """
        epochs = 0
        if alldata == None:
            alldata = self.ds
        if verbose == None:
            verbose = self.verbose
        # split the dataset randomly: validationProportion of the samples for validation
        trainingData, validationData = alldata.splitWithProportion(1-validationProportion)
        self.ds = trainingData
        bestweights = self.module.params.copy()
        bestverr = self.testOnData(validationData)
        trainingErrors = []
        validationErrors = [bestverr]
        while True:
            trainingErrors.append(self.train())
            validationErrors.append(self.testOnData(validationData))
            if epochs == 0 or validationErrors[-1] < bestverr:
                # one update is always done
                bestverr = validationErrors[-1]
                bestweights = self.module.params.copy()
            
            if maxEpochs != None and epochs >= maxEpochs:
                self.module.params[:] = bestweights
                break
            epochs += 1
            
            if len(validationErrors) >= continueEpochs*2:
                # have the validation errors started going up again?
                # compare the average of the last few to the previous few
                old = validationErrors[-continueEpochs*2:-continueEpochs]
                new = validationErrors[-continueEpochs:]
                if min(new) > max(old):
                    self.module.params[:] = bestweights
                    break
        trainingErrors.append(self.testOnData(trainingData))
        self.ds = alldata
        if verbose:
            print 'train-errors:', fListToString(trainingErrors, 6)
            print 'valid-errors:', fListToString(validationErrors, 6)
        return trainingErrors, validationErrors
            
            
