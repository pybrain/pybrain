# Neural network data analysis tool collection. Makes heavy use of the logging module.
# Can generate training curves during the run (from properly setup IPython and/or with
# TkAgg backend and interactive mode - see matplotlib documentation).
__author__ = "Martin Felder"
__version__ = "$Id$"

from pylab import ion, figure, draw
import csv
from numpy import Infinity
import logging

from pybrain.datasets                  import ClassificationDataSet, SequentialDataSet
from pybrain.tools.shortcuts           import buildNetwork
from pybrain.supervised                import BackpropTrainer, RPropMinusTrainer, Trainer
from pybrain.structure                 import SoftmaxLayer, LSTMLayer
from pybrain.utilities                 import setAllArgs
from pybrain.tools.plotting            import MultilinePlotter
from pybrain.tools.validation          import testOnSequenceData, ModuleValidator, Validator
from pybrain.tools.customxml           import NetworkWriter


class NNtools(object):
    """ Abstract class providing basic functionality to make neural network training more comfortable """

    def __init__(self, DS, **kwargs):
        """ Initialize with the training data set DS. All keywords given are set as member variables.
        The following are particularly important:

        :key hidden: number of hidden units
        :key TDS: test data set for checking convergence
        :key VDS: validation data set for final performance evaluation
        :key epoinc: number of epochs to train for, before checking convergence (default: 5)
        """
        self.DS = DS
        self.hidden = 10
        self.maxepochs = 1000
        self.Graph = None
        self.TDS = None
        self.VDS = None
        self.epoinc = 5
        setAllArgs(self, kwargs)
        self.trainCurve = None


    def initGraphics(self, ymax=10, xmax= -1):
        """ initialize the interactive graphics output window, and return a handle to the plot """
        if xmax < 0:
            xmax = self.maxepochs
        figure(figsize=[12, 8])
        ion()
        draw()
        #self.Graph = MultilinePlotter(autoscale=1.2 ) #xlim=[0, self.maxepochs], ylim=[0, ymax])
        self.Graph = MultilinePlotter(xlim=[0, xmax], ylim=[0, ymax])
        self.Graph.setLineStyle([0, 1], linewidth=2)
        return self.Graph


    def set(self, **kwargs):
        """ convenience method to set several member variables at once """
        setAllArgs(self, kwargs)


    def saveTrainingCurve(self, learnfname):
        """ save the training curves into a file with the given name (CSV format) """
        logging.info('Saving training curves into ' + learnfname)
        if self.trainCurve is None:
            logging.error('No training curve available for saving!')
        learnf = open(learnfname, "wb")
        writer = csv.writer(learnf, dialect='excel')
        nDataSets = len(self.trainCurve)
        for i in range(1, len(self.trainCurve[0]) - 1):
            writer.writerow([self.trainCurve[k][i] for k in range(nDataSets)])
        learnf.close()

    def saveNetwork(self, fname):
        """ save the trained network to a file """
        NetworkWriter.writeToFile(self.Trainer.module, fname)
        logging.info("Network saved to: " + fname)


#=======================================================================================================

class NNregression(NNtools):
    """ Learns to numerically predict the targets of a set of data, with optional online progress plots. """


    def setupNN(self, trainer=RPropMinusTrainer, hidden=None, **trnargs):
        """ Constructs a 3-layer FNN for regression. Optional arguments are passed on to the Trainer class. """
        if hidden is not None:
            self.hidden = hidden
        logging.info("Constructing FNN with following config:")
        FNN = buildNetwork(self.DS.indim, self.hidden, self.DS.outdim)
        logging.info(str(FNN) + "\n  Hidden units:\n    " + str(self.hidden))
        logging.info("Training FNN with following special arguments:")
        logging.info(str(trnargs))
        self.Trainer = trainer(FNN, dataset=self.DS, **trnargs)


    def runTraining(self, convergence=0, **kwargs):
        """ Trains the network on the stored dataset. If convergence is >0, check after that many epoch increments
        whether test error is going down again, and stop training accordingly.
        CAVEAT: No support for Sequential datasets!"""
        assert isinstance(self.Trainer, Trainer)
        if self.Graph is not None:
            self.Graph.setLabels(x='epoch', y='normalized regression error')
            self.Graph.setLegend(['training', 'test'], loc='upper right')
        epoch = 0
        inc = self.epoinc
        best_error = Infinity
        best_epoch = 0
        learncurve_x = [0]
        learncurve_y = [0.0]
        valcurve_y = [0.0]
        converged = False
        convtest = 0
        if convergence > 0:
            logging.info("Convergence criterion: %d batches of %d epochs w/o improvement" % (convergence, inc))
        while epoch <= self.maxepochs and not converged:
            self.Trainer.trainEpochs(inc)
            epoch += inc
            learncurve_x.append(epoch)
            # calculate errors on TRAINING data
            err_trn = ModuleValidator.validate(Validator.MSE, self.Trainer.module, self.DS)
            learncurve_y.append(err_trn)
            if self.TDS is None:
                logging.info("epoch: %6d,  err_trn: %10g" % (epoch, err_trn))
            else:
                # calculate same errors on TEST data
                err_tst = ModuleValidator.validate(Validator.MSE, self.Trainer.module, self.TDS)
                valcurve_y.append(err_tst)
                if err_tst < best_error:
                    # store best error and parameters
                    best_epoch = epoch
                    best_error = err_tst
                    bestweights = self.Trainer.module.params.copy()
                    convtest = 0
                else:
                    convtest += 1
                logging.info("epoch: %6d,  err_trn: %10g,  err_tst: %10g,  best_tst: %10g" % (epoch, err_trn, err_tst, best_error))
                if self.Graph is not None:
                    self.Graph.addData(1, epoch, err_tst)

                # check if convegence criterion is fulfilled (no improvement after N epoincs)
                if convtest >= convergence:
                    converged = True

            if self.Graph is not None:
                self.Graph.addData(0, epoch, err_trn)
                self.Graph.update()

        # training finished!
        logging.info("Best epoch: %6d, with error: %10g" % (best_epoch, best_error))
        if self.VDS is not None:
            # calculate same errors on VALIDATION data
            self.Trainer.module.params[:] = bestweights.copy()
            err_val = ModuleValidator.validate(Validator.MSE, self.Trainer.module, self.VDS)
            logging.info("Result on evaluation data: %10g" % err_val)
        # store training curve for saving into file
        self.trainCurve = (learncurve_x, learncurve_y, valcurve_y)

#=======================================================================================================

class NNclassifier(NNtools):
    """ Learns to classify a set of data, with optional online progress plots. """

    def __init__(self, DS, **kwargs):
        """ Initialize the classifier: the least we need is the dataset to be classified. All keywords given are set as member variables. """
        if not isinstance(DS, ClassificationDataSet):
            raise TypeError('Need a ClassificationDataSet to do classification!')
        NNtools.__init__(self, DS, **kwargs)
        self.nClasses = self.DS.nClasses  # need this because targets may be altered later
        self.clsnames = None
        self.targetsAreOneOfMany = False


    def _convertAllDataToOneOfMany(self, values=[0, 1]):
        """ converts all datasets associated with self into 1-out-of-many representations,
        e.g. with original classes 0 to 4, the new target for class 1 would be [0,1,0,0,0],
        or accordingly with other upper and lower bounds, as given by the values keyword """
        if self.targetsAreOneOfMany:
            return
        else:
            # convert all datasets to one-of-many ("winner takes all") representation
            for dsname in ["DS", "TDS", "VDS"]:
                d = getattr(self, dsname)
                if d is not None:
                    if d.outdim < d.nClasses:
                        d._convertToOneOfMany(values)
            self.targetsAreOneOfMany = True


    def setupNN(self, trainer=RPropMinusTrainer, hidden=None, **trnargs):
        """ Setup FNN and trainer for classification. """
        self._convertAllDataToOneOfMany()
        if hidden is not None:
            self.hidden = hidden
        FNN = buildNetwork(self.DS.indim, self.hidden, self.DS.outdim, outclass=SoftmaxLayer)
        logging.info("Constructing classification FNN with following config:")
        logging.info(str(FNN) + "\n  Hidden units:\n    " + str(self.hidden))
        logging.info("Trainer received the following special arguments:")
        logging.info(str(trnargs))
        self.Trainer = trainer(FNN, dataset=self.DS, **trnargs)


    def setupRNN(self, trainer=BackpropTrainer, hidden=None, **trnargs):
        """ Setup an LSTM RNN and trainer for sequence classification. """
        if hidden is not None:
            self.hidden = hidden
        self._convertAllDataToOneOfMany()

        RNN = buildNetwork(self.DS.indim, self.hidden, self.DS.outdim, hiddenclass=LSTMLayer, 
                           recurrent=True, outclass=SoftmaxLayer)
        logging.info("Constructing classification RNN with following config:")
        logging.info(str(RNN) + "\n  Hidden units:\n    " + str(self.hidden))
        logging.info("Trainer received the following special arguments:")
        logging.info(str(trnargs))
        self.Trainer = trainer(RNN, dataset=self.DS, **trnargs)


    def runTraining(self, convergence=0, **kwargs):
        """ Trains the network on the stored dataset. If convergence is >0, check after that many epoch increments
        whether test error is going down again, and stop training accordingly. """
        assert isinstance(self.Trainer, Trainer)
        if self.Graph is not None:
            self.Graph.setLabels(x='epoch', y='% classification error')
            self.Graph.setLegend(['training', 'test'], loc='lower right')
        epoch = 0
        inc = self.epoinc
        best_error = 100.0
        best_epoch = 0
        learncurve_x = [0]
        learncurve_y = [0.0]
        valcurve_y = [0.0]
        converged = False
        convtest = 0
        if convergence > 0:
            logging.info("Convergence criterion: %d batches of %d epochs w/o improvement" % (convergence, inc))
        while epoch <= self.maxepochs and not converged:
            self.Trainer.trainEpochs(inc)
            epoch += inc
            learncurve_x.append(epoch)
            # calculate errors on TRAINING data
            if isinstance(self.DS, SequentialDataSet):
                r_trn = 100. * (1.0 - testOnSequenceData(self.Trainer.module, self.DS))
            else:
                # FIXME: messy - validation does not belong into the Trainer...
                out, trueclass = self.Trainer.testOnClassData(return_targets=True)
                r_trn = 100. * (1.0 - Validator.classificationPerformance(out, trueclass))
            learncurve_y.append(r_trn)
            if self.TDS is None:
                logging.info("epoch: %6d,  err_trn: %5.2f%%" % (epoch, r_trn))
            else:
                # calculate errors on TEST data
                if isinstance(self.DS, SequentialDataSet):
                    r_tst = 100. * (1.0 - testOnSequenceData(self.Trainer.module, self.TDS))
                else:
                    # FIXME: messy - validation does not belong into the Trainer...
                    out, trueclass = self.Trainer.testOnClassData(return_targets=True, dataset=self.TDS)
                    r_tst = 100. * (1.0 - Validator.classificationPerformance(out, trueclass))
                valcurve_y.append(r_tst)
                if r_tst < best_error:
                    best_epoch = epoch
                    best_error = r_tst
                    bestweights = self.Trainer.module.params.copy()
                    convtest = 0
                else:
                    convtest += 1
                logging.info("epoch: %6d,  err_trn: %5.2f%%,  err_tst: %5.2f%%,  best_tst: %5.2f%%" % (epoch, r_trn, r_tst, best_error))
                if self.Graph is not None:
                    self.Graph.addData(1, epoch, r_tst)

                # check if convegence criterion is fulfilled (no improvement after N epoincs)
                if convtest >= convergence:
                    converged = True

            if self.Graph is not None:
                self.Graph.addData(0, epoch, r_trn)
                self.Graph.update()

        logging.info("Best epoch: %6d, with error: %5.2f%%" % (best_epoch, best_error))
        if self.VDS is not None:
            # calculate errors on VALIDATION data
            self.Trainer.module.params[:] = bestweights.copy()
            if isinstance(self.DS, SequentialDataSet):
                r_val = 100. * (1.0 - testOnSequenceData(self.Trainer.module, self.VDS))
            else:
                out, trueclass = self.Trainer.testOnClassData(return_targets=True, dataset=self.VDS)
                r_val = 100. * (1.0 - Validator.classificationPerformance(out, trueclass))
            logging.info("Result on evaluation data: %5.2f%%" % r_val)

        self.trainCurve = (learncurve_x, learncurve_y, valcurve_y)


