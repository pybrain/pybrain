

__author__ = 'Michael Isik'


from pybrain.supervised.trainers.svm import SVMTrainer
from pybrain.structure.modules.mcsvm import MCSVMOneAgainstAll
from trainer import Trainer



class MCSVMTrainer(Trainer):
    """ A trainer for multiclass support vector machines. Hence, use it in
        combination with a MCSVM module.
    """
    def __init__(self, module, dataset = None, **kwargs):
        """ Inititalize the MCSVMTrainer instance """
        Trainer.__init__(self,module)
        if dataset is not None:
            self.setData(dataset)
        self._verbosity = 0.
        self._trainer_class = SVMTrainer
        self.setParams(**kwargs)

    def setParams(self,**kwargs):
        """ Sets the trainer's parameters. See SVMTrainer.setParams(). """
        for key, value in kwargs.items():
            if key in ("cost"):
                self._C_desc = value
                assert not ( isinstance(self.module, MCSVMOneAgainstAll) and isinstance(value, dict) )
            elif key in ("tc", "trainer_class"):
                self._trainer_class = value
            elif key in ("verbose", "ver", "v"):
                self._verbosity = value
        self._sub_kwargs = kwargs


    def setData(self,dataset):
        """ Set training data by supplying an instance of SupervisedDataSet. """
        self.module._setData(dataset)

    def train(self):
        """ Trains the MCSVM module.

            MCSVM creates a sub module for each possible pair of classes.
            This method creates a regular SVMTrainer for each of these sub modules,
            and calls its train method.
        """
        X       = self.module._X
        Y       = self.module._Y
        classes = self.module._classes


        sub_modules = self.module.getSubModules()
        for sub_module in sub_modules:
            sub_trainer = self._trainer_class(sub_module, **self._sub_kwargs)
            if self._verbosity > 0:
                print "\n=== Training for classes:",sub_module.getClass(0),sub_module.getClass(1) # zzzz geht nicht fuer oneagainstall
            sub_trainer.train()
            if self._verbosity > 0:
                print "Finished training at step :", sub_trainer.stepno


    def trainEpochs(self,epochs=0):
        X       = self.module._X
        Y       = self.module._Y
        classes = self.module._classes

        sub_modules = self.module.getSubModules()
        for sub_module in sub_modules:
            sub_trainer = SVMTrainer(sub_module, **self._sub_kwargs)
            if self._verbosity > 0:
                print "\n=== Training for classes:",sub_module.getClass(0),sub_module.getClass(1)
            sub_trainer.trainEpochs(epochs)










