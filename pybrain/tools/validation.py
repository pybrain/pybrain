__author__ = 'Michael Isik'


from numpy.random import permutation
from numpy import array, array_split, apply_along_axis, concatenate, ones, dot, delete, append, zeros, argmax
import copy
from pybrain.datasets.importance import ImportanceDataSet
from pybrain.datasets.sequential import SequentialDataSet
from pybrain.datasets.supervised import SupervisedDataSet



class Validator(object):
    """ This class provides methods for the validation of calculated output
        values compared to their destined target values. It does
        not know anything about modules or other pybrain stuff. It just works
        on arrays, hence contains just the core calculations.

        The class has just classmethods, as it is used as kind of namespace
        instead of an object definition.
    """
    @classmethod
    def classificationPerformance(cls, output, target):
        """ Returns the hit rate of the outputs compared to the targets.

            :arg output: array of output values
            :arg target: array of target values
        """
        output = array(output)
        target = array(target)
        assert len(output) == len(target)
        n_correct = sum(output == target)
        return float(n_correct) / float(len(output))

    @classmethod
    def ESS(cls, output, target):
        """ Returns the explained sum of squares (ESS).

            :arg output: array of output values
            :arg target: array of target values
        """
        return sum((output - target) ** 2)

    @classmethod
    def MSE(cls, output, target, importance=None):
        """ Returns the mean squared error. The multidimensional arrays will get
            flattened in order to compare them.

            :arg output: array of output values
            :arg target: array of target values
            :key importance: each squared error will be multiplied with its
                corresponding importance value. After summing
                up these values, the result will be divided by the
                sum of all importance values for normalization
                purposes.
        """
        # assert equal shapes
        output = array(output)
        target = array(target)
        assert output.shape == target.shape
        if importance is not None:
            assert importance.shape == target.shape
            importance = importance.flatten()

        # flatten structures
        output = output.flatten()
        target = target.flatten()

        if importance is None:
            importance = ones(len(output))


        # calculate mse
        squared_error = (output - target) ** 2
        mse = dot(squared_error, importance) / sum(importance)


        return mse



class ClassificationHelper(object):
    """ This class provides helper methods for classification, like the
        conversion of one-of-many data to class indices data.

        The class has just classmethods, as it is used as kind of namespace
        instead of an object definition.
    """
    @classmethod
    def oneOfManyToClasses(cls, data):
        """ Converts data in one-of-many format to class indices format and
            and returns the result.

            :arg data: array of vectors, that are in the one-of-many format.
                         Each vector will be converted to the index of the
                         component with the maximum value.
        """
        return apply_along_axis(argmax, 1, data)





class SequenceHelper(object):
    """ This class provides helper methods for sequence handling.

        The class has just classmethods, as it is used as kind of namespace
        instead of an object definition.
    """
    @classmethod
    def getSequenceEnds(cls, dataset):
        """ Returns the indices of the last elements of the sequences stored
            inside dataset.

            :arg dataset: Must implement :class:`SequentialDataSet`
        """
        sequence_ends = delete(dataset.getField('sequence_index') - 1, 0)
        sequence_ends = append(sequence_ends, dataset.getLength() - 1)
#        print(sequence_ends; exit())
        sequence_ends = array(sequence_ends)
        return sequence_ends

    @classmethod
    def getSequenceStarts(cls, dataset):
        """ Returns the indices of the first elements of the sequences stored
            inside dataset.

            :arg dataset: Must implement :class:`SequentialDataSet`
        """
        return  list(dataset.getField('sequence_index'))

    @classmethod
    def getSequenceEndsImportance(cls, dataset):
        """ Returns the importance values of the last elements of the sequences
            stored inside dataset.

            :arg dataset: Must implement :class:`ImportanceDataSet`
        """
        importance = zeros(dataset.getLength())
        importance[cls.getSequenceEnds(dataset)] = 1.
        return importance





class ModuleValidator(object):
    """ This class provides methods for the validation of calculated output
        values compared to their destined target values. It especially handles
        pybrains modules and dataset classes.
        For the core calculations, the Validator class is used.

        The class has just classmethods, as it is used as kind of namespace
        instead of an object definition.
    """
    @classmethod
    def classificationPerformance(cls, module, dataset):
        """ Returns the hit rate of the module's output compared to the targets
            stored inside dataset.

            :arg module: Object of any subclass of pybrain's Module type
            :arg dataset: Dataset object at least containing the fields
                'input' and 'target' (for example SupervisedDataSet)
        """
        return ModuleValidator.validate(
            Validator.classificationPerformance,
            module,
            dataset)

    @classmethod
    def MSE(cls, module, dataset):
        """ Returns the mean squared error.

            :arg module: Object of any subclass of pybrain's Module type
            :arg dataset: Dataset object at least containing the fields
                'input' and 'target' (for example SupervisedDataSet)
        """
        return ModuleValidator.validate(
            Validator.MSE,
            module,
            dataset)


    @classmethod
    def validate(cls, valfunc, module, dataset):
        """ Abstract validate function, that is heavily used by this class.
            First, it calculates the module's output on the dataset.
            In advance, it compares the output to the target values of the dataset
            through the valfunc function and returns the result.

            :arg valfunc: A function expecting arrays for output, target and
                importance (optional). See Validator.MSE for an example.
            :arg module:  Object of any subclass of pybrain's Module type
            :arg dataset: Dataset object at least containing the fields
                'input' and 'target' (for example SupervisedDataSet)
        """
        target = dataset.getField('target')
        output = ModuleValidator.calculateModuleOutput(module, dataset)

        if isinstance(dataset, ImportanceDataSet):
            importance = dataset.getField('importance')
            return valfunc(output, target, importance)
        else:
            return valfunc(output, target)


    @classmethod
    def _calculateModuleOutputSequential(cls, module, dataset):
        """ Calculates the module's output on the dataset. Especially designed
            for datasets storing sequences.
            After a sequence is fed to the module, it has to be resetted.

            :arg dataset: Dataset object of type SequentialDataSet or subclass.
        """
        outputs = []
        for seq in dataset._provideSequences():
            module.reset()
            for i in range(len(seq)):
                output = module.activate(seq[i][0])
                outputs.append(output.copy())
        outputs = array(outputs)
        return outputs


    @classmethod
    def calculateModuleOutput(cls, module, dataset):
        """ Calculates the module's output on the dataset. Can be called with
            any type of dataset.

            :arg dataset: Any Dataset object containing an 'input' field.
        """
        if isinstance(dataset, SequentialDataSet) or isinstance(dataset, ImportanceDataSet):
            return cls._calculateModuleOutputSequential(module, dataset)
        else:
            module.reset()
            input = dataset.getField('input')
            output = array([module.activate(inp) for inp in input])
            return output





class CrossValidator(object):
    """ Class for crossvalidating data.
        An object of CrossValidator must be supplied with a trainer that contains
        a module and a dataset.
        Then the dataset ist shuffled and split up into n parts of equal length.

        A clone of the trainer and its module is made, and trained with n-1 parts
        of the split dataset. After training, the module is validated with
        the n'th part of the dataset that was not used during training.

        This is done for each possible combination of n-1 dataset pieces.
        The the mean of the calculated validation results will be returned.
    """
    def __init__(self, trainer, dataset, n_folds=5, valfunc=ModuleValidator.classificationPerformance, **kwargs):
        """ :arg trainer: Trainer containing a module to be trained
            :arg dataset: Dataset for training and testing
            :key n_folds: Number of pieces, the dataset will be splitted to
            :key valfunc: Validation function. Should expect a module and a dataset.
                            E.g. ModuleValidator.MSE()
            :key others: see setArgs() method
        """
        self._trainer = trainer
        self._dataset = dataset
        self._n_folds = n_folds
        self._calculatePerformance = valfunc
        self._max_epochs = None
        self.setArgs(**kwargs)

    def setArgs(self, **kwargs):
        """ Set the specified member variables.

        :key max_epochs: maximum number of epochs the trainer should train the module for.
        :key verbosity: set verbosity level
        """
        for key, value in list(kwargs.items()):
            if key in ("verbose", "ver", "v"):
                self._verbosity = value
            elif key in ("max_epochs"):
                self._max_epochs = value

    def validate(self):
        """ The main method of this class. It runs the crossvalidation process
            and returns the validation result (e.g. performance).
        """
        dataset = self._dataset
        trainer = self._trainer
        n_folds = self._n_folds
        l = dataset.getLength()
        inp = dataset.getField("input")
        tar = dataset.getField("target")
        indim = dataset.indim
        outdim = dataset.outdim
        assert l > n_folds

        perms = array_split(permutation(l), n_folds)

        perf = 0.
        for i in range(n_folds):
            # determine train indices
            train_perms_idxs = list(range(n_folds))
            train_perms_idxs.pop(i)
            temp_list = []
            for train_perms_idx in train_perms_idxs:
                temp_list.append(perms[ train_perms_idx ])
            train_idxs = concatenate(temp_list)

            # determine test indices
            test_idxs = perms[i]

            # train
            #print("training iteration", i)
            train_ds = SupervisedDataSet(indim, outdim)
            train_ds.setField("input"  , inp[train_idxs])
            train_ds.setField("target" , tar[train_idxs])
            trainer = copy.deepcopy(self._trainer)
            trainer.setData(train_ds)
            if not self._max_epochs:
                trainer.train()
            else:
                trainer.trainEpochs(self._max_epochs)

            # test
            #print("testing iteration", i)
            test_ds = SupervisedDataSet(indim, outdim)
            test_ds.setField("input"  , inp[test_idxs])
            test_ds.setField("target" , tar[test_idxs])
#            perf += self.getPerformance( trainer.module, dataset )
            perf += self._calculatePerformance(trainer.module, dataset)

        perf /= n_folds
        return perf

#    def getPerformance( self, module, dataset ):
#        inp    = dataset.getField("input")
#        tar    = dataset.getField("target")
#        indim  = module.indim
#        outdim = module.outdim

#        def forward(inp):
#            out = empty(outdim)
#            module._forwardImplementation(inp,out)
#            return out

#        out = apply_along_axis(forward,1,inp)
#        return self._calculatePerformance(out,tar)
#        return self._calculatePerformance(module, dataset)


    def _calculatePerformance(self, output, target):
        raise NotImplementedError()



def testOnSequenceData(module, dataset):
    """ Fetch targets and calculate the modules output on dataset.
    Output and target are in one-of-many format. The class for each sequence is
    determined by first summing the probabilities for each individual sample over
    the sequence, and then finding its maximum."""
    target = dataset.getField("target")
    output = ModuleValidator.calculateModuleOutput(module, dataset)

    # determine last indices of the sequences inside dataset
    ends = SequenceHelper.getSequenceEnds(dataset)
    ##format = "%d"*len(ends)
    summed_output = zeros(dataset.outdim)
    # class_output and class_target will store class labels instead of
    # one-of-many values
    class_output = []
    class_target = []
    for j in range(len(output)):
        # sum up the output values of one sequence
        summed_output += output[j]
#            print(j, output[j], " --> ", summed_output)
        # if we reached the end of the sequence
        if j in ends:
            # convert summed_output and target to class labels
            class_output.append(argmax(summed_output))
            class_target.append(argmax(target[j]))

            # reset the summed_output to zeros
            summed_output = zeros(dataset.outdim)

    ##print(format % tuple(class_output))
    ##print(format % tuple(class_target))

    class_output = array(class_output)
    class_target = array(class_target)
#    print(class_target)
#    print(class_output)
    return Validator.classificationPerformance(class_output, class_target)





