__author__ = "Martin Felder"
__version__ = '$Id: exampleRNN.py 1503 2008-09-13 15:25:06Z bayerj $'
try:
    from svm import svm_model
except ImportError:
    raise ImportError("Cannot find LIBSVM installation. Make sure svm.py and svmc.* are in the PYTHONPATH!")

class SVMUnit(object):
    """ This unit represents an Support Vector Machine and is implemented through the
    LIBSVM Python interface. It functions somewhat like a Model or a Network, but combining
    it with other PyBrain Models is currently discouraged. Its main function is to compare
    against feed-forward network classifications. You cannot get or set model parameters, but
    you can load and save the entire model in LIBSVM format. Sequential data and backward
    passes are not supported. See the corresponding example code for usage. """

    def __init__(self, indim=0, outdim=0, model=None):
        """ Initializes as empty module.

        If `model` is given, initialize using this LIBSVM model instead. `indim`
        and `outdim` are for compatibility only, and ignored."""
        self.reset()
        # set some dummy input/ouput dimensions - these become obsolete when
        # the SVM is initialized
        self.indim = 0
        self.outdim = 0
        self.setModel(model)

    def reset(self):
        """ Reset input and output buffers """
        self.input = None
        self.output = None

    def setModel(self, model):
        """ Set the SVM model. """
        self.model = model

    def loadModel(self, filename):
        """ Read the SVM model description from a file """
        self.model = svm_model(filename)

    def saveModel(self, filename):
        """ Save the SVM model description from a file """
        self.model.save(filename)

    def forwardPass(self, values=False):
        """ Produce the output from the current input vector, or process a
        dataset.

        If `values` is False or 'class', output is set to the number of the
        predicted class. If True or 'raw', produces decision values instead.
        These are stored in a dictionary for multi-class SVM. If `prob`, class
        probabilities are produced. This works only if probability option was
        set for SVM training."""
        if values == "class" or values == False:
            # predict the output class right away
            self.output = self.model.predict(self.input)
        elif values == 'raw' or values == True:
            # return a dict of decision values for each one-on-one class
            # combination (i,j)
            self.output = self.model.predict_values(self.input)
        else:  # values == "prob"
            # return probability (works only for multiclass!)
            self.output = self.model.predict_probability(self.input)

    def activateOnDataset(self, dataset, values=False):
        """ Run the module's forward pass on the given dataset unconditionally
        and return the output as a list.

        :arg dataset: A non-sequential supervised data set.
        :key values: Passed trough to forwardPass() method."""
        out = []
        inp = dataset['input']
        for i in range(inp.shape[0]):
            self.input = inp[i, :]
            # carry out forward pass to get decision values for each class combo
            self.forwardPass(values=values)
            out.append(self.output)
        return out

    def getNbClasses(self):
        """ return number of classes the current model uses """
        return self.model.get_nr_class()


