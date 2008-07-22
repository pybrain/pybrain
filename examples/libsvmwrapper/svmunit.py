# $Id: svmunit.py 310 2008-05-05 10:07:43Z felder $
from pybrain.datasets import SupervisedDataSet
from svm import svm_model

class SVMUnit(object):
    """ This unit represents an SVM and is implemented through the LIBSVM Python interface. 
    $Id: svmunit.py 310 2008-05-05 10:07:43Z felder $ """
    
    def __init__(self, input = [], output = [0.0], model=None):
        self.reset()
        # CHECKME: really  in the constructor?
        self.input = input 
        # CHECKME: this looks dodgy too...
        self.output = output
        self.setModel(model)
    
    def reset(self):
        """ reset buffers and delete the SVM model """
        self.model = None
        
    def setModel(self, model):
        """ initialize the SVM model """
        self.model = model
        
    def loadModel(self, filename):
        """ read the SVM model description from a file """
        self.model = svm_model(filename)
        
    def saveModel(self, filename):
        """ read the SVM model description from a file """
        self.model.save(filename)
        
        
    def forwardPass(self, values=False, dataset=None):
        """ produce the output from the current input vector, or process a dataset """
        if isinstance(dataset, SupervisedDataSet):
            # process an entire dataset and return result as a vector
            out = []
            inp = dataset['input']
            for i in range(inp.shape[0]):
                self.input = inp[i,:]
                # carry out forward pass to get decision values for each class combo
                self.forwardPass(values=values)
                out.append(self.output) 
            return out
            
        if values=="class" or values==False:
            # predict the output class right away
            self.output = self.model.predict(self.input)
        elif values=='raw' or values==True:
            # return a dict of decision values for each one-on-one class 
            # combination (i,j)
            self.output = self.model.predict_values(self.input)
        else:  # values == "prob"
            # return probability (works only for multiclass!)
            self.output = self.model.predict_probability(self.input)
            
        
    def getNbClasses(self):
        """ return number of classes the current model uses """
        return self.model.get_nr_class()
    
