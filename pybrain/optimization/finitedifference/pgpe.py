__author__ = 'Frank Sehnke, sehnke@in.tum.de, Tom Schaul'

from scipy import ones, random

from pybrain.auxiliary import GradientDescent
from fd import FiniteDifferences


# TODO: find better names for fakt and fakt2 variables!

class PGPE(FiniteDifferences):
    """ Policy Gradients with Parameter Exploration"""
    
    batchSize = 2
    epsilon = 2.0 #Initial value of sigmas
    wDecay = 0.001 #lasso weight decay (0 to deactivate)

    def _additionalInit(self):
        self.gdSig = GradientDescent()    
        self.sigList = ones(self.numParameters)*self.epsilon #Stores the list of standard deviations (sigmas)
        self.gdSig.init(self.sigList)
        self.baseline = None
        
    def perturbation(self):
        # generates a difference vector with the given standard deviations
        return random.normal(0., self.sigList)
            
    def _learnStep(self):
        """ calculates the gradient and executes a step in the direction
            of the gradient, scaled with a learning rate alpha. """
        deltas = self.perturbation()
        #reward of positive and negative perturbations
        reward1 = self._oneEvaluation(self.current + deltas)        
        reward2 = self._oneEvaluation(self.current - deltas)
        self.mreward=(reward1+reward2)/2.                
        if self.baseline is None: 
            # first learning step
            self.baseline=self.mreward
            fakt=0.
            fakt2=0.          
        else: 
            #calc the gradients
            if reward1!=reward2:
                #gradient estimate alla SPSA but with liklihood gradient and normalization
                fakt=(reward1-reward2)/(2.*self.bestEvaluation-reward1-reward2) 
            else: 
                fakt=0.
            #normalized sigma gradient with moving average baseline
            fakt2=(self.mreward-self.baseline)/(self.bestEvaluation-self.baseline)             
        #update baseline        
        self.baseline=0.9*self.baseline+0.1*self.mreward             
        # update parameters and sigmas
        self.current = self.gd(fakt*deltas-self.current*self.sigList*self.wDecay)   
        if fakt2> 0.: #for sigma adaption alg. follows only positive gradients
            self.sigList=self.gdSig(fakt2*(deltas*deltas-self.sigList*self.sigList)/self.sigList) #apply sigma update
        
        
        
        
class PGPENoSym(PGPE):
    """ Like normal PGPE but without symmetric sampling """
    
    def perturbate(self):
        """ perturb the parameters. """
        self.ds.append('deltas', self.deltas) #add perturbation to the data set
        self.module._setParameters(self.original + self.deltas) #set the actual perturbed parameters in module

    def learn(self):
        """ calculates the gradient and executes a step in the direction
            of the gradient, scaled with a learning rate alpha. """
        assert self.ds != None
        assert self.module != None
                       
        # calculate the gradient
        self.reward=0.0 #reward of positive perturbation
        seqLen=self.ds.getNumSequences() #number of sequences done for learning
        self.change=self.original.copy()*0.0
        self.sigChange=self.sigList.copy()*0.0
        for seq in range(seqLen):
            _, _, reward = self.ds.getSequence(seq)
            #add up the rewards of positive and negative perturbation role outs respectivly
            self.reward=sum(reward)

            #check if reward is the best observed up to now
            if self.reward > self.best: self.best= self.reward

            #some checks at the first learnign sequence
            if self.baseline==0.0: 
                self.baseline=self.reward/2.0
                fakt=0.0
            else: 
                #calc the gradients
                fakt=(self.reward-self.baseline)/(self.best-self.baseline) #normalized gradient with moving average baseline
            self.baseline=0.9*self.baseline+0.1*self.reward #update baseline
            
            # update parameters and sigmas
            self.change += self.gd(fakt*self.deltas)-self.original
            self.gd.init(self.original)               
            if fakt> 0.0: 
                self.sigChange+=self.gdSig(fakt*(self.deltas*self.deltas-self.sigList*self.sigList)/self.sigList)-self.sigList #apply sigma update
                self.gdSig.init(self.sigList)               
        self.original+=self.change
        self.gd.init(self.original)               
        self.sigList+=self.sigChange
        self.gdSig.init(self.sigList)               
        self.module.reset() # reset the module 
        self.genDifVect() #generate a new perturbation vector
