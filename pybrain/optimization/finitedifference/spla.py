__author__ = 'Frank Sehnke, sehnke@in.tum.de'

from scipy import ones, zeros, random
from time import sleep
from finitediff import FDLearner
from pybrain.auxiliary import GradientDescent

class SPLA(FDLearner):
    def __init__(self):
        # standard parameters
        self.epsilon = 2.0 #Initial value of sigmas
        self.baseline=0.0 #Moving average baseline, used for sigma adaption
        self.best=-1000000.0 #TODO ersetzen durch -inf
        self.symCount=1.0 #Switch for symetric sampling
        self.gd = GradientDescent()
        self.gdSig = GradientDescent()
        self.wDecay = 0.001 #lasso weight decay (0 to deactivate)

    def setModule(self, module):
        """Sets and initializes all module settings"""
        FDLearner.setModule(self, module)
        self.original = zeros(self.module.params.shape) #Stores the parameter set
        self.sigList=ones(self.module.params.shape) #Stores the list of standard deviations (sigmas)
        self.initSigmas()
        self.deltas=zeros(self.module.params.shape) #the parameter difference vector for exploration
        self.module._setParameters(self.original) #initializes the module parameter set to zeros
        self.gd.init(self.original)
        self.numOParas=len(self.original)

    def initSigmas(self):
        self.sigList*=self.epsilon #initialize sigmas to epsilon
        self.gdSig.init(self.sigList)
        
    def genDifVect(self):
        # generates a difference vector with the given standard deviations
        self.deltas=random.normal(0.0, self.sigList)
        
    def perturbate(self):
        """ perturb the parameters. """
        self.symCount*=-1.0 #change sign of perturbation
        self.ds.append('deltas', self.symCount*self.deltas) #add perturbation to the data set
        self.module._setParameters(self.original + self.symCount*self.deltas) #set the actual perturbed parameters in module

    def learn(self):
        """ calculates the gradient and executes a step in the direction
            of the gradient, scaled with a learning rate alpha. """
        assert self.ds != None
        assert self.module != None
                       
        # calculate the gradient
        reward1=0.0 #reward of positive perturbation
        reward2=0.0 #reward of negative perturbation
        sym=1.0 #perturbation switch
        seqLen=self.ds.getNumSequences() #number of sequences done for learning
        for seq in range(seqLen):
            sym*=-1.0
            _, _, reward = self.ds.getSequence(seq)
            #add up the rewards of positive and negative perturbation role outs respectively
            if sym==1.0: reward1+=sum(reward)
            else: reward2+=sum(reward)
        #normate rewards by seqLen 
        reward1/=float(seqLen)
        reward2/=float(seqLen)
        self.reward=(reward1+reward2)
        reward1*=2.0
        reward2*=2.0

        #check if reward is the best observed up to now
        if reward1 > self.best: self.best= reward1
        if reward2 > self.best: self.best= reward2

        #some checks at the first learning sequence
        if self.baseline==0.0: 
            self.baseline=self.reward/2.0
            fakt=0.0
            fakt2=0.0
            if seqLen/2 != float(seqLen)/2.0: 
                print "ATTENTON!!! SPLA uses symetric sampling! Number of episodes per learning step must be even! (2 for deterministic settings, >2 for stochastic settings) A numer of episodes of ", seqLen, "is odd."
                while(True): sleep(1)
        else: 
            #calc the gradients
            if reward1!=reward2:
                #gradient estimate alla SPSA but with liklihood gradient and normalization
                fakt=(reward1-reward2)/(2.0*self.best-reward1-reward2) 
            else: fakt=0.0
            #normalized sigma gradient with moving average baseline
            fakt2=(self.reward-self.baseline)/(self.best-self.baseline) 
        self.baseline=0.9*self.baseline+0.1*self.reward #update baseline
            
        # update parameters and sigmas
        self.original = self.gd(fakt*self.deltas-self.original*self.sigList*self.wDecay)   
        print abs(self.original).sum()/self.numOParas            
        if fakt2> 0.0: #for sigma adaption alg. follows only positive gradients
            self.sigList=self.gdSig(fakt2*(self.deltas*self.deltas-self.sigList*self.sigList)/self.sigList) #apply sigma update
        self.module.reset() # reset the module 
        self.genDifVect() #generate a new perturbation vector

class SPLANoSym(SPLA):
    #Like normal SPLA but without symetric sampling
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
