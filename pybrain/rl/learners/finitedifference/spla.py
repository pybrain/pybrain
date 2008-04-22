__author__ = 'Frank Sehnke, sehnke@in.tum.de'

from scipy import ones, zeros, mean, dot, ravel, sign
from scipy.linalg import pinv
from scipy import random
from pybrain.rl.learners.finitedifference.fd import FDLearner
from pybrain.rl.learners.finitedifference import FDBasic
from pybrain.auxiliary import GradientDescent

class SPLA(FDLearner):
    def __init__(self):
        # standard parameters
        self.alpha = 0.2 #Stepsize for parameter adaption
        self.alphaSig=0.085 #Stepsize for sigma adaption
        self.epsilon = 2.0 #Initial value of sigmas
        self.baseline=0.0 #Moving average baseline, used for sigma adaption
        self.best=-1000000.0 #TODO ersetzen durch -inf
        self.symCount=1.0 #Switch for symetric sampling
        self.gd = GradientDescent()
        self.gdSig = GradientDescent()

        
    def setModule(self, module):
        """Sets and initializes all module settings"""
        FDLearner.setModule(self, module)
        self.original = zeros(self.module.params.shape) #Stores the parameter set
        self.sigList=ones(self.module.params.shape) #Stores the list of standard deviations (sigmas)
        self.initSigmas()
        self.deltas=zeros(self.module.params.shape) #the parameter difference vector for exploration
        self.module._setParameters(self.original) #initializes the module parameter set to zeros
        self.gd.init(self.original)

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
            state, action, reward = self.ds.getSequence(seq)
            #add up the rewards of positive and negative perturbation role outs respectivly
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

        #some checks at the first learnign sequence
        if self.baseline==0.0: 
            self.baseline=self.reward
            fakt=0.0
            fakt2=0.0
            if seqLen/2 != float(seqLen)/2.0: print "ATTENTON!!! SPLA uses symetric sampling! Number of episodes per learning step must be even! (2 for deterministic settings, >2 for stochastic settings) A numer of episodes of ", seqLen, "is odd."
        else: 
            #calc the gradients
            fakt=(reward1-reward2)/(2.0*self.best-reward1-reward2) #gradient estimate alla SPSA but with liklihood gradient and normalization
            fakt2=(self.reward-self.baseline)/(self.best-self.baseline) #normalized sigma gradient with moving average baseline
        self.baseline=0.9*self.baseline+0.1*self.reward #update baseline
            
        # update parameters and sigmas
        #self.original=self.original+self.alpha*fakt*self.deltas #apply parameter update 
        self.original = self.gd(fakt*self.deltas)               
        if fakt2> 0.0: #for sigma adaption alg. follows only positive gradients
            #self.sigList=self.sigList+self.alphaSig*fakt2*(self.deltas*self.deltas-self.sigList*self.sigList)/self.sigList #apply sigma update
            self.sigList=self.gdSig(fakt2*(self.deltas*self.deltas-self.sigList*self.sigList)/self.sigList) #apply sigma update

        self.module.reset() # reset the module 
        self.genDifVect() #generate a new perturbation vector
