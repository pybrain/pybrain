__author__ = 'Frank Sehnke, sehnke@in.tum.de'

from scipy import ones, zeros, mean, dot, ravel, sign, random, sqrt
from time import sleep
from pybrain.rl.learners.finitedifference.fd import FDLearner
from pybrain.auxiliary import GradientDescent

#This class uses SPSA in general, but uses the likelihood gradient and a simpler exploration decay
class SimpleSPSA(FDLearner):
    def __init__(self):
        # standard parameters
        self.epsilon = 2.0 #Initial value of exploration size
        self.baseline=0.0 #Moving average baseline, used just for visualisation
        self.best=-1000000.0 #TODO ersetzen durch -inf
        self.symCount=1.0 #Switch for symetric sampling
        self.gd = GradientDescent()
        self.gamma=0.9995 #Exploration decay factor

    def setModule(self, module):
        """Sets and initializes all module settings"""
        FDLearner.setModule(self, module)
        self.original = zeros(self.module.params.shape) #Stores the parameter set
        self.module._setParameters(self.original) #initializes the module parameter set to zeros
        self.gd.init(self.original)
        self.numOParas=len(self.original)
        self.genDifVect()

    def genDifVect(self):
        # generates a uniform difference vector with the given epsilon
        self.deltas = (random.randint(0,2,self.numOParas)*2-1)*self.epsilon
        
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
            self.baseline=self.reward*0.99
            fakt=0.0
            if seqLen/2 != float(seqLen)/2.0: 
                print "ATTENTON!!! SPSA uses symetric sampling! Number of episodes per learning step must be even! (2 for deterministic settings, >2 for stochastic settings) A numer of episodes of ", seqLen, "is odd."
                while(True): sleep(1)
        else: 
            #calc the gradients
            if reward1!=reward2:
                #gradient estimate alla SPSA but with liklihood gradient and normalization (see also "update parameters")
                fakt=(reward1-reward2)/(2.0*self.best-reward1-reward2) 
            else: fakt=0.0
        self.baseline=0.9*self.baseline+0.1*self.reward #update baseline
            
        # update parameters
        # as a simplification we use alpha = alpha * epsilon**2 for decaying the stepsize instead of the usual use method from SPSA
        # resulting in the same update rule like for PGPE
        self.original = self.gd(fakt*self.epsilon*self.epsilon/self.deltas)
        # reduce epsilon by factor gamma
        # as another simplification we let the exploration just decay with gamma. 
        # Is similar to the decreasing exploration in SPSA but simpler.
        self.epsilon *= self.gamma
        self.module.reset() # reset the module 
        self.genDifVect() #generate a new perturbation vector
