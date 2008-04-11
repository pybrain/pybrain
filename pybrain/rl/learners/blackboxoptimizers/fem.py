__author__ = 'Daan Wierstra and Tom Schaul'

from scipy import pi, power, exp, sqrt, dot, rand, ones, eye, zeros, outer, reshape, var
from scipy.linalg import inv, det
from numpy.random import multivariate_normal
from numpy import array, size, multiply

from pybrain.utilities import drawIndex
from blackboxoptimizer import BlackBoxOptimizer



class FEM(BlackBoxOptimizer):
    """ Fitness expectation-maximization"""
    
    batchsize = 200 #a.k.a: lambda
    numberOfCenters = 1  #a.k.a: k
    rangemins = None
    rangemaxs = None
    initCovariances = None
    bilinearFactor = 20
    gini = 0.25
    giniPlusX = 1
    giniScale = 5
    unlawfulExploration = 1.2
    alternativeUpdates = False
    slidingBatches = False
    
    def __init__(self, f, **args):
        BlackBoxOptimizer.__init__(self, f, **args)
        self.alphas = ones(self.numberOfCenters)/self.numberOfCenters
        self.mus = []
        self.sigmas = []

        self.tau = 1.
        if self.rangemins == None:
            self.rangemins = -ones(self.xdim)
        if self.rangemaxs == None:
            self.rangemaxs = ones(self.xdim)
        if self.initCovariances == None:
            self.initCovariances = eye(self.xdim)
            
        for dummy in range(self.numberOfCenters):
            self.mus.append(rand(self.xdim) * (self.rangemaxs-self.rangemins) + self.rangemins)
            self.sigmas.append(dot(eye(self.xdim), self.initCovariances))
            
    def optimize(self):
        if self.slidingBatches:
            self.optimizeWithSlidingBatch()
            return
        generation = 0
        self.allsamples = []
        while True:
            # produce samples
            samples = []
            densities = zeros((self.batchsize, self.numberOfCenters))
            fitnesses = zeros(self.batchsize)
            for k in range(self.batchsize):
                chosenOne = drawIndex(self.alphas, True)
                samples.append(multivariate_normal(self.mus[chosenOne], self.unlawfulExploration * self.sigmas[chosenOne]))
            
                # attribute weightings to all samples
                for c in range(self.numberOfCenters):
                    densities[k, c] = self.alphas[c] * self.multivariateNormalPdf(reshape(samples[-1], (self.xdim, 1)), 
                                                                 reshape(self.mus[c], (self.xdim, 1)), 
                                                                 self.unlawfulExploration * self.sigmas[c])
                # sample-wise normalization
                densities[k,:] /= sum(densities[k,:])

                fitnesses[k] = self.targetfun(samples[-1])
            
            # determine (transformed) fitnesses
            transformedFitnesses = self.shapingFunction(fitnesses)
            #self.updateTau(fitnesses, transformedFitnesses)
            
            weightings = multiply(outer(transformedFitnesses, ones(self.numberOfCenters)), densities)
                
            for c in range(self.numberOfCenters):
                # update alpha
                self.alphas[c] = sum(weightings[:,c])
                
                #update mu
                newMu = zeros(self.xdim)
                for i in range(self.batchsize):
                    newMu += weightings[i,c] * samples[i]
                newMu /= sum(weightings[:,c])
                if not self.alternativeUpdates or generation%2==0:
                    self.mus[c] = newMu
                
                #update sigma
                newSigma = zeros((self.xdim, self.xdim))
                for i in range(self.batchsize):
                    dif = -self.mus[c]+samples[i]
                    newSigma += weightings[i,c] * outer(dif, dif) 
                newSigma /= sum(weightings[:,c])
                #newSigma *= self.unlawfulExploration
                if not self.alternativeUpdates or generation%2==1:
                    self.sigmas[c] = newSigma #self.sigmas[c]
                
            # nomalize alphas
            self.alphas /= sum(self.alphas)
            
            self.allsamples.extend(samples)
            
            generation += 1
            print 'gen: ', generation
            print 'alphas: ', self.alphas
            #print 'mus:', self.mus
            #print 'sigmas:', self.sigmas
            print 'min,max: ',max(fitnesses), min(fitnesses)
            print
            
            if len(self.allsamples)> self.maxEvals:
                break
    
    
    def optimizeWithSlidingBatch(self):
        generation = 0
        totalsamples = 0
        self.allsamples = []
        samples = range(self.batchsize)
        densities = zeros((self.batchsize, self.numberOfCenters))
        fitnesses = zeros(self.batchsize)

        while True:
            # produce samples
            for k in range(self.batchsize):
                chosenOne = drawIndex(self.alphas, True)
                #print self.mus[chosenOne], self.sigmas[chosenOne]
                samples[k] = multivariate_normal(self.mus[chosenOne], self.sigmas[chosenOne])

                fitnesses[k] = self.targetfun(samples[k])
            
                totalsamples += 1
                
                if totalsamples < self.batchsize:
                    continue
                            
                for i in range(self.batchsize):
                    # attribute weightings to all samples
                    for c in range(self.numberOfCenters):
                        densities[i, c] = self.alphas[c] * self.multivariateNormalPdf(reshape(samples[i], (self.xdim, 1)), 
                                                                     reshape(self.mus[c], (self.xdim, 1)), 
                                                                     self.sigmas[c])
                    # sample-wise normalization
                    densities[i,:] /= sum(densities[i,:])

                # determine (transformed) fitnesses
                transformedFitnesses = self.shapingFunction(fitnesses)
                #self.updateTau(fitnesses, transformedFitnesses)
            
                weightings = multiply(outer(transformedFitnesses, ones(self.numberOfCenters)), densities)
                
                for c in range(self.numberOfCenters):
                    # update alpha
                    self.alphas[c] = sum(weightings[:,c])
                
                    #update mu
                    newMu = zeros(self.xdim)
                    for i in range(self.batchsize):
                        newMu += weightings[i,c] * samples[i]
                    newMu /= sum(weightings[:,c])
                    if not self.alternativeUpdates or generation%2==0:
                        self.mus[c] = newMu
                
                    #update sigma
                    newSigma = zeros((self.xdim, self.xdim))
                    for i in range(self.batchsize):
                        dif = -self.mus[c]+samples[i]
                        newSigma += weightings[i,c] * outer(dif, dif) 
                    newSigma /= sum(weightings[:,c])
                    newSigma *= self.unlawfulExploration
                    if not self.alternativeUpdates or generation%2==1:
                        self.sigmas[c] = newSigma #self.sigmas[c]
                
                # nomalize alphas
                self.alphas /= sum(self.alphas)
                
            generation += 1
            print 'gen: ', generation
            print 'alphas: ', self.alphas
            print 'mus:', self.mus
            print 'sigmas:', self.sigmas
            print 'min,max: ',max(fitnesses), min(fitnesses)
            print
        
    
    def multivariateNormalPdf(self, z, x, sigma):
        assert z.shape[1] == 1 and x.shape[1] == 1    
        tmp = -0.5 * dot(dot((z-x).T, inv(sigma)), (z-x))[0,0]
        res = (1./power(2.0*pi,self.xdim/2.)) * (1./sqrt(det(sigma))) * exp(tmp)
        return res   
    
    
    def shapingFunction(self, R):
        return self.smoothSelectiveRanking(R)#self.bilinearRanking(R)#exp(self.tau * R)        
    
    def updateTau(self, R, U):
        self.tau = sum(U)/dot((R - self.task.minReward), U)
        
    def rankedFitness(self, R):
        """ produce a linear ranking of the fitnesses in R. """        
        l = sorted(list(enumerate(R)), cmp = lambda a,b: cmp(a[1],b[1]))
        l = sorted(list(enumerate(l)), cmp = lambda a,b: cmp(a[1],b[1]))
        return array(map(lambda (r, dummy): r, l))

    # TODO: put this outside
    def smoothSelectiveRanking(self, R):
        """ a smooth ranking function that gives more importance to examples with better fitness. """
        def smoothup(x):
            """ produces a mapping from [0,1] to [0,1], with a specific gini coefficient. """
            return power(x, 2/self.gini-1)
        ranks = self.rankedFitness(R)
        res = zeros(self.batchsize)
        for i in range(len(ranks)):
            res[i] = ranks[i]*self.giniPlusX + self.batchsize*self.giniScale * smoothup(ranks[i]/float(self.batchsize-1))
        return res

    def normalizedRankedFitness(self, R):
        return array((R - R.mean())/sqrt(var(R))).flatten()

    def bilinearRanking(self, R):
        ranks = self.rankedFitness(R)
        res = zeros(size(R))
        transitionpoint = 4*len(ranks)/5
        kill = 0#len(ranks)/2
        for i in range(len(ranks)):
            if ranks[i] < transitionpoint:
                if ranks[i] >= kill:
                    res[i] = ranks[i]
                else:
                    res[i] = 0.0
            else:
                res[i] = ranks[i]+(ranks[i]-transitionpoint)*self.bilinearFactor
        return res