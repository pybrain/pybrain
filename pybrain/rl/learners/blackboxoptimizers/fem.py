__author__ = 'Daan Wierstra and Tom Schaul'

from scipy import pi, power, exp, sqrt, dot, rand, ones, eye, zeros, outer, reshape, var
from scipy.linalg import inv, det, sqrtm, norm
from numpy.random import multivariate_normal
from numpy import array, size, multiply, average, argmax

from pybrain.utilities import drawIndex
from blackboxoptimizer import BlackBoxOptimizer



class FEM(BlackBoxOptimizer):
    """ Fitness expectation-maximization"""
    
    #batchsize = 3000#15*15*1 #a.k.a: lambda
    #numberOfCenters = 1  #a.k.a: k
    #rangemins = None
    #rangemaxs = None
    #initCovariances = None
    #bilinearFactor = 20
    #gini = 0.2 #0.02
    #giniPlusX = 0.01  #0.15
    #giniScale = 5.0
    #unlawfulExploration = 1.17 #1.15
    #alternativeUpdates = False
    #onlineLearning = True
    #temperature = 10.0
    #learningrate = 0.01
    
    batchsize = 50#15*15*1 #a.k.a: lambda
    numberOfCenters = 1  #a.k.a: k
    rangemins = None
    rangemaxs = None
    initCovariances = None
    bilinearFactor = 20
    gini = 0.15 #0.02
    giniPlusX = 0.15  #0.15
    giniScale = 5.0
    unlawfulExploration = 1.0 #1.15
    alternativeUpdates = False
    onlineLearning = True
    temperature = 10.0
    maxupdate = 0.3
    elitist = False 
    superelitist = False
    evalMus = True
    
    def __init__(self, evaluator, evaluable, **parameters):
        BlackBoxOptimizer.__init__(self, evaluator, evaluable, **parameters)
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

    def _batchLearn(self, maxSteps):
        self.allsamples = []
        self.muevals = []
        
        if self.verbose:
            print
            print "==================="        
            print "Fitness Expectation Maximization"
            print "==================="
            if self.onlineLearning:
                print 'ONLINE'
            else:
                print "OFFLINE"
            print "batchsize:", self.batchsize
            print "gini:", self.gini
            print "unlawfulExploration:", self.unlawfulExploration
            print "numberOfCenters:", self.numberOfCenters
            print "giniPlusX:", self.giniPlusX
            if self.onlineLearning: # CHECKME 
                print "maxupdate", self.maxupdate
                print "elitist", self.elitist
                print "superelitist", self.superelitist
            print
        
        if self.onlineLearning:
            self.optimizeOnline(maxSteps)
        else:
            self.optimize(maxSteps)
            
    def optimize(self, maxSteps):
        generation = 0
        while len(self.allsamples) +len(self.muevals) + self.batchsize <= maxSteps:
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
                                                                 self.sigmas[c])
                # sample-wise normalization
                densities[k,:] /= sum(densities[k,:])
                
                
                # importance sampling for unlawfulexploration
                for c in range(self.numberOfCenters):
                    densities[k, c] *= self.multivariateNormalPdf(reshape(samples[k], (self.xdim, 1)), 
                                                                reshape(self.mus[c], (self.xdim, 1)), 
                                                                self.unlawfulExploration * self.sigmas[c]) / \
                                       self.multivariateNormalPdf(reshape(samples[k], (self.xdim, 1)), 
                                                                reshape(self.mus[c], (self.xdim, 1)),
                                                                self.sigmas[c])

                fitnesses[k] = self.evaluator(samples[-1])
            
            # determine (transformed) fitnesses
            transformedFitnesses = self.shapingFunction(fitnesses)
            
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
                #newSigma *= 1.0#self.unlawfulExploration
                
                if not self.alternativeUpdates or generation%2==1:
                    self.sigmas[c] = newSigma #self.sigmas[c]
                
            # nomalize alphas
            self.alphas /= sum(self.alphas)
                
            generation += 1
            self.allsamples.extend(samples)
            
            if self.evalMus == True:
                for m in self.mus:
                    me = self.evaluator(m)
                    if me > self.bestEvaluation:
                        self.bestEvaluation, self.bestEvaluable = me, m
                    self.muevals.append(me)
                
            if self.verbose:
                print 'gen: ', generation, 'max,min,avg: ',max(fitnesses), min(fitnesses), average(fitnesses)
                if self.evalMus:
                    print '    mu-fitness(es):', self.muevals[-len(self.mus):]
                
            if max(fitnesses)> self.bestEvaluation:
                bestindex = argmax(fitnesses)
                self.bestEvaluation, self.bestEvaluable = fitnesses[bestindex], samples[bestindex]
            
            if max(fitnesses) >= self.desiredEvaluation:
                break
                
            if len(self.allsamples)+len(self.muevals) >= maxSteps:
                break


    def optimizeOnline(self, maxSteps):
        generation = 0
        samples = range(self.batchsize)
        densities = zeros((self.batchsize, self.numberOfCenters))
        fitnesses = zeros(self.batchsize)
        totalsamples = 0

        while True:
            for k in range(self.batchsize):
                chosenOne = drawIndex(self.alphas, True)
                samples[k] = multivariate_normal(self.mus[chosenOne], self.unlawfulExploration * self.sigmas[chosenOne])
                totalsamples += 1

                # attribute weightings to all samples
                for c in range(self.numberOfCenters):
                    densities[k, c] = self.alphas[c] * self.multivariateNormalPdf(reshape(samples[k], (self.xdim, 1)), 
                                                                 reshape(self.mus[c], (self.xdim, 1)), 
                                                                 self.sigmas[c])
                # sample-wise normalization
                densities[k,:] /= sum(densities[k,:])


                # importance sampling for unlawfulexploration
                for c in range(self.numberOfCenters):
                    densities[k, c] *= self.multivariateNormalPdf(reshape(samples[k], (self.xdim, 1)), 
                                                                reshape(self.mus[c], (self.xdim, 1)), 
                                                                self.unlawfulExploration * self.sigmas[c]) / \
                                       self.multivariateNormalPdf(reshape(samples[k], (self.xdim, 1)), 
                                                                reshape(self.mus[c], (self.xdim, 1)),
                                                                self.sigmas[c])


                fitnesses[k] = self.evaluator(samples[k])
                if totalsamples == 1:
                    bestfitnessever = fitnesses[k]
                    bestsampleever = samples[k].copy()
                if fitnesses[k] > bestfitnessever:
                    bestfitnessever = fitnesses[k]
                    bestsampleever = samples[k].copy()

                # determine (transformed) fitnesses
                transformedFitnesses = self.shapingFunction(fitnesses)
                transformedFitnesses /= max(transformedFitnesses)

                weightings = multiply(outer(transformedFitnesses, ones(self.numberOfCenters)), densities)
                weightings = self.maxupdate * weightings / sum(weightings)

                if generation == 0 and totalsamples == self.batchsize:
                    self.mus[0] = bestsampleever.copy()
                if generation >= 1:
                    for c in range(self.numberOfCenters):
                        lr = weightings[k,c]
                                            
                        self.alphas[c] = (1.0-lr)*self.alphas[c] + lr
                        
                        #update mu
                        newMu = zeros(self.xdim)
                        newMu = (1.0-weightings[k,c]) * self.mus[c] + weightings[k,c] * samples[k]
                        
                        # in case of batch-elitism
                        if self.elitist:
                            if max(fitnesses) == fitnesses[k]:
                                newMu = samples[k].copy()
                            else:
                                newMu = self.mus[c].copy()
                        # real elitism
                        if self.superelitist:
                            newMu = bestsampleever.copy()
                            
                        self.mus[c] = newMu

                        #update sigma
                        newSigma = zeros((self.xdim, self.xdim))
                        dif = -self.mus[c]+samples[k]
                        newSigma = (1.0-lr) * self.sigmas[c] + lr * outer(dif, dif) 
                        self.sigmas[c] = newSigma 

                    # nomalize alphas
                    self.alphas /= sum(self.alphas)
                
                if totalsamples >= maxSteps:
                    break

            generation += 1
            self.allsamples.extend(samples)
            if self.evalMus == True:
                for m in self.mus:
                    me = self.evaluator(m)
                    if me > self.bestEvaluation:
                        self.bestEvaluation, self.bestEvaluable = me, m
                    self.muevals.append(me)
                    
            if self.verbose:
                print 'gen: ', generation, 'max,min,avg: ',max(fitnesses), min(fitnesses), average(fitnesses)
                if self.evalMus:
                    print '    mu-fitness(es):', self.muevals[-len(self.mus):]
                
            
            if max(fitnesses)> self.bestEvaluation:
                bestindex = argmax(fitnesses)
                self.bestEvaluation, self.bestEvaluable = fitnesses[bestindex], samples[bestindex]
            
            if max(fitnesses) >= self.desiredEvaluation:
                break
            
            if len(self.allsamples)+len(self.muevals) >= maxSteps:
                break

            
    def multivariateNormalPdf(self, z, x, sigma):
        assert z.shape[1] == 1 and x.shape[1] == 1    
        tmp = -0.5 * dot(dot((z-x).T, inv(sigma)), (z-x))[0,0]
        res = (1./power(2.0*pi,self.xdim/2.)) * (1./sqrt(det(sigma))) * exp(tmp)
        return res       
    
    def shapingFunction(self, R):
        #return self.exponentialRanking(R)#self.smoothSelectiveRanking(R)#self.bilinearRanking(R)#exp(self.tau * R)        
        return self.smoothSelectiveRanking(R)#self.bilinearRanking(R)#exp(self.tau * R)        
        #return self.simpleRanking(R)
    
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
        
    def exponentialRanking(self, R):
        res = zeros(self.batchsize)
        ranks = self.rankedFitness(R)
        for i in range(len(ranks)):
            res[i] = exp((ranks[i]/(self.batchsize-1.0)) * self.temperature)
        return res
        
    def simpleRanking(self, R):
        res = self.rankedFitness(R) / (self.batchsize - 1.0)
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