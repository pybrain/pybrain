__author__ = 'Daan Wierstra and Tom Schaul'

from blackboxoptimizer import BlackBoxOptimizer
from numpy.random import randn, multivariate_normal
from random import random, randint
from scipy import ones, zeros, mat, eye, dot, argmax, average, array, size, exp, sign, power, sqrt, var, pi, log, reshape
from scipy.linalg import pinv2, norm, det, inv
from pybrain.tools.functions import safeExp

# TODO: linear dependence of the logalphas in phi - problem???

# TODO: take over selection
# TODO: take over KL-adaptarion
# TODO: take over x-replacement
# TODO: re-introduce basealpha? maybe with a tanh transform?
# TODO: maybe split gradients for each center, to keep lambda tractable
# TODO: learning rate adaptation
# TODO: investigate the use of black magic
# TODO: kosher determinant positivation


# flag determining if the nasty redundnat alpha column disappears (then it's -1)
# this might not work with all variations of the algorithm
schwupps = 0

class NaturalEvolutionStrategies(BlackBoxOptimizer):
    """ do the optimization using natural fitness gradients """
    
    # --- parameters for the algorithm ---
    lambd = None  # batch size
    mu = 1        # number of centers
    lr = 0.05   # learning rate
    lrSigma = None  # specific learning rate for the sigmas
    
    # one-fith-rule and varations
    onefifth = False
    onefifthAdaptation = 1.5
    advancedfifth = False
    dsigma = None
    fifthProportion = 0.2 
    fifthReset = True
    
    lrPeters = False
    momentum = None
    
    importanceSampling = False
    slidingbatch = False
    
    
    #ridge regression instead of svd pinv2
    ridge = False
    ridgeconstant = 0.000001
    
    
    # initial sigmas
    initialsearchrange = 1.0
    perturbedInitSigma = False
    initSigmaCoeff = 0.3    
    initSigmaRandCoeff = 0.01
    
    # fitness ranking/smoothing
    ranking = None
    gini = 0.1
    giniPlusX = 1
    giniScale = 5
    bilinearFactor = 20
    
    # RPROP
    rprop = False
    rpropMinUpdate = 1e-20
    rpropMaxUpdate = 1e3
    rpropInitDelta = 1-6
    etaPlus = 1.8
    etaMin = 0.1        
    rpropUseGradient = False
    
    # positivation of sigma diagonal
    epsilon = 1e-7
    positivationK = 0.5
    minDiagonal = 0
    
    # --- execution parameters ---
    maxgens = 1e5 # maximal nb of generations 
    verbose = False
    plotsymbol = '-'
    silent = False
    returnall = False # return all xs and sigmas
    
    minimize = False
    
    
    def __init__(self, f, **parameters):
        BlackBoxOptimizer.__init__(self, f, **parameters)
        n = self.xdim        
        
        # internal executution variables
        self.generation = 0  # current generation
        self.fevals = 0      # nb of evaluations
        self.verybestfit = None # best fitness achieved
        self.verybestx = None   # with this x
        
        # determine batch size
        minlambd = 1 + self.mu*(1+n+n*n)
        if not self.lambd:
            self.lambd = minlambd
        if not self.silent and self.lambd < minlambd:
            print "Warning! Underconstrained linear regression"
        
        if not self.lrSigma:
            self.lrSigma = self.lr 
            
            
        self.genfitnesses = []
                        
        self.alpha = ones(self.mu)/self.mu  # relative probabilities of drawing from each of the mu centers
        self.basealpha = ones(self.mu)/self.mu
        # list of the mu current centers      
        if self.x0 == None:
            self.x = [self.initialsearchrange * mat(randn(n,1))]
        else:
            self.x0 = reshape(mat(self.x0), (n,1))            
            self.x = [self.x0]
            
        for i in range(self.mu - 1):
            # CHECKME: should the centers not be different initially?
            self.x.append(self.x[0].copy())  
        self.fx = zeros(self.mu)            # function values for x         
        self.factorSigma = [self.genInitSigmaFactor()]                 # the cholesky(??) factor version of sigma              
        for dummy in range(self.mu - 1): 
            self.factorSigma.append(self.factorSigma[-1]*1.1)
        self.sigma = []                     # the list of the mu current covariance matrices        
        for dummy in range(self.mu): 
            self.sigma.append(self.factorSigma[dummy].T*self.factorSigma[dummy])             
        self.zs = [None]*self.lambd         # most recent batch of lambd samples drawn           
        self.chosenCenter = [None]*self.lambd   # the center chosen for drawing z        
        self.R = mat(zeros((self.lambd, 1)))    # function values for zs        
        self.w = None                       # the vector with the computed updates for all parameters
        self.rellhood = zeros(self.lambd)    # vector containing rel. likelihood under current policy

        # a special matrix, containing the log-derivatives    
        self.phi = mat(ones((self.lambd, 1 + schwupps + self.mu*(n*n+n+1))))
                 
        # iRPROP+ learning rate multiplier gamma for every generation
        self.delta = []
        self.delta.append(self.rpropInitDelta * ones(1 + (self.mu*(n*n+n+1))))
        # store old parameters
        self.wStored = []
        self.rpropPerformance = []    
        self.oldParams = []
        
        # used for the one-fith-rule
        self.lrMultiplier = 1.    
        self.blackmagic = 1.        
        if self.advancedfifth and not self.dsigma:
            self.dsigma = float(n+8)/4 # CMA heuristic
            
        if self.returnall:
            self.allsigmas = []
            self.allxs = []
            for i in range(self.mu):
                self.allsigmas.append(self.sigma[i].copy())
                self.allxs.append(self.x[i].copy())
        
    def optimize(self):  
        #assert self.minimize == False
        if self.tfun != None: self.tfun.reset()
        while self.generation < self.maxgens:
            detbefore = det(self.sigma[0])
            #xc = self.x[0].copy()
            #sc = self.sigma[0].copy()
            if self.slidingbatch:
                if self.generation > 0:
                    self.oneSlidingGeneration()
                else:
                    self.oneGeneration(False)
            else:
                self.oneGeneration()
            detafter = det(self.sigma[0])
            #if False: #detafter < 1e-6 or detafter>2.0:
            #    print detbefore
            #    print detafter
            #    print sc
            #    print xc
            #    print self.sigma[0]
            #    print self.x[0]
            #if detb > 1.0:
            #    print self.sigma[0]
            self.generation += 1
            if self.fevals + self.lambd > self.maxEvals:
                break
            if not self.checkStability():
                break
            if self.verbose:
                self.printCurrentValues()
            if not self.silent:
                print 'Generation', self.generation, '- fitness:', max(self.fx), 
                if self.verbose:
                    detbefore, detafter
                else:
                    print
            self.genfitnesses.append(max(self.fx))
            if self.verbose:
                print self.alpha
                print [m.T for m in self.x]
            
            #if self.generation > 12:
            #    if self.genfitnesses[-1] > self.genfitnesses[-10]:
            #        self.lr *= 1.1
            #        self.lrSigma *= 1.1
            #    else:
            #        self.lr *= 0.5
            #        self.lrSigma *= 0.5
            #print self.sigma[0]
            #print self.factorSigma[0]
            #self.sigma[0] = mat(diag(diag(self.sigma[0])))
            #self.factorSigma[0] = mat(diag(diag(self.factorSigma[0])))
            #print self.sigma[0]
            #print -self.verybestfit, self.stopPrecision
            #if -self.verybestfit < self.stopPrecision:
            if self.stoppingCriterion():
                break
            if self.returnall:
                # TODO: make this cleaner, also for mu > 1
                # this is now fragged up
                self.allxs.append(array(self.x[0]).flatten())
                self.allsigmas.append(self.sigma[0].copy())
        
        if not self.silent:
            print self.fevals, 'evaluations.'    
            print 'Best overall fitness found', self.verybestfit      
        if self.returnall:
            return self.verybestx, self.allxs, self.allsigmas
        else:   
            return self.verybestx
        
    def stoppingCriterion(self):
        return self.verybestfit >= self.stopPrecision
    
    def oneGeneration(self, update = True):
        """ execute one generation of the algorithm """
        # evaluate at the centers
        for m in range(self.mu):
            self.fx[m] = self.evaluateAt(self.x[m])            
        
        # generate new samples   
        for k in range(self.lambd):
            self.oneSample(k)
            
        # some optional adaptations
        if self.fifthReset:
            self.lrMultiplier = 1.        
        if self.onefifth:
            if self.percentageBetter() >= self.fifthProportion:
                self.lrMultiplier *= self.onefifthAdaptation
                #self.factorSigma[0] *= self.onefifthAdaptation
                #self.lr *= self.onefifthAdaptation
                #self.lrSigma *= self.onefifthAdaptation
            else:
                #self.lr /= 2*self.onefifthAdaptation
                #self.lrSigma /= self.onefifthAdaptation
                self.lrMultiplier /= self.onefifthAdaptation
                #self.factorSigma[0] /= 2*self.onefifthAdaptation                    
        elif self.advancedfifth:
            # a more fancy version of the onefifth-rule            
            chi = self.computeChi()            
            self.lrMultiplier *= exp(1/self.dsigma * (self.avgDistanceOfBest(self.fifthProportion) - chi)/chi)        
        self.blackmagic = self.lrMultiplier
        
        # produce the new update vector
        if not self.ridge:
            self.w = pinv2(self.phi)*self.rankFitness()      
        else:
            self.w = (self.phi.T * self.phi + self.ridgeconstant*eye(self.phi.shape[1])).I * self.phi.T * self.rankFitness() 
            
        # one fifth rule comes in here:
        self.w[self.mu-1+self.mu*self.xdim:-1] *= self.lrMultiplier
        
        if self.lrPeters:
            rvec = self.rankFitness()
            relR = array(rvec) - average(rvec)*ones((self.lambd,1))
            # phino is phi with no 1 at end
            phino = mat(ones((self.phi.shape[0], self.phi.shape[1] - 1)))
            for i in range(self.lambd):
                for j in range(self.phi.shape[1] - 1):
                    phino[i,j] = self.phi[i,j]
            grad = array(dot(phino.T, relR))
            fisher = array(dot(phino.T, phino))
            newlr = sqrt(inv(dot(dot(grad.T,inv(fisher)),grad))[0,0])
        
            self.w *= newlr
            #print "lrPeters:", newlr
        
        if self.rprop:  
            self.rpropUpdate(self.w)
        else:
            # learning rates come in here:
            self.w[0:self.mu+schwupps+self.mu*self.xdim] *= self.lr
            self.w[self.mu+schwupps+self.mu*self.xdim:-1] *= self.lrSigma
            if update:
                self.updateVariables(self.w)
            
    def oneSlidingGeneration(self):
        """ execute one sliding generation of the algorithm """
        # evaluate at the centers
        for m in range(self.mu):
            self.fx[m] = self.evaluateAt(self.x[m])            
        
        # CHECKME: maybe do this for every sample?
        
        if self.lrPeters:
            rvec = self.rankFitness()
            relR = array(rvec) - average(rvec)*ones((self.lambd,1))
            # phino is phi without the 1 at the end
            phino = mat(ones((self.phi.shape[0], self.phi.shape[1] - 1)))
            for i in range(self.lambd):
                for j in range(self.phi.shape[1] - 1):
                    phino[i,j] = self.phi[i,j]
            grad = array(dot(phino.T, relR))
            fisher = array(dot(phino.T, phino))
            newlr = sqrt(inv(dot(dot(grad.T,inv(fisher)),grad))[0,0])
        else:
            newlr = 1.0
        
            #print "lrPeters:", newlr
            
        # some optional adaptations
        if self.fifthReset:
            self.lrMultiplier = 1.        
        if self.onefifth:
            if self.percentageBetter() > self.fifthProportion:
                self.lrMultiplier *= self.onefifthAdaptation
            else:
                self.lrMultiplier /= self.onefifthAdaptation                    
        elif self.advancedfifth:
            # a more fancy version of the onefifth-rule            
            chi = self.computeChi()            
            self.lrMultiplier *= exp(1/self.dsigma * (self.avgDistanceOfBest(self.fifthProportion) - chi)/chi)        
        self.blackmagic = self.lrMultiplier
        
        # generate one new sample
        for k in range(self.lambd):
            self.oneSample(k)            
            
            fitnesses = self.rankFitness()
            
            phino = self.phi.copy()
            # importance sampling
            # TODO for all mu
            if self.importanceSampling:
                for k2 in range(self.lambd):
                    importance = self.multivariateNormalPdf(self.zs[k2], self.x[0], self.sigma[0])/self.rellhood[k2]
                    #print importance,
                    phino[k2] *= importance
                    fitnesses[k2] *= importance
                
            # produce the new update vector
            if not self.ridge:
                self.w = pinv2(phino)*fitnesses
                #print self.w[:,0]
                #self.w = solve(phino,fitnesses)
            else:   
                self.w = (self.phi.T * self.phi + self.ridgeconstant*eye(self.phi.shape[1])).I * self.phi.T * self.rankFitness() 
                
            # one fifth rule comes in here:
            self.w[self.mu+self.mu*self.xdim:-1] *= self.lrMultiplier
                
                
            if self.rprop:  
                self.rpropUpdate(self.w)
            else:
                # learning rates come in here:
                self.w *= newlr
                self.w[0:self.mu+self.mu*self.xdim] *= self.lr
                self.w[self.mu+self.mu*self.xdim:-1] *= self.lrSigma
                self.updateVariables(self.w)
            #print self.alpha
            
            
    def oneSample(self, k):
        """ produce one new sample and update phi correspondingly """
        thesum = 0.0
        for i in range(self.mu):
            thesum += exp(self.basealpha[i])
        for i in range(self.mu):
            self.alpha[i] = exp(self.basealpha[i])/thesum
        choosem = self.chooseCenter()
        self.chosenCenter[k] = choosem
        z = mat(multivariate_normal(array(self.x[choosem]).flatten(), self.sigma[choosem])).T
        self.zs[k] = z
        self.R[k] = self.evaluateAt(z)
        # TODO make for all mu
        if self.importanceSampling:
            self.rellhood[k] = self.multivariateNormalPdf(z, self.x[0], self.sigma[0])
        logderivbasealpha = zeros((self.mu, 1))
        logderivx = zeros((self.mu, self.xdim))
        logderivfactorsigma = zeros((self.mu, self.xdim, self.xdim))
        for m in range(self.mu):
            self.sigma[m] = dot(self.factorSigma[m].T,self.factorSigma[m])
            if self.mu > 1:
                relresponsibility = (self.alpha[m] * self.multivariateNormalPdf(z, self.x[m], self.sigma[m]) / 
                                 sum(map(lambda mm: self.alpha[mm]*self.multivariateNormalPdf(z, self.x[mm], self.sigma[mm]), range(self.mu))))
                #print 'relres', relresponsibility
            else:
                relresponsibility = 1.0
            if self.mu > 1:
                logderivbasealpha[m] = relresponsibility * (1.0 - self.alpha[m])
            else:
                logderivbasealpha[m] = 0.0
            logderivx[m] = relresponsibility * (self.sigma[m].I * (z - self.x[m])).flatten()                
            A = 0.5 * self.sigma[m].I * (z - self.x[m]) * (z - self.x[m]).T * self.sigma[m].I - 0.5 * self.sigma[m].I
            logderivsigma_m = self.blackmagic * relresponsibility * A#0.5 * (A + diag(diag(A)))  #* 2.0
            logderivfactorsigma[m] = self.factorSigma[m]*(logderivsigma_m + logderivsigma_m.T)
        #print 'logalpha', logderivbasealpha.flatten(), self.alpha, sum(logderivbasealpha)
        tmp = self.combineParams(logderivbasealpha, logderivx, logderivfactorsigma)
        self.phi[k] = tmp
                
    def updateVariables(self, vanillaupdatevector):
        """ update the internal variables, using the provided vector of updates """
        n = self.xdim
        
#        maxupdate = updatevector[0,0]
#        for i in range(updatevector.shape[0]):
#            if abs(updatevector[i,0]) > maxupdate:
#                maxupdate = abs(updatevector[i,0])
#        if maxupdate > 0.1:
#            updatevector *= 0.1/maxupdate
             
        updatevector = vanillaupdatevector.copy()
        if self.momentum:
            updatevector = vanillaupdatevector.copy()
            if self.generation > 1:
                updatevector += self.momentum*self.momentumVector
            self.momentumVector = updatevector.copy()
        #print "bf", self.basealpha
        if self.generation > 1:
            self.basealpha[:] += array(updatevector[0:self.mu]).flatten()
        #print "af", self.basealpha
        self.alpha = self.transformAlphas(self.basealpha)
        for i in range(self.mu):
            if self.alpha[i] < 0.05:
                #print 'XXX before', self.basealpha, self.alpha
                bestbase = max(self.alpha)
                bestindex = argmax(self.alpha)
                self.basealpha[i] = log((self.alpha[i] + bestbase)/2.0)
                self.basealpha[bestindex]  = log((self.alpha[i] + bestbase)/2.0)
                self.x[i] = self.x[bestindex].copy()
                self.factorSigma[i] = self.factorSigma[bestindex] * 1.2
                self.alpha = self.transformAlphas(self.basealpha)
                #print 'XXX after', self.basealpha
                
        offset = self.mu+schwupps+self.mu*n                
        for m in range(self.mu):
            deltax = mat(updatevector[self.mu+schwupps+m*n:self.mu+schwupps+(m+1)*n]).reshape(n, 1)            
            deltafactorsigma = mat(updatevector[offset+m*n*n:offset+(m+1)*n*n]).reshape(n,n) 
            #self.triagularReshape(updatevector[offset+m*n*n:offset+(m+1)*n*n]) 
            self.x[m] = self.x[m]+deltax
            self.factorSigma[m] += deltafactorsigma
            #self.factorSigma[m] = self.positivizeDiagonal(self.factorSigma[m])
            self.sigma[m] = dot(self.factorSigma[m].T,self.factorSigma[m])        
        
    def rpropUpdate(self, w):
        """ edit the update vector according to the rprop mechanism. """
        n = self.xdim
        self.wStored.append(w.copy())
        self.rpropPerformance.append(self.fx[0])
        self.oldParams.append(self.combineParams(self.alpha, self.x, self.factorSigma))            
        if self.generation > 0: 
            neww = zeros(len(w))
            self.delta.append(zeros((self.mu*(n*(n+1)/2+n+1))))            
            for i in range(len(w)-1):
                self.delta[self.generation][i] = self.delta[self.generation - 1][i]
                assert len(self.wStored[self.generation]) == len(self.wStored[self.generation-1])
                if self.wStored[self.generation][i] * self.wStored[self.generation-1][i] > 0.0:
                    self.delta[self.generation][i] = min(self.delta[self.generation-1][i] * self.etaPlus, self.rpropMaxUpdate)
                    if self.rpropUseGradient:
                        neww[i] = self.wStored[self.generation][i] * self.delta[self.generation][i]
                    else:
                        neww[i] = sign(self.wStored[self.generation][i]) * self.delta[self.generation][i]
                elif self.wStored[self.generation][i] * self.wStored[self.generation-1][i] < 0.0:
                    self.delta[self.generation][i] = max(self.delta[self.generation - 1][i] * self.etaMin, self.rpropMinUpdate)
                    if self.rpropPerformance[self.generation] < self.rpropPerformance[self.generation - 1]:                    
                        # undo the last update
                        neww[i] = self.oldParams[self.generation-1][i] - self.oldParams[self.generation][i]                        
                    self.wStored[self.generation][i] = 0.0
                elif self.wStored[self.generation][i] * self.wStored[self.generation - 1][i] == 0.0:
                    if self.rpropUseGradient:
                        neww[i] = self.wStored[self.generation][i] * self.delta[self.generation][i]
                    else:
                        neww[i] = sign(self.wStored[self.generation][i]) * self.delta[self.generation][i]
            self.updateVariables(neww)              
            
    def rankFitness(self):
        """ possibly produce a ranking or another transformation on the fitness values """
        if self.ranking == 'linear':
            return mat(self.rankedFitness(self.R)).T
        elif self.ranking == 'normal':
            return mat(self.normalizedRankedFitness(self.R)).T
        elif self.ranking == 'smooth':
            return mat(self.smoothSelectiveRanking(self.R)).T
        elif self.ranking == 'bilinear':
            return mat(self.bilinearRanking(self.R)).T
        else:
            return self.R       
    
    def genInitSigmaFactor(self):
        """ depending on the algorithm settings, we start out with in identity matrix, or perturb it """
        if self.perturbedInitSigma:
            res = mat(eye(self.xdim)*self.initSigmaCoeff+randn(self.xdim, self.xdim)*self.initSigmaRandCoeff)            
        else:
            res = mat(eye(self.xdim)*self.initSigmaCoeff)
        return res 
            

    def chooseCenter(self):
        """ randomly draw an index, according to the probabilities in alpha """
        x = random()
        s = 0
        for i, val in enumerate(self.alpha):
            s += val
            if x <= s: return i
        print "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"
        print self.alpha
        print
        return randint(0, self.mu - 1)
            
    def evaluateAt(self, z):        
        """ evaluate the function """
        res = self.targetfun(array(z).flatten())
        if self.minimize:
            res = -res
        self.fevals += 1
        if self.verybestfit == None or res > self.verybestfit:
            self.verybestfit = res
            self.verybestx = z  
        return res      
               
    def percentageBetter(self):
        """ determine what proportion of samples gave a better fitness than their center """
        better = 0.
        for k in range(self.lambd):
            if self.R[k] > self.fx[self.chosenCenter[k]]:
                better += 1
        return better/self.lambd
    
    def combineParams(self, alpha, x, factorsigma):
        """ fill an array with the given values, eliminating redundancy, and appending a 1"""
        n = self.xdim        
        res = ones(1+schwupps+self.mu*(n*n+n+1))            
        # the last value can be derived from the other ones, because their sum has to be 1
        res[0:self.mu+schwupps] = alpha.flatten().copy()
        offset = self.mu+schwupps+self.mu*n
        for m in range(self.mu):
            res[self.mu+schwupps+n*m:self.mu+schwupps+n*(m+1)] = x[m].flatten()
            # sigma contains redundant information because it is symetric
            #tmp = self.getLowerTriagularValues(factorsigma[m])
            res[offset:offset+n*n] = factorsigma[m].flatten()
            offset += n*n
        return res
            
    def checkStability(self):
        for m in range(self.mu):
            if (max(abs(array(self.x[m]).flatten())) > 1e20 or
                max(abs(array(self.sigma[m]).flatten())) > 1e40):            
                print 'Numerical explosion: abort.'
                print 'Evaluations', self.fevals
                print 'Alpha', self.alpha
                print 'X', self.x
                print 'Sigma', self.sigma#Olist.append(NaturalFitnessGradient(RosenbrockFunction(5, xopt = [10]*5), lr = 0.02, lrSigma = 0.005, lambd = 120, ranking = 'smooth', maxEvals = 20000))

                return False
        return True
    
    def computeChi(self, evals = 100):
        """ compute an estimate of the distance from the centers to the generated points """
        # CHECKME: correct handling of multiple centers?
        s = 0
        for dummy in range(evals):
            m = self.chooseCenter()
            z = mat(multivariate_normal(array(self.x[m]).flatten(), self.sigma[m])).T            
            s += norm(self.x[m] - z)
        return s/evals
    
    def avgDistanceOfGood(self, prop = 0.2):
        """ compute the average distance of the top fitness proportion of the samples to their centers """
        avgdistgood = 0
        for i, r in enumerate(self.rankedFitness(self.R)):
            if r > (self.lambd-1) * (1-prop):
                avgdistgood += norm(self.x[self.chosenCenter[i]]-self.zs[i])
        return avgdistgood / ((self.lambd-1) * prop)
    
    def printCurrentValues(self):
        # TODO: maybe make it more readable
        print 'Alpha', self.alpha
        print 'x', self.x
        print 'sigma', self.sigma
        print 'W', self.w
        print 'Phi', self.phi                                        
        
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
        res = zeros(self.lambd)
        for i in range(len(ranks)):
            res[i] = ranks[i]*self.giniPlusX + self.lambd*self.giniScale * smoothup(ranks[i]/float(self.lambd-1))
        return res
    
    def normalizedRankedFitness(self, R):
        return array((R - R.mean())/sqrt(var(R))).flatten()
    
    def bilinearRanking(self, R):
        ranks = self.rankedFitness(R)
        res = zeros(size(R))
        transitionpoint = 4*len(ranks)/5
        for i in range(len(ranks)):
            if ranks[i] < transitionpoint:
                res[i] = ranks[i]
            else:
                res[i] = ranks[i]+(ranks[i]-transitionpoint)*self.bilinearFactor
        return res    
    
    def multivariateNormalPdf(self, z, x, sigma):
        assert z.shape[1] == 1 and x.shape[1] == 1    
        tmp = -0.5 * ((z-x).T * sigma.I * (z-x))[0,0]
        # CHECKME: hacked explosion security - might be done smoothlier?
        if tmp > 100:   
            print "WRONG", tmp
        res = power(2*pi,-self.xdim/2.) * (1/sqrt(det(sigma))) * exp(tmp)
        #print res
        return res    
    
    def transformAlphas(self, base):
        #print 'base', base
        return safeExp(base)/sum(safeExp(base))