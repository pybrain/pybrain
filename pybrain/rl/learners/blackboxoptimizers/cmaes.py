__author__ = 'Tom Schaul, tom@idsia.ch; Sun Yi, yi@idsia.ch'

from numpy import floor, log, eye, zeros, array, sqrt, sum, mat, dot
from numpy import exp, triu, diag, matrix, power, ravel
from numpy.linalg import eig, norm
from numpy.random import randn

from blackboxoptimizer import BlackBoxOptimizer

# 2009-04-05, Major revise: get rid of scipy matrix

class CMAES(BlackBoxOptimizer):
    """ CMA-ES: Evolution Strategy with Covariance Matrix Adaptation for
    nonlinear function minimization.
    This code is a close transcription of the provided matlab code.
    """

    minimize = True
    stopPrecision = 1e-6

    online = False
    keepCenterHistory = False

    # Additional parameters for importance mixing
    lambd = None        # override CMA heuristics for batch size
    forceUpdate = 0.01  # refresh rate

    def _importanceMixing(self, N, lambd, xmean, sigma, B, D,
                          xmean0, sigma0, B0, D0, arz, arx, arfitness):
        """
        Performing importance mixing based on old samples.
        """
        # The real covariance matrix (sigma**2) * dot(dot(B,D),dot(B,D).T)

        # Things required to compute the probability
        c = -0.5 * sum(log(diag(sigma*D)**2))
        c0 = -0.5 * sum(log(diag(sigma0*D0)**2))

        invA = dot(diag(1/diag(sigma*D)), B.T)
        invA0 = dot(diag(1/diag(sigma0*D0)), B0.T)

        # two auxiliary functions for computing probability
        def prob(x, L):
            ctr = tile(xmean.reshape(N,1), (1,L))
            return c - 0.5*sum(dot(invA, x-ctr)**2, 0)

        def prob0(x, L):
            ctr0 = tile(xmean0.reshape(N,1), (1,L))
            return c0 - 0.5*sum(dot(invA0, x-ctr0)**2, 0)

        # first step, forward
        pr0 = prob0(arx, lambd)
        pr = prob(arx, lambd)
        p = minimum(1, exp(pr-pr0) * (1-forceUpdate))
        acpt = rand(1,lambd) < p
        t = filter(lambda i: acpt[i], xrange(lambd))

        arx[:,xrange(len(t))] = arx[:,t]
        arz[:,xrange(len(t))] = arz[:,t]
        arfitness[xrange(len(t))] = arfitness[t]
        nreq = lambd - len(t)   # number of new samples required

        # second step, backward
        req = nreq
        while req > 0:
            splz = randn(N, lambd)
            splx = tile(xmean.reshape(N,1),(1,lambd)) + sigma * dot(dot(B,D),splz)
            pr = prob(splx, lambd)
            pr0 = prob0(splx, lambd0)
            p = maximum(forceUpdate, 1 - exp(pr0-pr))
            acpt = rand(1,lambd) < p
            t = filter(lambda i: acpt[i], xrange(lambd))
            splz = splz[:,t]
            splx = splx[:,t]
            if len(t) >= req:
                arz[:,xrange(lambd-req,lambd)] = splz[:,xrange(req)]
                arx[:,xrange(lambd-req,lambd)] = splx[:,xrange(req)]
                break
            else:
                arz[:,xrange(lambd-req,lambd-req+len(t))] = splz
                arx[:,xrange(lambd-req,lambd-req+len(t))] = splx
                req -= len(t)
        for i in xrange(lambd-nreq,lambd): arfitness[i] = self.evaluator(arx[:,i])
        return arz, arx, arfitness, nreq

    def _batchLearn(self, maxSteps = None):
        N = self.xdim
        strfitnessfct = self.evaluator
        xmean = array(self.x0)
        #xmean = mat(self.x0).reshape(self.xdim, 1)
        sigma = 0.5         # coordinate wise standard deviation (step size)

        if self.keepCenterHistory:
            self.allCenters = []

        # Strategy parameter setting: Selection
        if self.lambd:
            lambd = self.lambd
        else:
            lambd = int(4+floor(3*log(N)))  # population size, offspring number
        mu = int(floor(lambd/2))        # number of parents/points for recombination
        #weights = log(mu+1)-log(matrix(range(1,mu+1))).T # muXone array for weighted recombination
        weights = log(mu+1)-log(array(xrange(1,mu+1)))      # use array
        weights = weights/sum(weights)     # normalize recombination weights array
        mueff=sum(weights)**2/sum(power(weights, 2)) # variance-effective size of mu

        # Strategy parameter setting: Adaptation
        cc = 4/float(N+4)               # time constant for cumulation for covariance matrix
        cs = (mueff+2)/(N+mueff+3)      # t-const for cumulation for sigma control
        mucov = mueff                   # size of mu used for calculating learning rate ccov
        ccov = ((1/mucov) * 2/(N+1.4)**2 + (1-1/mucov) *   # learning rate for
                 ((2*mueff-1)/((N+2)**2+2*mueff)))         # covariance matrix
        damps = 1 + 2*max(0, sqrt((mueff-1)/(N+1))-1) + cs
        # damping for sigma usually close to 1 former damp == damps/cs

        # Initialize dynamic (internal) strategy parameters and constants
        pc = zeros((N,1))
        ps = zeros((N,1))                   # evolution paths for C and sigma
        B = eye(N,N)                      # B defines the coordinate system
        D = eye(N,N)                      # diagonal matrix D defines the scaling
        C = dot(dot(B,D),dot(B,D).T)                # covariance matrix
        chiN=N**0.5*(1-1./(4.*N)+1/(21.*N**2))
        # expectation of ||N(0,I)|| == norm(randn(N,1))

        # -------------------- Generation Loop --------------------------------
        counteval = 0 # the next 40 lines contain the 20 lines of interesting code
        #arfitness = mat(zeros((lambd,1)))
        #arx = mat(zeros((N,lambd)))
        arfitness = zeros(lambd)
        arx = zeros(N, lambd)

        while counteval+lambd <= maxSteps:
            # !!! This part is modified for importance mixing
            if counteval == 0:
                # Generate and evaluate lambda offspring
                arz = randn(N,lambd)
                arx = tile(xmean,(1,lambd)) + sigma * dot(dot(B,D), arz)
                for k in xrange(lambd):
                    arfitness[k] = self.evaluator(arx[:,k])
                    counteval += 1
            else:
                arz, arx, arfitness, neweval = self._importanceMixing(N, lambd, xmean, sigma, B, D,
                          xmean0, sigma0, B0, D0, arz, arx, arfitness)
                xmean0, sigma0, B0, D0 = xmean, sigma, B, D
                counteval += neweval

            # Sort by fitness and compute weighted mean into xmean
            arfitness, arindex = sorti(arfitness)  # minimization
            arz = arz[:,arindex]
            arx = arx[:,arindex]
            xmean = dot(arx[:,xrange(mu)], weights)
            zmean = dot(arz[:,xrange(mu)], weights)

            if self.keepCenterHistory:
                self.allCenters.append(xmean)

            # Cumulation: Update evolution paths
            ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * dot(B,zmean)                 # Eq. (4)
            hsig = norm(ps)/sqrt(1-(1-cs)**(2*counteval/float(lambd)))/chiN < 1.4 + 2./(N+1)
            pc = (1-cc)*pc + hsig * sqrt(cc*(2-cc)*mueff) * dot(dot(B,D),zmean)    # Eq. (2)

            # Adapt covariance matrix C
            C = ((1-ccov) * C                    # regard old matrix      % Eq. (3)
                 + ccov * (1/mucov) * (outer(pc,pc) #pc*pc.T   # plus rank one update
                 + (1-hsig) * cc*(2-cc) * C)
                 + ccov * (1-1/mucov)            # plus rank mu update
                   * dot(dot(arx[:,xrange(mu)],diag(weights)),arx[:,xrange(mu)].T))

            # Adapt step size sigma
            sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1))             # Eq. (5)

            # Update B and D from C
            # This is O(N^3). When strategy internal CPU-time is critical, the
            # next three lines should be executed only every (alpha/ccov/N)-th
            # iteration, where alpha is e.g. between 0.1 and 10
            C=triu(C)+triu(C,1).T # enforce symmetry
            Ev, B = eig(C)          # eigen decomposition, B==normalized eigenvectors
            D = sqrt(diag(Ev))      #diag(ravel(sqrt(Ev))) # D contains standard deviations now

            if self.verbose:
                print counteval, ': ', arfitness[0]

            self.bestEvaluable = arx[:,0]
            self.bestEvaluation = arfitness[0]
            self.notify()

            # Break, if fitness is good enough
            if arfitness[0] <= self.desiredEvaluation:
                if self.verbose:
                    print "Stopped since fitness supposedly good enough", arfitness[0], self.desiredEvaluation
                break
            # or convergence is reached
            if abs((arfitness[0]-arfitness[-1])/arfitness[0]+arfitness[-1]) <= self.stopPrecision:
                if self.verbose:
                    print "Converged."
                break

def sorti(vect):
    """ sort, but also return the indices-changes """
    tmp = sorted(map(lambda (x,y): (y,x), enumerate(ravel(vect))))
    res1 = map(lambda x: x[0], tmp)
    res2 = map(lambda x: int(x[1]), tmp)
    return res1, res2
