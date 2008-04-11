__author__ = 'Tom Schaul, tom@idsia.ch'

from numpy import floor, log, eye, zeros, array, sqrt, dot, sum, mat
from numpy import exp, triu, diag, matrix, power
from numpy.linalg import eig
from numpy.random import randn

from blackboxoptimizer import BlackBoxOptimizer


class CMAES(BlackBoxOptimizer):
    """ CMA-ES: Evolution Strategy with Covariance Matrix Adaptation for
    nonlinear function minimization. 
    This code is a close transcription of the provided matlab code.
    """
    
    plotsymbol = 'o'
    silent = True
    
    def optimize(self):    
        assert self.minimize == True
        if self.tfun != None: 
            self.tfun.reset()
        N = self.xdim
        strfitnessfct = self.targetfun
        xmean = mat(self.x0).reshape(self.xdim, 1)

        sigma = 0.5         # coordinate wise standard deviation (step size)
      
        # Strategy parameter setting: Selection  
        lambd = int(4+floor(3*log(N)))  # population size, offspring number
        mu = int(floor(lambd/2))        # number of parents/points for recombination
        weights = log(mu+1)-log(matrix(range(1,mu+1))).T # muXone array for weighted recombination
        
        weights = weights/sum(weights)     # normalize recombination weights array
        mueff=sum(weights)**2/sum(power(weights, 2)) # variance-effective size of mu
        
        # Strategy parameter setting: Adaptation
        cc = 4/float(N+4)               # time constant for cumulation for covariance matrix
        cs = (mueff+2)/(N+mueff+3)      # t-const for cumulation for sigma control
        mucov = mueff                   # size of mu used for calculating learning rate ccov
        ccov = ((1/mucov) * 2/(N+1.4)**2 + (1-1/mucov) *   # learning rate for
                 ((2*mueff-1)/((N+2)**2+2*mueff)))         # covariance matrix
        damps = 1 + 2*max(0, sqrt((mueff-1)/(N+1))-1) + cs # damping for sigma 
                                                         # usually close to 1
                                                         # former damp == damps/cs    
    
        # Initialize dynamic (internal) strategy parameters and constants
        pc = zeros((N,1))
        ps = zeros((N,1))                   # evolution paths for C and sigma
        B = eye(N,N)                      # B defines the coordinate system
        D = eye(N,N)                      # diagonal matrix D defines the scaling
        C = B*D*(B*D).T                   # covariance matrix
        chiN=N**0.5*(1-1./(4.*N)+1/(21.*N**2)) # expectation of 
                                          #   ||N(0,I)|| == norm(randn(N,1))
      
        # -------------------- Generation Loop --------------------------------
        counteval = 0 # the next 40 lines contain the 20 lines of interesting code 
        arfitness = mat(zeros((lambd,1)))
        arx = mat(zeros((N,lambd)))    
        while counteval < self.maxEvals:
        
            # Generate and evaluate lambda offspring
            arz = mat(randn(N,lambd))     # array of normally distributed mutation vectors
            for k in range(0,lambd):
                arx[:,k] = xmean + sigma * (B*D * arz[:,k])    # add mutation  % Eq. (1)            
                arfitness[k] = strfitnessfct(array(arx[:,k]).flatten())  # objective function call
                counteval = counteval+1
            
            # Sort by fitness and compute weighted mean into xmean
            arfitness, arindex = sorti(arfitness)  # minimization
            xmean = arx[:,arindex[0:mu]]*weights    # recombination, new mean value
            zmean = arz[:,arindex[0:mu]]*weights    # == sigma^-1*D^-1*B'*(xmean-xold)
            
            # Cumulation: Update evolution paths
            ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * (B * zmean)                 # Eq. (4)
            hsig = norm(ps)/sqrt(1-(1-cs)**(2*counteval/float(lambd)))/chiN < 1.4 + 2./(N+1)
            pc = ((1-cc)*pc + hsig * sqrt(cc*(2-cc)*mueff) * (B * D * zmean))    # Eq. (2)
        
            # Adapt covariance matrix C
            C = ((1-ccov) * C                    # regard old matrix      % Eq. (3)
                 + ccov * (1/mucov) * (pc*pc.T   # plus rank one update
                 + (1-hsig) * cc*(2-cc) * C)
                 + ccov * (1-1/mucov)            # plus rank mu update 
                   * (B*D*arz[:,arindex[0:mu]])
                   *  diag(flat(weights)) * (B*D*arz[:,arindex[0:mu]]).T)               
        
            # Adapt step size sigma
            sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1))             # Eq. (5)
            
            # Update B and D from C
            # This is O(N^3). When strategy internal CPU-time is critical, the
            # next three lines should be executed only every (alpha/ccov/N)-th
            # iteration, where alpha is e.g. between 0.1 and 10 
            C=triu(C)+triu(C,1).T # enforce symmetry
            Ev, B = eig(C)          # eigen decomposition, B==normalized eigenvectors
            D = diag(flat(sqrt(Ev))) # D contains standard deviations now
            
            if not self.silent:
                print counteval, ': ', arfitness[0]
            
            # Break, if fitness is good enough
            if arfitness[0] <= self.stopPrecision:
                break
            
        return arx[:, arindex[1]]
    
def norm(x):
    return sqrt(dot(flat(x),flat(x)))

def flat(m):
    return array(m).flatten(1)

def sorti(vect):
    """ sort, but also return the indices-changes """
    tmp = sorted(map(lambda (x,y): (y,x), enumerate(flat(vect))))
    res1 = map(lambda x: x[0], tmp)
    res2 = map(lambda x: int(x[1]), tmp)
    return res1, res2

    