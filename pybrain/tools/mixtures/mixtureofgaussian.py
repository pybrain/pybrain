__author__ = 'Frank Sehnke, sehnke@in.tum.de'


from scipy import random, zeros, ones, sqrt, exp, sin, cos, log

stND = zeros(1000)
for i in range(1000):
    x = -4.0 + float(i) * 8.0 / 1000.0
    stND[i] = 1.0 / 2.51 * exp(-0.5 * (x) ** 2)

class MixtureOfGaussians:
    def __init__(self, typ, numOGaus=10, alphaA=0.02, alphaM=0.02, alphaS=0.02):
        self.typ = typ
        self.alphaA = alphaA
        self.alphaM = alphaM
        self.alphaS = alphaS
        self.minSig = 0.000001
        self.numOGaus = numOGaus #Number of Gaussians
        self.rangeMin = -20.0
        self.rangeMax = 20.0
        self.epsilon = (self.rangeMax - self.rangeMin) / (sqrt(2.0) * float(self.numOGaus - 1)) #Initial value of sigmas

        self.propFakt = 1.0 / float(self.numOGaus)
        self.distFakt = 1.0 / float(self.numOGaus - 1)
        self.distRange = self.rangeMax - self.rangeMin

        self.sigma = ones(self.numOGaus)
        self.mue = zeros(self.numOGaus)
        self.alpha = ones(self.numOGaus)
        self.sigma *= self.epsilon
        self.alpha /= float(self.numOGaus)
        self.alpha = self.invSigmo(self.alpha)
        for i in range(self.numOGaus):
            self.mue[i] = self.distRange * float(i) * self.distFakt + self.rangeMin
        self.baseline = 0.0
        self.best = 0.000001

    def getStND(self, x, mue=0.0, sig=1.0):
        x = (x - mue) / sig
        if abs(x) >= 4.0: return 0.000000001
        x = int((x + 4.0) / 8.0 * 1000)
        return stND[x] / sig

    #generate complete mixture for plotting
    def plotGaussian(self, col, dm):
        from pylab import plot
        if dm == "max": scal = 1.0
        else: scal = 10.0
        ret = []
        xList = []
        for i in range(1000):
            x = float(i) * 1.5 * self.distRange / 1000.0 + self.rangeMin * 1.5
            xList.append(x)
            ret.append(self.getGaus(self.alpha, self.mue, self.sigma, x) * scal)
        plot(xList, ret, col)

    #get mixture for point x
    def getGaus(self, alpha, mue, sigma, x):
        sigmoA = self.sigmo(self.alpha)
        dens = zeros(self.numOGaus)
        for g in range(self.numOGaus):
            dens[g] = self.getStND(x, mue[g], sigma[g])
        return sum(sigmoA * dens)

    #reward functions for testing
    def testRewardFunction(self, x, typ, noise=0.000001):
        if typ == "growSin":
            return (sin((x - self.rangeMin) / 3.0) + 1.5 + x / self.distRange) / 4.0 + random.normal(0, noise)
        if typ == "rastrigin":
            n = x / self.distRange * 10.0
            if abs(n) > 5.0: n = 0.0
            # FIXME: imprecise reimplementation of the Rastrigin function that exists already
            # in rl/environments/functions...
            return (20.0 + n ** 2 - 10.0 * cos(2.0 * 3.1416 * n)) / 55.0 + random.normal(0, noise)
        if typ == "singleGaus":
            return self.getStND(x) + random.normal(0, noise)
        return 0.0

    def drawSample(self, dm):
        sum = 0.0
        rndFakt = random.random()
        if dm == "max":
            for g in range(self.numOGaus):
                sum += self.sigmo(self.alpha[g])
                if rndFakt < sum:
                    if self.sigma[g] < self.minSig: self.sigma[g] = self.minSig
                    x = random.normal(self.mue[g], self.sigma[g])
                    break
            return x
        if dm == "dist":
            return rndFakt * self.distRange + self.rangeMin
        return 0.0

    def learn(self, x, y, dm="max", typ="logLiklihood"):
        #learning overalls
        norm = zeros(self.numOGaus)
        nOver = self.getGaus(self.alpha, self.mue, self.sigma, x)
        for g in range(self.numOGaus):
            norm[g] = self.getStND(x, self.mue[g], self.sigma[g]) / nOver
        if dm == "max": self.baseline = 0.99 * self.baseline + 0.01 * y
        if y > self.best: self.best = y
        fakt = (y - self.baseline) / (self.best - self.baseline)
        if fakt < -1.0: fakt = -1.0


        #alpha learning
        sigmoA = self.sigmo(self.alpha)
        self.alpha += self.alphaA * self.propFakt * sigmoA * (1.0 - sigmoA) * fakt * norm #(1.0-sigmoA)*
        sigmoA = self.sigmo(self.alpha)
        sigmoA /= sum(sigmoA)
        self.alpha = self.invSigmo(sigmoA)

        #mue learning
        sigmoA = self.sigmo(self.alpha)
        self.mue += self.alphaM * fakt * (x - self.mue) * sigmoA * norm

        #sigma learning
        if fakt > 0.0:
            self.sigma += self.alphaS * fakt * ((x - self.mue) ** 2 - self.sigma ** 2) / self.sigma * sigmoA * norm

    def sigmo(self, a):
        return 1.0 / (1.0 + exp(-1.0 * a))

    def invSigmo(self, a):
        return - log(1.0 / a - 1.0)

    #plots the choosen reward function without noise
    def plotReward(self, col):
        from pylab import plot
        xList = []
        yList = []
        for i in range(1000):
            x = float(i) * 1.5 * self.distRange / 1000.0 + self.rangeMin * 1.5
            xList.append(x)
            yList.append(self.testRewardFunction(x, self.typ))
        plot(xList, yList, col)

    def getSample(self, dm="max"):
        sampleX = self.drawSample(dm)
        return sampleX

    def sample(self, wi, dm, learning="logLiklihood", noise=0.2, plt=True):
        if plt:
            self.plotGaussian('r', dm)
            self.plotReward('y')
        xList = []
        yList = []
        for i in range(wi):
            sampleX = self.getSample(dm)
            sampleY = self.testRewardFunction(sampleX, self.typ, noise)
            self.learn(sampleX, sampleY, dm, "logLiklihood")

            if plt:
                if i / 1 == float(i) / 1.0:
                    xList.append(sampleX)
                    yList.append(sampleY)
                if i == wi / 4:
                    self.plotGaussian('g', dm)
                if i == wi / 2:
                    self.plotGaussian('b', dm)

        if plt:
            from pylab import show, scatter, legend, axis
            self.plotGaussian('k', dm)
            scatter(xList, yList, 1)
            v = [-30.5, 30.5, -0.5, 1.5]
            axis(v)
            s4 = repr(wi / 4) + 'Sample'
            s2 = repr(wi / 2) + 'Sample'
            s1 = repr(wi) + 'Sample'
            legend(('meanReward', 'InitMixture', s4, s2, s1), loc=0, shadow=True)
            show()

if __name__ == '__main__':
    #m=MixtureOfGaussians("rastrigin", 20)
    #m.sample(10000, "dist")
    m = MixtureOfGaussians("rastrigin", 10)
    m.sample(10000, "max")

