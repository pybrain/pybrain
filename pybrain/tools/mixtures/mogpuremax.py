__author__ = 'Frank Sehnke, sehnke@in.tum.de'


from scipy import random, zeros, ones, exp, sqrt, cos, log

stND = zeros(1000)
for i in range(1000):
    x = -4.0 + float(i) * 8.0 / 1000.0
    stND[i] = 1.0 / 2.51 * exp(-0.5 * (x) ** 2)


# FIXME: different class name?
class MixtureOfGaussians:
    def __init__(self, numOGaus=10, alphaA=0.02, alphaM=0.02, alphaS=0.02):
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
        for i in range(self.numOGaus):
            self.mue[i] = self.distRange * float(i) * self.distFakt + self.rangeMin
        self.baseline = 0.0
        self.best = 0.000001

    def getStND(self, x, mue=0.0, sig=1.0):
        x = (x - mue) / sig
        if abs(x) >= 4.0: return 0.000000001
        x = int((x + 4.0) / 8.0 * 1000)
        return stND[x] / sig

    #get mixture for point x
    def getGaus(self, alpha, mue, sigma, x):
        dens = zeros(self.numOGaus)
        for g in range(self.numOGaus):
            dens[g] = self.getStND(x, mue[g], sigma[g])
        return sum(self.sigmo(alpha) * dens)

    def drawSample(self):
        sum = 0.0
        rndFakt = random.random()
        for g in range(self.numOGaus):
            sum += self.sigmo(self.alpha[g])
            if rndFakt < sum:
                if self.sigma[g] < self.minSig: self.sigma[g] = self.minSig
                x = random.normal(self.mue[g], self.sigma[g])
                break
        return x

    def learn(self, x, y):
        #learning overalls
        norm = zeros(self.numOGaus)
        nOver = self.getGaus(self.alpha, self.mue, self.sigma, x)
        for g in range(self.numOGaus):
            norm[g] = self.getStND(x, self.mue[g], self.sigma[g]) / nOver
        self.baseline = 0.99 * self.baseline + 0.01 * y
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

    def getSample(self):
        sampleX = self.drawSample()
        return sampleX

if __name__ == '__main__':
    m = MixtureOfGaussians()
    for i in range(10000):
        x = m.getSample()
        n = x / m.distRange * 10.0
        if abs(n) > 5.0: n = 0.0
        y = (20.0 + n ** 2 - 10.0 * cos(2.0 * 3.1416 * n)) / 55.0 + random.normal(0, 0.2) #one dimensional rastrigin
        m.learn(x, y)
    print(m.alpha)
    print(m.mue)
    print(m.sigma)

