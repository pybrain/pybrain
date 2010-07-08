__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import zeros, array, amin, amax, sqrt

from colormaps import ColorMap

class CiaoPlot(ColorMap):
    """ CIAO plot of coevolution performance with respect to the best
    individuals from previous generations (Hall of Fame).
    Requires 2 populations.  """

    @staticmethod
    def generateData(evaluator, hof1, hof2, symmetric=True):
        assert len(hof1) == len(hof2)
        gens = len(hof1)
        res = zeros((gens, gens))
        for g1, ind1 in enumerate(hof1):
            for g2, ind2 in enumerate(hof2[:g1 + 1]):
                res[g1, g2] = evaluator(ind1, ind2)
                if symmetric:
                    res[g2, g1] = res[g1, g2]
                elif g1 == g2:
                    # TODO: chack this!
                    res[g1, g2] += evaluator(ind2, ind1)
                else:
                    res[g2, g1] = evaluator(ind2, ind1)
        return res


    def __init__(self, evaluator, hof1, hof2, **args):
        if 'symmetric' in args:
            M = CiaoPlot.generateData(evaluator, hof1, hof2, symmetric=args['symmetric'])
            del args['symmetric']
        else:
            M = CiaoPlot.generateData(evaluator, hof1, hof2)
        M *= 1 / (amin(M) - amax(M))
        M -= amin(M)
        self.relData = M
        ColorMap.__init__(self, M, minvalue=0, maxvalue=1, **args)


if __name__ == '__main__':
    x = array(range(100))
    h1 = x * 4
    h2 = x + 20 * sqrt(x)
    def evo(x, y):
        return x - y
    from pylab import cm
    p = CiaoPlot(evo, h1, h2, cmap=cm.hot).show()
