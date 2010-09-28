__author__ = 'Tom Schaul, tom@idsia.ch, and Daan Wierstra'


from multimodal import MultiModalFunction
from scipy import sqrt, tile, swapaxes, ravel, eye, randn
import scipy 


class LennardJones(MultiModalFunction):
    """ The classical atom configuration problem. The problem dimension must be a multiple of 3, and the 
    input are the Cartesian coordinates of all atoms."""
    
    def f(self, x):
        N = self.xdim/3
        coords =  x.reshape((N,3))
        distances = sqrt(scipy.sum((tile(coords, (N, 1, 1))-swapaxes(tile(coords, (N, 1, 1)), 0, 1))**2, axis=2))+eye(N)
        return 2*sum(ravel(distances**-12 - distances**-6))
        
    def _exampleConfig(self, numatoms, noise=0.05, edge=2.):
        """ Arranged in an approximate cube of certain edge length. """
        assert numatoms % 8 == 0
        x0 = randn(3,2,2,2,numatoms/8) * noise * edge - edge/2
        x0[0,0] += edge
        x0[1,:,0] += edge
        x0[2,:,:,0] += edge
        x0 = x0.reshape(3, numatoms).T
        return x0.flatten()
    
    @property
    def desiredValue(self):
        N = self.xdim/3
        return self.BEST_KNOWN_TABLE[N]+1e-5
    
    BEST_KNOWN_TABLE = {2: -1,
                        3: -3,
                        4: -6,
                        5: -9.103852,
                        6: -12.712062,
                        7: -16.505384,
                        8: -19.821489,
                        9: -24.113360,
                        10:-28.422532,
                        11:-32.765970,
                        12:-37.967600,
                        13:-44.326801,
                        14:-47.845157,
                        15:-52.322627,
                        16:-56.815742,
                        17:-61.317995,
                        18:-66.530949,
                        19:-72.659782,
                        20:-77.177043,
                        21:-81.684571,
                        22:-86.809782,
                        23:-92.844472,
                        24:-97.348815,
                        25:-102.372663,
                        26:-108.315616,
                        27:-112.873584,
                        }
    