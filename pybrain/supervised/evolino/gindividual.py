__author__ = 'Michael Isik'


class Individual(object):
    """ Simple abstract template for a minimal individual """
    def getGenome(self):
        """ Should return a reference to the genome.
        """
        raise NotImplementedError()

    def copy(self):
        """ Should return a full copy of the individual
        """
        raise NotImplementedError()


