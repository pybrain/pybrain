__author__ = 'Tom Schaul, tom@idsia.ch'

from innermemetic import InnerMemeticSearch
from inversememetic import InverseMemeticSearch

class InnerInverseMemeticSearch(InnerMemeticSearch, InverseMemeticSearch):
    """ inverse of inner memetic search"""

    def _learnStep(self):
        self.switchMutations()
        InnerMemeticSearch._learnStep(self)
        self.switchMutations()