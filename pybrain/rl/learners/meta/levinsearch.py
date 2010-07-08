__author__ = 'Tom Schaul, tom@idsia.ch'


def timeBoundExecution(algo, maxtime):
    """ wrap the algo, to stop execution after it has used all its allocated time """
    # TODO
    return algo


class LevinSeach:
    """ a.k.a. Universal Search

    Note: don't run this, it's a bit slow... but it will solve all your problems! """

    def stoppingCriterion(self, val):
        return val == True

    def run(self, input, generator):
        complexities = {}

        # an iterator over all valid programs, by increasing complexity, and in
        # lexicographical order, together with its code's complexity.
        piter = generator.orderedEnumeration()

        maxLevin = 1
        # every phase goes through all programs of a certain Levin-complexity.
        while True:

            # generate all programs that might be needed in this phase.
            c = 0
            while c <= maxLevin:
                try:
                    c, p = piter.next()
                except StopIteration:
                    break
                if c not in complexities:
                    complexities[c] = [p]
                else:
                    complexities[c].append(p)

            # execute all programs, but with a set time-limit:
            # every phase the total time used doubles (= 2**maxLevin)
            for c in range(maxLevin):
                for p in complexities[c]:
                    boundP = timeBoundExecution(p, 2**(maxLevin-c)/maxLevin)
                    res = boundP.run(input)
                    if self.stoppingCriterion(res):
                        return res

            maxLevin += 1