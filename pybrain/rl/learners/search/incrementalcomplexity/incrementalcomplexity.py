__author__ = 'Tom Schaul, tom@idsia.ch'


# TODO: inheritance!

class IncrementalComplexitySearch(object):
    """ Draft of an OOPS-inspired search that incrementally expands the search space
    and the allocated time (to a population of search processes). """
    
    def __init__(self, initSearchProcess, maxPhases = 10, searchSteps = 50, desiredFitness = None):
        self.maxPhases = maxPhases
        self.searchSteps = searchSteps
        self.desiredFitness = desiredFitness
        self.processes = [initSearchProcess]
        self.phase = 0
        
    def optimize(self, **args):
        while self.phase <= self.maxPhases and not self.problemSolved():
            self._onePhase(**args)
            # increase the number of processes
            for p in self.processes[:]:
                self.processes.append(p.newSimilarInstance())
            self.increaseSearchSpace()
            self.phase += 1
            
        # return best evolvable
        best = -1e100
        for p in self.processes:
            if p.bestFitness > best:
                best = p.bestFitness
                res = p.evolvable
        return res
    
    def _onePhase(self, verbose = True, **args):
        if verbose:
            print 'Phase', self.phase
        for p in self.processes:
            p.search(self.searchSteps, **args)
            if verbose:
                print '', p.bestFitness, p.evolvable.weightLengths
    
    def increaseSearchSpace(self):
        for p in self.processes:
            p.increaseMaxComplexity()
    
    def problemSolved(self):
        if self.desiredFitness != None:
            for p in self.processes:
                if p.bestFitness > self.desiredFitness:
                    return True
        return False        