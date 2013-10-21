from pybrain.optimization.optimizer import TabuOptimizer


class TabuHillClimber(TabuOptimizer):
    """Applies the tabu proccess in addition to a hill climbing search."""

    evaluatorIsNoisy = False

    def _learnStep(self):
        """generate a new a evaluable by mutation and check if it is tabu, repeat until a non-tabu                                         evaluable is created then keep it and update the tabu list iff the new evaluable is an improvement"""
        
        if self.evaluatorIsNoisy:
            self.bestEvaluation = self._oneEvaluation(self.bestEvaluable)
        tabu=True
        old=self.bestEvaluable
        while tabu:
            challenger = self.bestEvaluable.copy()
            challenger.mutate()
            tabu=False
            for t in self.tabuList:
                if t(challenger):
                    tabu=True
        self._oneEvaluation(challenger)
        if all(challenger.params[x]==self.bestEvaluable.params[x] for x in range(0,len(challenger))):
            self.tabuList.append(self.tabuGenerator(old,self.bestEvaluable))
            l=len(self.tabuList)
            if l > self.maxTabuList:
                self.tabuList=self.tabuList[(l-self.maxTabuList):l]
