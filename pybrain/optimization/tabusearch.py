from pybrain.optimization.optimizer import TabuOptimizer


class TabuHillClimber(TabuOptimizer, HillClimber):
    """Applies the tabu proccess in addition to a hill climbing search."""
