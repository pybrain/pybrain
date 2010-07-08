__author__ = 'Justin Bayer, bayerj@in.tum.de'


from pybrain.datasets.dataset import DataSet


class BenchmarkDataSet(DataSet):

    def __init__(self):
        super(BenchmarkDataSet, self).__init__()
        self.addField('Average Reward', 1)
        self.addField('Episode Length', 1)
        self.linkFields(['Average Reward', 'Episode Length'])

    def _initialValues(self):
        return tuple(), dict()
