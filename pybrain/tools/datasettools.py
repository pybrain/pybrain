# This tool converts a sequential data set into a number of equally sized windows,
# to be used for supervised training.
__author__ = "Martin Felder"


from numpy import r_, array, isfinite
from pybrain.datasets import SequentialDataSet


def convertSequenceToTimeWindows(DSseq, NewClass, winsize):
    """ Converts a sequential classification dataset into time windows of fixed length.
    Assumes the correct class is given at the last timestep of each sequence. Incomplete windows at the
    sequence end are pruned. No overlap between windows.

    :arg DSseq: the sequential data set to cut up
    :arg winsize: size of the data window
    :arg NewClass: class of the windowed data set to be returned (gets initialised with indim*winsize, outdim)"""
    assert isinstance(DSseq, SequentialDataSet)
    #assert isinstance(DSwin, SupervisedDataSet)

    DSwin = NewClass(DSseq.indim * winsize, DSseq.outdim)
    nsamples = 0
    nseqs = 0
    si = r_[DSseq['sequence_index'].flatten(), DSseq.endmarker['sequence_index']]
    for i in xrange(DSseq.getNumSequences()):
        # get one sequence as arrays
        input = DSseq['input'][si[i]:si[i + 1], :]
        target = DSseq['target'][si[i]:si[i + 1], :]
        nseqs += 1
        # cut this sequence into windows, assuming class is given at the last step of each sequence
        for k in range(winsize, input.shape[0], winsize):
            inp_win = input[k - winsize:k, :]
            tar_win = target[k - 1, :]
            DSwin.addSample(inp_win.flatten(), tar_win.flatten())
            nsamples += 1
            ##print("added sample %d from sequence %d: %d - %d" %( nsamples, nseqs, k-winsize, k-1))
    print("samples in original dataset: ", len(DSseq))
    print("window size * nsamples = ", winsize * nsamples)
    print("total data points in original data: ", len(DSseq) * DSseq.indim)
    print("total data points in windowed dataset: ", len(DSwin) * DSwin.indim)
    return DSwin

def windowSequenceEval(DS, winsz, result):
    """ take results of a window-based classification and assess/plot them on the sequence
    WARNING: NOT TESTED!"""
    si_old = 0
    idx = 0
    x = []
    y = []
    seq_res = []
    for i, si in enumerate(DS['sequence_index'][1:].astype(int)):
        tar = DS['target'][si - 1]
        curr_x = si_old
        correct = 0.
        wrong = 0.
        while curr_x < si:
            x.append(curr_x)
            if result[idx] == tar:
                correct += 1.
                y += [1., 1.]
            else:
                wrong += 1.
                y += [0., 0.]
            idx += 1
            #print("winidx: ", idx)
            curr_x += winsz
            x.append(curr_x)

        seq_res.append(100. * correct / (correct + wrong))
        print("sequence %d correct: %g12.2%%" % (i, seq_res[-1]))

    seq_res = array(seq_res)
    print("total fraction of correct sequences: ", 100. * float((seq_res >= 0.5).sum()) / seq_res.size)


class DataSetNormalizer(object):
    """ normalize a dataset according to a stored LIBSVM normalization file """
    def __init__(self, fname=None, meanstd=False):
        self.dim = 0
        self.meanstd = meanstd
        if fname is not None:
            self.load(fname)

    def load(self, fname):
        f = file(fname)
        c = []
        # the first line determines whether we interpret the file as
        # giving min/max of features or mean/std
        x = f.readline()
        self.meanstd = False if x == 'x' else True

        # the next line gives the normalization bounds
        bounds = array(f.readline().split()).astype(float)
        for line in f:
            c.append(array(line.split()).astype(float)[1:])
        self.dim = len(c)
        c = array(c)
        self.par1 = c[:, 0]
        self.par2 = c[:, 1]
        self.scale = (bounds[1] - bounds[0]) / (c[:, 1] - c[:, 0])
        self.newmin = bounds[0]
        self.newmax = bounds[1]

    def save(self, fname):
        f = file(fname, "w+")
        f.write('x\n')
        f.write('%g %g' % (self.newmin, self.newmax))
        for i in range(self.dim):
            f.write('%d %g %g' % (i + 1, self.par1[i], self.par2[i]))
        f.close()

    def normalizePattern(self, y):
        return (y - self.par1) * self.scale + self.newmin

    def normalize(self, ds, field='input'):
        """ normalize dataset or vector wrt. to stored min and max """
        if self.dim <= 0:
            raise IndexError("No normalization parameters defined!")
        dsdim = ds[field].shape[1]
        if self.dim != dsdim:
            raise IndexError("Dimension of normalization params does not match DataSet field!")
        newfeat = ds[field]
        if self.meanstd:
            for i in range(dsdim):
                divisor = self.par2[i] if self.par2[i] > 0 else 1.0
                newfeat[:, i] = (newfeat[:, i] - self.par1[i]) / divisor
        else:
            for i in range(dsdim):
                scale = self.scale[i] if isfinite(self.scale[i]) else 1.0
                newfeat[:, i] = (newfeat[:, i] - self.par1[i]) * scale + self.newmin
        ds.setField(field, newfeat)

    def calculate(self, ds, bounds=[-1, 1], field='input'):
        self.dim = ds[field].shape[1]
        if self.meanstd:
            self.par1 = ds[field].mean(axis=0)
            self.par2 = ds[field].std(axis=0)
        else:
            self.par1 = ds[field].min(axis=0)
            self.par2 = ds[field].max(axis=0)
            self.scale = (bounds[1] - bounds[0]) / (self.par2 - self.par1)
        self.newmin = bounds[0]
        self.newmax = bounds[1]


