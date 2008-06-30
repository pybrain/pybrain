# This tool converts a sequential data set into a number of equally sized windows, 
# to be used for supervised training.

__author__ = "Martin Felder"
__version__ = '$Id$' 
from os.path import join
from numpy import r_, array
from pybrain.datasets import SupervisedDataSet, SequentialDataSet, ClassificationDataSet, SequenceClassificationDataSet
import pylab as p

def convertSequenceToTimeWindows(DSseq, NewClass, winsize):
    """ converts a sequential dataset into time windows of fixed length (for use with MLPs) """
    assert isinstance(DSseq, SequentialDataSet)
    #assert isinstance(DSwin, SupervisedDataSet)
    
    DSwin = NewClass(DSseq.indim*winsize, DSseq.outdim)
    nsamples = 0
    nseqs = 0
    si = r_[DSseq['sequence_index'].flatten(), DSseq.endmarker['sequence_index']]
    for i in xrange(DSseq.getNumSequences()):
        # get one sequence as arrays
        input = DSseq['input'][si[i]:si[i+1],:]
        target = DSseq['target'][si[i]:si[i+1],:]
        nseqs += 1
        # cut this sequence into windows, assuming class is given at the last step
        for k in range(winsize, input.shape[0], winsize):
            inp_win = input[k-winsize:k,:]
            tar_win = target[k-1,:]   
            DSwin.addSample(inp_win.flatten(), tar_win.flatten())
            nsamples += 1
            ##print "added sample %d from sequence %d: %d - %d" %( nsamples, nseqs, k-winsize, k-1)
    print "samples in original dataset: ", len(DSseq)
    print "window size * nsamples = ", winsize*nsamples
    print "total data points in original data: ", len(DSseq) * DSseq.indim
    print "total data points in windowed dataset: ", len(DSwin) * DSwin.indim
    return DSwin

def windowSequenceEval(DS, winsz, result):
    """ take results of a window-based classification and assess/plot them on the sequence """
    si_old = 0
    idx = 0 
    x = []
    y = []
    seq_res = []
    for si in DS['sequence_index'][1:]:
        tar = DS['target'][si-1]
        curr_x = si_old
        correct = 0.
        wrong  =  0.
        while curr_x < si:    
            x.append(curr_x)
            if result[idx]==tar:
                correct += 1.
                y += [1.,1.]
            else:
                wrong += 1.
                y += [0.,0.]
            idx += 1
            print "winidx: ", idx
            curr_x += winsz
            x.append(curr_x)
        
        seq_res.append(100.*correct/(correct+wrong))
        print "sequence %d correct: %g2.2%%"
        
    seq_res = array(seq_res)
    print "total fraction of correct sequences: ", 100.*float((seq_res>=0.5).sum())/seq_res.size
        
    
    
    
    
if __name__ == "__main__":
    winsize = 5
    pathtodata = '/maxdat/Data/Calogero/v1.1'
    fname = "V1-4_333Hz_norm"
    print "loading file "+ join(pathtodata,fname+'.pkl') 
    DSseq = SequenceClassificationDataSet.reconstruct( join(pathtodata,fname+'.pkl') )
    #DSseq.setField('input', DSseq['input'][:,3:])
    print "indim seq:", DSseq.indim
    print "winsize: ", winsize
    DSwin = convertSequenceToTimeWindows(DSseq, ClassificationDataSet, winsize)
    DSwin.saveToFile(join(pathtodata,fname+'_win%d.pkl'%winsize), protocol=-1, arraysonly=True)
    print "indim win:", DSwin.indim
   