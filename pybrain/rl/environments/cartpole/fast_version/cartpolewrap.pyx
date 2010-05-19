# C imports

cdef extern from "stdlib.h":
    ctypedef int size_t
    void *malloc(size_t size)
    void free(void *)

cdef extern from "cartpole.h":
    cdef void initCartPole(int markov_, int numPoles_, int maxsteps_)
    cdef void reset()
    cdef unsigned int getObservationDimension()
    cdef void getObservation(double * input)
    cdef void doAction(double * output)
    cdef int trialFinished()
    cdef double getReward()


# helper functions

cdef double * buildDoubleArray(int size):
    cdef int c
    cdef double * res
    res = <double*> malloc(size*sizeof(double))
    for c from 0 <= c < size:
        res[c] = 0.0
    return res


# methods visible from the outside:

def performAction(double action):
    cdef double * tmp
    tmp = buildDoubleArray(1)
    tmp[0] = action
    doAction(tmp)
    free(tmp)

def getObs():
    cdef int c, dim
    cdef double * tmp
    dim = getObservationDimension()
    tmp = buildDoubleArray(dim)
    getObservation(tmp)
    l = []
    for c from 1 <= c < dim:
        l.append(tmp[c])
    free(tmp)
    return l

def res():
    reset()

def init(int markov_, int numPoles_, int maxsteps_):
    initCartPole(markov_, numPoles_, maxsteps_)

def isFinished():
    return trialFinished()

def getR():
    return getReward()
