""" Utilities to get results produced by experiments. """

__author__ = 'Tom Schaul, tom@idsia.ch'

import os

from pybrain.tools.xml import NetworkReader


def getTaggedFiles(dir, tag, extension = '.pickle'):
    """ return a list of all files in the specified directory
    with a name stating with the given tag (and the specified extension). """
    allfiles = os.listdir(dir)
    res = []
    for f in allfiles:
        if f[-len(extension):] == extension and f[:len(tag)] == tag:
            res.append(dir+f)
    return res
    

if __name__ == '__main__':
    dir = '../temp/capturegame/1/'
    tag = 'N'
    ext = '.xml'
    files = getTaggedFiles(dir, tag, ext)
    nets = []
    otherdata = {}
    for f in files:
        n = NetworkReader.readFrom(f)
        nets.append(n) 
        print n
        if hasattr(n, '_unknown_argdict'):
            otherdata[n] = n._unknown_argdict.copy()
            del n._unknown_argdict
            for k, val in otherdata[n].items():
                print k, val
        print
        