from __future__ import print_function

__author__ = 'Tom Schaul, tom@idsia.ch'

import os
import pickle

def getAllFilesIn(dir, tag='', extension='.pickle'):
    """ return a list of all filenames in the specified directory
    (with the given tag and/or extension). """
    allfiles = os.listdir(dir)
    res = []
    for f in allfiles:
        if f[-len(extension):] == extension and f[:len(tag)] == tag:
            res.append(f[:-len(extension)])
    return res


def selectSome(strings, requiredsubstrings=[], requireAll=True):
    """ Filter the list of strings to only contain those that have at least
    one of the required substrings. """
    if len(requiredsubstrings) == 0:
        return strings
    res = []
    for s in strings:
        if requireAll:
            bad = False
            for rs in requiredsubstrings:
                if s.find(rs) < 0:
                    bad = True
                    break
            if not bad:
                res.append(s)
        else:
            for rs in requiredsubstrings:
                if s.find(rs) >= 0:
                    res.append(s)
                    break
    return res


def pickleDumpDict(name, d):
    """ pickle-dump a variable into a file """
    try:
        f = open(name + '.pickle', 'w')
        pickle.dump(d, f)
        f.close()
        return True
    except Exception as e:
        print(('Error writing into', name, ':', str(e)))
        return False


def pickleReadDict(name):
    """ pickle-read a (default: dictionnary) variable from a file """
    try:
        f = open(name + '.pickle')
        val = pickle.load(f)
        f.close()
    except Exception as e:
        print(('Nothing read from', name, ':', str(e)))
        val = {}
    return val


def addToDictFile(name, key, data, verbose=False):
    if verbose:
        print(('.',))
    d = pickleReadDict(name)
    if key not in d:
        d[key] = []
    d[key].append(data)
    pickleDumpDict(name, d)
    if verbose:
        print(':')


