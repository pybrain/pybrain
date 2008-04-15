__author__ = 'Tom Schaul, tom@idsia.ch'


import os
from scipy import array, size

from pybrain.tests.drafts.nesexperiments import pickleReadDict, pickleDumpDict

filedir = '.'
allfiles = os.listdir(filedir)
for f in allfiles:
    if f[-7:] == '.pickle':
        if f[-12:-7] != 'small':
            f2 = f[:-7]+'small.pickle'
            allfits = array(map(lambda x:x[1], pickleReadDict(f[:-7])))
            if size(allfits) > 0:
                pickleDumpDict(f2[:-7], allfits)
            
