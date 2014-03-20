__author__ = 'Michael Isik'

from pybrain.datasets import SupervisedDataSet


class SVMData(SupervisedDataSet):
    """ Reads data files in LIBSVM/SVMlight format """
    def __init__(self, filename=None):
        SupervisedDataSet.__init__(self, 0, 0)

        self.nCls = 0
        self.nSamples = 0
        self.classHist = {}
        self.filename = ''
        if filename is not None:
            self.loadData(filename)


    def loadData(self, fname):
        """ decide which format the data is in """
        self.filename = fname
        if fname.find('.mat') >= 0:
            self.loadMATdata(fname)
        elif fname.find('.svm') >= 0:
            self.loadSVMdata(fname)
        else:
            # dataset consists of raw ascii columns
            self.loadRawData(fname)


    def _setDataFields(self, x, y):
        if not len(x): raise Exception("no input data found")
        SupervisedDataSet.__init__(self, len(x[0]), 1)
        self.setField('input'  , x)
        self.setField('target' , y)

        flat_labels = list(self.getField('target').flatten())
        classes = list(set(flat_labels))
        self._classes = classes
        self.nClasses = len(classes)
        for class_ in classes:
            self.classHist[class_] = flat_labels.count(class_)



    def loadMATdata(self, fname):
        """ read Matlab file containing one variable called 'data' which is an array
            nSamples x nFeatures+1 and contains the class in the first column """
        from mlabwrap import mlab #@UnresolvedImport
        from numpy import float
        d = mlab.load(fname)
        self.nSamples = d.data.shape[0]
        x = []
        y = []
        for i in range(self.nSamples):
            label = int(d.data[i, 0])


            x.append(d.data[i, 1:].astype(float).tolist())
            y.append([ float(label) ])
        self._setDataFields(x, y)

    def loadSVMdata(self, fname):
        """ read svm sparse format from file 'fname' (with labels only)
            output: [attributes[], labels[]] """

        x = []
        y = []
        nFeatMax = 0
        for line in open(fname, 'r').readlines():
            # format is:
            # <class>  <featnr>:<featval>  <featnr>:<featval> ...
            # (whereby featnr starts at 1)
            if not line: break
            line = line.split()
            label = float(line[0])


            feat = []
            nextidx = 1
            for r in line[1:]:
                # construct list of features, taking care of sparsity
                (idx, val) = r.split(':')
                idx = int(idx)
                for _ in range(nextidx, idx):
                    feat.append(0) # zzzzwar hier ein bug??
                feat.append(float(val))
                nextidx = idx + 1
            nFeat = len(feat)
            if nFeatMax < nFeat: nFeatMax = nFeat

            x.append(feat)
            y.append([ label ])
            self.nSamples += 1

        for xi in x:
            while len(xi) < nFeatMax:
                xi.append(0.)

        self._setDataFields(x, y)


    def loadRawData(self, fname):
        """ read svm sparse format from file 'fname' (with labels only)
            output: [attributes[], labels[]] """
        targetfile = open(fname.replace('data', 'targets'), 'r')
        x = []
        y = []
        for line in open(fname, 'r').readlines():
            if not line: break
            targline = targetfile.readline()
            targline = map(int, targline.split())
            for i, v in enumerate(targline):
                if v:
                    label = i
                    break
            feat = map(float, line.split())
            x.append(feat)
            y.append([float(label)])
            self.nSamples += 1
        self.nCls = len(targline)
        targetfile.close()
        self._setDataFields(x, y)

    def getNbClasses(self):
        return self.nCls

    def getNbSamples(self):
        return self.nSamples

    def getTargets(self):
        """ return the targets of the dataset, preserving the current sample pointer """
        self.storePointer()
        self.reset()
        targets = []
        while not self.endOfSequences():
            input, target, dummy = self.getSample()
            targets.append(target)
        self.recallPointer()
        return targets

    def getFileName(self):
        return self.filename

    def getClass(self, idx):
        return self._classes[idx]

    def getClassHistogram(self):
        """ return number of values per class as list of integers """
        return self.classHist


############################################################################
if __name__ == '__main__':
    d = SVMData()
    d.clear()
    d.loadSVMdata(r'M:\Data\Johan\svm\trials_scale.svm')
    print(d.getSample())
    print(d.getSample())

