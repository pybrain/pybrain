import itertools
import os
import scipy
import struct

from pybrain.datasets import SupervisedDataSet


def labels(filename):
    fp = file(filename)
    magicnumber, length = struct.unpack('>ii', fp.read(8))
    assert magicnumber in (2049, 2051), ("Not an MNIST file: %i" % magicnumber)
    for _ in xrange(length):
        label, = struct.unpack('B', fp.read(1))
        yield label


def images(filename):
    fp = file(filename,'rb')
    chunk = fp.read(16)
    magicnumber, length, numrows, numcols = struct.unpack('>iiii', chunk)
    assert magicnumber in (2049, 2051), ("Not an MNIST file: %i" % magicnumber)
    imagesize = numrows * numcols
    for _ in xrange(length):
        imagestring = fp.read(imagesize)
        image = struct.unpack('B' * imagesize, imagestring)
        yield scipy.array(image)


def flaggedArrayByIndex(idx, length):
    arr = scipy.zeros(length)
    arr[idx] = 1.
    return arr


def makeMnistDataSets(path):
    """Return a pair consisting of two datasets, the first being the training
    and the second being the test dataset."""
    test = SupervisedDataSet(28 * 28, 10)
    test_image_file = os.path.join(path, 't10k-images-idx3-ubyte')
    test_label_file = os.path.join(path, 't10k-labels-idx1-ubyte')
    test_images = images(test_image_file)
    test_labels = (flaggedArrayByIndex(l, 10) for l in labels(test_label_file))

    for image, label in itertools.izip(test_images, test_labels):
        test.addSample(image, label)

    train = SupervisedDataSet(28 * 28, 10)
    train_image_file = os.path.join(path, 'train-images-idx3-ubyte')
    train_label_file = os.path.join(path, 'train-labels-idx1-ubyte')
    train_images = images(train_image_file)
    train_labels = (flaggedArrayByIndex(l, 10) for l in labels(train_label_file))
    for image, label in itertools.izip(train_images, train_labels):
        train.addSample(image, label)

    return train, test
