import pylearn2.models.svm as p_svm
import numpy
import sys
from model.pylearn2_ECCV2014.meanAvgPrec2 import meanAveragePrecisionTheano
import theano
import pickle

n_classes = 12
dup = 100

def onehot(x):
    def subonehot(y):
        z = numpy.zeros((n_classes,))
        z[y] = 1.
        return z
    return numpy.array([subonehot(y) for y in x])

labels_test = numpy.loadtxt(sys.argv[1])
test_depth = len(labels_test)/dup

out = onehot(numpy.loadtxt(sys.argv[2]))
avg = numpy.array([[0]*n_classes]*test_depth)

for i in range(dup):
    avg += out[i*test_depth:(i+1)*test_depth]

avg = avg/dup

var1 = theano.tensor.fmatrix()
var2 = theano.tensor.fmatrix()
mAP = theano.function([var1, var2], meanAveragePrecisionTheano(var1, var2))(numpy.cast['float32'](labels_test[:test_depth]), numpy.cast['float32'](avg))
print mAP[:3] #min, mean, max
