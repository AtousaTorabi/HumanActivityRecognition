import pylearn2.models.svm as p_svm
import numpy
import sys
from model.pylearn2_ECCV2014.meanAvgPrec2 import meanAveragePrecisionTheano
import theano
import pickle

n_classes = 12
dup = 100

args = [0.5, 'linear', 1.0, 0.0, 3]
if len(sys.argv) > 2:
    for i, l in enumerate(sys.argv[2:]):
        if i != 1:
            args[i] = float(l)
        else:
            args[i] = l

svm = p_svm.DenseMulticlassSVM(*args)

def unonehot(x):
    return numpy.argmax(x, axis=1)

def onehot(x):
    def subonehot(y):
        z = numpy.zeros((n_classes,))
        z[y] = 1.
        return z
    return numpy.array([subonehot(y) for y in x])

data_train = numpy.loadtxt(sys.argv[1] + '_train_feats.txt')
train_depth = len(data_train)/dup 
labels_train = numpy.loadtxt(sys.argv[1] + '_train_labels.txt')
labels_train = unonehot(labels_train)

data_test = numpy.loadtxt(sys.argv[1] + '_test_feats.txt')
test_depth = len(data_test)/dup
labels_test = numpy.loadtxt(sys.argv[1] + '_test_labels.txt')

svm.fit(data_train, labels_train)
out = svm.predict(data_test)
out = onehot(out)

avg = numpy.array([[0]*n_classes]*test_depth)

for i in range(dup):
    avg += out[i*test_depth:(i+1)*test_depth]

avg = avg/dup

numpy.savetxt('out_values.txt', avg)
pickle.dump(svm, file('out_svm.pkl', mode='w'))
var1 = theano.tensor.fmatrix()
var2 = theano.tensor.fmatrix()
mAP = theano.function([var1, var2], meanAveragePrecisionTheano(var1, var2))(numpy.cast['float32'](labels_test[:test_depth]), numpy.cast['float32'](avg))
print mAP[:3] #min, mean, max
