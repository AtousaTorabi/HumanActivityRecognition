from model.pylearn2_ECCV2014 import atousaDatasetHOHA2 as adh
import numpy

train = adh.AtousaDataset('train')
train_iter = train.iterator(num_batches=13)
test_iter = adh.AtousaDataset('test').iterator(num_batches=14)

kparam = 0
cross_count = 0

train_dists = []
test_dists = []

vec_or = numpy.vectorize(lambda x, y: x or y)
vec_and = numpy.vectorize(lambda x, y: x and y)

for i in range(13):
    next = train_iter.next()[0].reshape((64, 42600))
    for k in range(64):
        my_train_iter = train.iterator(num_batches=13)
        for j in range(13):
            next_train = my_train_iter.next()[0].reshape((64, 42600))
            for l in range(64):
                if j != i or k != l:
                    cross_count += 1
                dist = numpy.sum(numpy.where(vec_and(vec_or(next[k] != 0, next_train[l] != 0), abs(next[k]) != abs(next_train[l])), numpy.power(next_train[l] - next[k], 2)/(next_train[l] + next[k]), 0))
                dist /= 2
                train_dists.append(dist)

kparam = numpy.mean(numpy.array(train_dists))/cross_count
train_dists = numpy.exp(-numpy.array(train_dists)/kparam)

for i in range(14):
    next = test_iter.next()[0].reshape((64, 42600))
    for k in range(64):
        train_iter = train.iterator(num_batches=13)
        for j in range(13):
            next_train = train_iter.next()[0].reshape((64, 42600))
            for l in range(64):
                dist = numpy.sum(numpy.where(vec_and(vec_or(next[k] != 0, next_train[l] != 0), abs(next[k]) != abs(next_train[l])), numpy.power(next_train[l] - next[k], 2)/(next_train[l] + next[k]), 0))
                dist /= 2
                test_dists.append(dist)

test_dists = numpy.exp(-numpy.array(test_dists)/kparam)

numpy.savetxt('train_kern.txt', train_dists)
numpy.savetxt('test_kern.txt', test_dists)
