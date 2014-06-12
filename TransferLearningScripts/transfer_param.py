import sys
import os
import cPickle
import gzip
from itertools import islice

from theano import tensor as T
from theano import function

from pylearn2.config import yaml_parse
from pylearn2.utils import serial

import numpy as np

import pylearn2_ECCV2014


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage %s: model1.pkl model2.yaml model2.pkl" % sys.argv[0])
        exit(1)

    ### Load model
    model1 = None
    with open(sys.argv[1], "rb") as fp:
        model1 = cPickle.load(fp)
    print("---------------------------------------")
    print(model1)
    print("---------------------------------------")
    print('')

    ## Load second model description
    print(sys.argv[2])
    with open(sys.argv[2]) as fp2:
        model2 = yaml_parse.load(fp2)

    print(model2)
    model2 = model2['model']

    print("---------------------------------------")
    print(model2)
    print("---------------------------------------")


    lastlayer = model2.layers[-1]
    model2 = model1
    model2.layers[-1] = lastlayer

    # ## Transfer parameters
    # for l in xrange(1, len(model1.layers)):
    #     model2.layers[l-1] = model1.layers[l]

    print('')
    print("---------------------------------------")
    print (model2)
    print("---------------------------------------")
    serial.save(sys.argv[3], model2, on_overwrite='backup')


    # ## Test
    # x = T.tensor4('x')
    # fprop = function([x], model2.fprop(x))

    # r = np.random.rand(64, 3000, 1, 1).astype(np.float32)
    # fprop(r)
