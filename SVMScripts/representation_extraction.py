import sys
import os
import cPickle
import gzip

from theano import tensor as T
from theano import function
import numpy as np

import pylearn2_ECCV2014

from pylearn2_ECCV2014.atousaDatasetHOHA2 import AtousaDataset



def get_basename(string):
    string = os.path.basename(string)
    return  os.path.splitext(os.path.splitext(string)[0])[0]

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage %s: model duplication outdir" % sys.argv[0])
        exit(1)


    print("WARNING BATCH HAVE TO BE CONSIDERED IN A SEQUENTIAL MANNER FOR THIS TO WORK")

    duplication = int(sys.argv[2])

    ### Load model
    model = None
    print(sys.argv[1])
    with open(sys.argv[1], "rb") as fd:
        model = cPickle.load(fd)
    print(model)


    ### Chop-off the last layers
    model.layers.pop()

    ### Compile fprop function
    x = T.tensor4("x")
    get_representation = function([x], model.fprop(x))


    for which_set in ['train', 'test']:
        ### Load dataset
        dataset = pylearn2_ECCV2014.atousaDatasetHOHA2.AtousaDataset(which_set='test')

        ### Dimension of the intermediate representation
        dim = 480
        nb_class = 12
        nb_elem = dataset.Index.shape[0]
        print(nb_elem)
        nb_feats = nb_elem * duplication

        ### Intermediate features labels
        feats = np.zeros((nb_feats, dim))
        labels = np.zeros((nb_feats, nb_class))
        files_idx = np.zeros((nb_feats, 2))

        ### FIXME fill file idx
        for d in xrange(0, duplication):
            batch_size = 64
            cur = 0
            batches = dataset.iterator(batch_size=batch_size)
            i = 0
            for x, y in batches:
                feats[i*batch_size+d*nb_elem:(i+1)*batch_size+d*nb_elem, :] = get_representation(x)
                labels[i*batch_size+d*nb_elem:(i+1)*batch_size+d*nb_elem, :] = y
                for b in xrange(0, batch_size):
                    files_idx[b] = i*batch_size+b
                i = i + 1

        np.savetxt(sys.argv[3] + '_' + which_set + '_feats.txt', feats)
        np.savetxt(sys.argv[3] + '_' + which_set + '_labels.txt', labels)
        np.savetxt(sys.argv[3] + '_' + which_set + '_file_idx.txt', files_idx)


