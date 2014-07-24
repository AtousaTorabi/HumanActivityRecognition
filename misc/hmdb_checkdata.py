import sys
import os
import cPickle
import gzip
from itertools import islice

from theano import tensor as T
from theano import config
from theano import function
import numpy as np


from HumanActivityRecognition.dataset.HMDBDataset_fft import HMDBfftDataset



def oneofc(pred, nb_classes):
    opred = zeros((pred.shape[0], nb_classes))
    for i in  pred.shape[0]:
        opred[i, pred[i]] =1
    return opred

def oneofc_inv(pred):
    out = np.argmax(pred, axis=1)
    return out



if __name__ == '__main__':

    # if len(sys.argv) < 1:
    #     print("Usage %s: out [nb_run_model]" % sys.argv[0])
    #     exit(1)



    nb_run_model = 1
    ### Load dataset
    dataset = HMDBfftDataset(which_set='train',
                             data_path='/data/lisatmp3/ballasn/HMDB/cuboids',
                             split=int(1))

    batches = dataset.iterator(batch_size = 32,
                               shuffle_batch = False)

    cur = 0
    cur2 = 0
    for x, y in batches:

        if cur < 5:
            cur += 1
            cur2 += 32
            continue

        ex = x[22, :, :, :, :]
        print dataset.video_ids[cur2+22], ex.shape
        for t in xrange(0, ex.shape[2]):
            for x in xrange(0, ex.shape[0]):
                for y in xrange(0, ex.shape[1]):
                    for c in xrange(0, ex.shape[3]):
                        sys.stderr.write('%.7f\t' % ex[x, y, t, c])
                    sys.stderr.write('\n')
        exit(1)

