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

    if len(sys.argv) < 4:
        print("Usage %s: model split out [nb_run_model]" % sys.argv[0])
        exit(1)



    outputdir =  sys.argv[3]
    nb_run_model = 1
    if len(sys.argv) >= 5:
        model_avg = int(sys.argv[4])

    ### Load model
    model = None
    with open(sys.argv[1], "rb") as file:
        model = cPickle.load(file)
    print(model)

    ### Load dataset
    dataset = HMDBfftDataset(which_set='test',
                             data_path='/state/partition2/ballas_n/HMDB/cuboids',
                             split=int(sys.argv[2]),)

    ### Compile network prediction function
    tensor5 = T.TensorType(config.floatX, (False,)*5)
    x = tensor5("x")
    predict = function([x], model.fprop(x))

    ### Compute prediction of training set
    batch_size = 64
    nb_classes = dataset.labels.shape[1]
    preds  = np.zeros((dataset.nb_examples, nb_classes), dtype=config.floatX)
    ## Binary labels
    blabels = np.zeros((dataset.nb_examples, nb_classes), dtype=config.floatX)
    

    cur = 0
    batches = dataset.iterator(batch_size=batch_size,shuffle_batch=False)
    for x, y in batches:
        #print(cur, np.argmax(y, axis=1))
        pred_tmp = np.zeros((batch_size, nb_classes), dtype=config.floatX)
        for i in xrange(0, nb_run_model):
            pred_tmp = predict(x)

        pred_tmp = pred_tmp / nb_run_model
        preds[cur*batch_size:(cur+1)*batch_size] = pred_tmp
        blabels[cur*batch_size:(cur+1)*batch_size] = y
        cur = cur +1

    print(blabels.shape)
    print(preds.shape)

    ### Keep only the original number of example
    ### (Not necessary a multiple of batch_size)
    nb_examples = dataset.orig_nb_examples 
    preds = preds[0:nb_examples, :]
    blabels = blabels[0:nb_examples, :]

    ### Transform pred in 1outofc binary vector
    labels = oneofc_inv(blabels)
    prlabels = oneofc_inv(preds)
    reflabels = oneofc_inv(dataset.labels[0:nb_examples, :])
    assert np.equal(labels, reflabels).all()

    ### Create list to save good_classif and missclassif
    goodclassif = []
    missclassif = []
    for i in xrange(0, nb_classes):
        goodclassif.append([])
        missclassif.append([])


    ### Compute Score
    # Print scores and save missclassified files
    for i in xrange(0, preds.shape[0]):
        print i, preds[i, :], prlabels[i], labels[i]
        if prlabels[i] == labels[i]:
            goodclassif[labels[i]].append(dataset.video_ids[i])
        else:
            missclassif[labels[i]].append(dataset.video_ids[i])
        
    
    pre = np.zeros(nb_classes)
    rec = np.zeros(nb_classes)
    ### Compute true/false positive, negative
    for c in xrange(0, nb_classes):
        tp = np.sum((labels == c) & (prlabels == c))
        fp = np.sum((labels != c) & (prlabels == c));
        tn = np.sum((labels != c) & (prlabels != c));
        fn = np.sum((labels == c) & (prlabels != c));

        pre[c] = tp / (tp + fp + 0.0001) 
        rec[c] = tp / (tp + fn + 0.0001) 
        print dataset.labels_name[c], pre[c], rec[c]
        #exit(1)

    print "Acc (avg):", np.mean(pre, axis=0), "Rec (avg):", np.mean(rec, axis=0)

    ### Write goodclassif and missclassif
    for i in xrange(0, nb_classes):
        filename = os.path.join(outputdir, "goodclassif_" + str(i) + ".txt")
        with open(filename, "w") as fd:
            for item in goodclassif[i]:
                fd.write("%s\n" % item.strip())
        filename = os.path.join(outputdir, "missclassif_" + str(i) + ".txt")
        with open(filename, "w") as fd:
            for item in missclassif[i]:
                fd.write("%s\n" % item.strip())

    
