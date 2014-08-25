import os
import sys
import numpy
import tables
import math
import gzip
import copy

import numpy as np




def save_npy(filelist, indir, outdir):

    #print outdir, os.path.join(outdir, "features.py")


    ### Idx, used to save the index associated with each video in a hdf files
    nb_cuboids = 20*20*29
    nb_channel = 33
    features = np.zeros((len(filelist), nb_cuboids*nb_channel))
    cur = 0

    for filename in filelist:
        ### Load features
        feat = np.loadtxt(os.path.join(indir, filename.strip()))
        ### Flatten features
        feat = np.reshape(feat, feat.shape[0]*feat.shape[1])
        ### Copy the features (with 0 padding if necessary)
        print feat.shape
        features[cur, 0:feat.shape[0]] = feat
        cur += 1

    ### Save in npy format
    np.save(os.path.join(outdir, "features.py"), features)


def save_hdf(filelist, indir, outdir):

    outfile = os.path.join(outdir, "features.hdf")
    print outdir, outfile

    ### Idx, used to save the index associated with each video in a hdf files
    nb_cuboids = 40*40*29
    nb_channel = 33

    ### Create hdf tab
    f = tables.openFile(outfile, 'w')
    atom = tables.Float32Atom()
    ### No compression for now
    filters = tables.Filters(complib='blosc', complevel=0)
    hdf_shape = (len(filelist), nb_cuboids*nb_channel)
    feats = f.createCArray(f.root, 'denseFeat', atom,
                           hdf_shape, filters=filters)

    ### Fill the hdf tab
    cur = 0
    for filename in filelist:
        ### Load features
        feat = np.loadtxt(os.path.join(indir, filename.strip()))
        ### Flatten features
        feat = np.reshape(feat, feat.shape[0]*feat.shape[1])
        ### Copy the features (with 0 padding if necessary)
        print feat.shape

        feats[cur, 0:feat.shape[0]] = feat
        if feat.shape[0] != nb_cuboids*nb_channel:
            feats[cur, feat.shape[0]:] = 0
        cur += 1

    ### Save
    f.flush()
    f.close()


def read_list(filename):
    l = []
    with open(filename) as fp:
        for line in fp:
            l.append(line.strip())

    return l




if __name__ == "__main__":

    if (len(sys.argv) != 5):
        print("%s: trajlist labels indir outdir" % sys.argv[0])
        exit(1)

    l = read_list(sys.argv[1])
    labels = np.loadtxt(sys.argv[2])



    ### Add the flip version
    l2 = []
    #labels2 = np.zeros((2*labels.shape[0], labels.shape[1]))
    labels2 = np.zeros((labels.shape[0], labels.shape[1]))
    for i in xrange(0, len(l)):
        #print l[i][:-8]
        #new_name = l[i][:-8] + "_flip.traj.gz"
        #new_name = l[i][:-8] + "long_0"
        new_name = l[i][:-16] + ".cuboidsmall_0"
        #l2.append(l[i])
        l2.append(new_name)
        labels2[i, :] = labels[i, :]
        #labels2[2*i, :] = labels[i, :]
        #labels2[2*i+1, :] = labels[i, :]
    l = l2
    labels = labels2

    ### Make the number of example divisible by 128 (usual batch size)
    batch_size = 128
    nb_examples = len(l2)
    if (nb_examples % batch_size != 0):
        to_add = batch_size - nb_examples % batch_size
        labels = np.append(labels, labels[0:to_add, :], axis=0)
        for e in xrange(0, to_add):
            l.append(l[i])


    print len(l), labels.shape
    assert (len(l) == labels.shape[0])


    ### Dbg
    for i in xrange(0, labels.shape[1]):
        print i, np.where(labels[:, i] == 1)[0].shape
    ### End Dbg


    ## Shuffle example
    idx = np.arange(0, len(l))
    np.random.shuffle(idx)
    l2 = copy.copy(l)
    lab2 = np.zeros_like(labels)
    for i in xrange(0, len(l)):
         l2[i]      = l[idx[i]]
         lab2[i, :] = labels[idx[i], :]
    l = l2
    labels = lab2

    print len(l), labels.shape
    assert len(l) == labels.shape[0]

    ### Dbg
    for i in xrange(0, labels.shape[1]):
        print i, np.where(labels[:, i] == 1)[0].shape
    ### End Dbg


    ## Save the features
    save_hdf(l, sys.argv[3], sys.argv[4])
    ## Save the labels
    np.savetxt(os.path.join(sys.argv[4], "labels.txt"), labels)
    ## Save permutation
    np.savetxt(os.path.join(sys.argv[4], "rng_idx.txt"), idx)
    ## Save file list
    with open(os.path.join(sys.argv[4], 'files_lst.txt'), 'w') as fd:
        for item in l:
            fd.write("%s\n" % item.strip())
