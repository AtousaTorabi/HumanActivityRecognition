import os
import sys
import numpy
import tables
import math
import gzip

import numpy as np




def save_npy(filelist, indir, outdir):

    print outdir, os.path.join(outdir, "features.py")


    ### Idx, used to save the index associated with each video in a hdf files
    nb_cuboids = 3600
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


def read_list(filename):
    l = []
    with open(filename) as fp:
        for line in fp:
            l.append(line)

    return l

if __name__ == "__main__":

    if (len(sys.argv) != 5):
        print("%s: trajlist labels indir outdir" % sys.argv[0])
        exit(1)

    l = read_list(sys.argv[1])
    labels = np.loadtxt(sys.argv[2])
    print(labels.shape)

    ## Shuffle example
    idx = np.arange(0, len(l))
    np.random.shuffle(idx)
    l2 = l
    lab2 = labels
    for i in xrange(0, len(l)):
         l2[i]      = l[idx[i]]
         lab2[i, :] = labels[idx[i], :]
    l = l2
    labels = lab2


    ## Save the features
    save_npy(l, sys.argv[3], sys.argv[4])
    ## Save the labels
    np.savetxt(os.path.join(sys.argv[4], "labels.txt"), labels)
    ## Save permutation
    np.savetxt(os.path.join(sys.argv[4], "rng_idx.txt"), idx)
    ## Save file list
    with open(os.path.join(sys.argv[4], 'files_lst.txt'), 'w') as fd:
        for item in l:
            fd.write("%s\n" % item.strip())
