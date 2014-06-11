import os
import math
import warnings
import numpy
import random
import thread
import time
try:
    import tables
except ImportError:
    warnings.warn("Couldn't import tables, so far DenseFeat is "
            "only supported with PyTables")
import numpy as np
from theano import config
from pylearn2.datasets import dataset
from pylearn2.utils.serial import load
from pylearn2.utils.string_utils import preprocess
from pylearn2.space import VectorSpace, CompositeSpace

class AtousaDataset(dataset.Dataset):



    nbTags = 12
    nb_feats = 426
    nb_bins = 400
    nb_samples = 10
    vidShape = [nb_feats, nb_bins, nb_samples]
    vidSize = vidShape[0] * vidShape[1] * vidShape[2]
    mapper = {'train': 0, 'valid': 1, 'test': 2}

    def __init__(self, data_path,
                 which_set,
                 axes = ('b', 0, 1, 'c')):

        if which_set == 'train':
            self.bin_idx = np.loadtxt(data_path +'/train/bins_idx.txt')
            self.traj_idx = np.loadtxt(data_path+'/train/trajs_idx.txt')
            self.labels  = np.loadtxt(data_path + '/train/labels.txt')
            self.data_path = data_path + '/train/'

            ### Fix generation bugs
            self.nbExamples = 832
            self.labels =  self.labels[0:832, :]
        else:
           self.bin_idx = np.loadtxt(data_path +'/test/bins_idx.txt')
           self.traj_idx = np.loadtxt(data_path+'/test/trajs_idx.txt')
           self.labels  = np.loadtxt(data_path + '/test/labels.txt')
           self.data_path = data_path + '/test/'
           #self.nbExamples = self.labels.shape[0]
           ### Fix generation bugs
           self.nbExamples = 896
           self.labels =  self.labels[832:1728, :]
         



        print(self.labels.shape)

        self.nb_examples = self.nbExamples
        self.which_set = which_set
        self.axes = axes


    def get_minibatch(self, firstIdx, lastIdx, batch, batch_size,
                      data_specs, return_tuple):

       x = np.zeros(self.vidShape + [lastIdx - firstIdx,],
                    dtype = "float32")
       y = np.zeros([lastIdx - firstIdx, self.nbTags],
                       dtype="float32")

       ### Open the hdf batch
       #batch = lastIdx / batch_size
       file_n = self.data_path + "%07d" % batch + ".hdf"
       self.h5file = tables.openFile(file_n, mode = "r")
       data = self.h5file.getNode('/', "denseFeat")

       assert(lastIdx - firstIdx == batch_size)


       ## Fix generation bug
       bin_offset = 0
       for i in xrange(0 + batch_size, lastIdx, 64):
           bin_offset += self.bin_idx[i - 1][1]

       lastbinIdx = self.bin_idx[lastIdx - 1, 1] -1 + bin_offset
       print(self.traj_idx[lastbinIdx, 1], data.shape[0])
       assert(self.traj_idx[lastbinIdx, 1] == data.shape[0])



       ### Sample each videos
       for v in xrange(firstIdx, lastIdx):

           batchIdx = v - firstIdx

           ### Sample each each bins
           for b in xrange(int(self.bin_idx[v][0]),
                           int(self.bin_idx[v][1])):
               if b - self.bin_idx[v][0] >= self.nb_bins:
                   break
               ### Compute the trajectory start and end for the current bin
               start = self.traj_idx[b + bin_offset][0]
               end = self.traj_idx[b + bin_offset][1]

               s = end - start

               ### Sample traj
               if s == 0:
                   continue
               elif s < self.nb_samples:
                   datatmp = data[start:end,:]
                   t = int(math.ceil(self.nb_samples/s))
                   dd = datatmp
                   for k in range(0,t):
                       datatmp = np.append(datatmp, dd, axis=0)
                   np.random.shuffle(datatmp)
                   R = random.randint(0,(len(datatmp)-self.nb_samples))
                   datatmp = datatmp[R:R+self.nb_samples, 10:436]
               else:
                   R = random.randint(0,(s-self.nb_samples))
                   datatmp=data[start + R:start + R + self.nb_samples, 10:436]

               x[:, int(b - self.bin_idx[v][0]), :, batchIdx] = datatmp.transpose()

           y[batchIdx,:] = self.labels[v];


       self.h5file.close()
       print "Returned a minibatch"
       return x, y
    def get_design_matrix(self, topo = False):
	return np.zeros((self.nbExamples, 1))


    def iterator(self, mode=None, batch_size=None, num_batches=None, topo=None,
                 targets=False, data_specs=None, return_tuple=False, rng=None):

        # We ignore mode, topo and targets because screw them!
        return AtousaDatasetIterator(self, batch_size, num_batches,
                                     data_specs, return_tuple, rng)


    def has_targets(self):
        return True

    def get_topo_batch_axis(self):
        """
        Returns the index of the axis that corresponds to different examples
        in a batch when using topological_view.
        """
        return self.axes.index('b')

class AtousaDatasetIterator:

    stochastic = False

    def __init__(self, dataset=None, batch_size=None, num_batches=None,
                 data_specs=False, return_tuple=False, rng=None):

        self._dataset = dataset
        self._dataset_size = dataset.nbExamples

        # Validate the inputs
        assert dataset is not None

        if batch_size is None:
            if num_batches is not None:
                batch_size = int(np.ceil(self._dataset_size / float(num_batches)))
            else:
                raise ValueError("Provide at least one of batch_size or num_batches")
        elif batch_size is not None:
            if num_batches is not None:
                max_num_batches = np.ceil(self._dataset_size / float(batch_size))
                if num_batches > max_num_batches:
                    raise ValueError("dataset of %d examples can only provide "
                                     "%d batches with batch_size %d, but %d "
                                     "batches were requested" %
                                     (self._dataset_size, max_num_batches,
                                      batch_size, num_batches))
            else:
                num_batches = np.ceil(self._dataset_size / float(batch_size))

        # Define instance attributes
        #if rng is None:
        #    self._rng = random.Random(1)
        #else:
        #    self._rng = rng
        self._rng = random.Random(1)
        self._batch_size = batch_size
        self._num_batches = int(num_batches)
        self._next_batch_no = 0
        self._batch_order = range(self._num_batches)
        self._rng.shuffle(self._batch_order)
        self._return_tuple = return_tuple
        self._data_specs = data_specs
        self.num_examples = self._dataset_size # Needed by Dataset interface
        self._timer14=time.time()
        print self.num_examples


    def __iter__(self):
        return self

    def next(self):
        timer13 = 0
        if self._next_batch_no >= self._num_batches:
            print self.num_examples
            print self._num_batches
            raise StopIteration()
        else:
            # Determine minibatch start and end idx
            first = self._batch_order[self._next_batch_no] * self._batch_size
            if first + self._batch_size > self._dataset_size:
                last = self._dataset_size
            else:
                last = first + self._batch_size
            timer13 -= time.time()
            data =  self._dataset.get_minibatch(first, last,
                                                self._batch_order[self._next_batch_no],
                                                self._batch_size,
                                                self._data_specs,
                                                self._return_tuple)

            timer13 += time.time()
            print self._next_batch_no
            #print timer13

            #print time.time() - self._timer14
            self._timer14 = time.time()
            self._next_batch_no += 1
            return data




