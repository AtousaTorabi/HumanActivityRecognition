import os
import math
import warnings
import numpy as np
import random
import thread
import Queue
import time
try:
    import tables
except ImportError:
    warnings.warn("Couldn't import tables, so far DenseFeat is "
            "only supported with PyTables")

from theano import config
from pylearn2.datasets import dataset
from pylearn2.utils.serial import load
from pylearn2.utils.string_utils import preprocess
from pylearn2.space import VectorSpace, CompositeSpace


from pylearn2_ECCV2014.atousaDataset import AtousaDataset,AtousaDatasetIterator



class HMDBDataset(dataset.Dataset):

    nbTags = 51
    nb_feats = 426
    nb_bins = 400
    nb_samples = 10
    vidShape = [nb_feats, nb_bins, nb_samples]
    vidSize = vidShape[0] * vidShape[1] * vidShape[2]
    mapper = {'train': 0, 'valid': 1, 'test': 2}

    def __init__(self, data_path,
                 split, which_set,
                 axes = ('b', 0, 1, 'c')):

        if which_set == 'train':
            self.bin_idx = np.loadtxt(data_path +'/%s_1/bins_idx.txt'%split)
            self.traj_idx = np.loadtxt(data_path+'/%s_1/trajs_idx.txt'%split)
            self.labels  = np.loadtxt(data_path + '/%s_1/labels.txt' % split)
            self.data_path = data_path + '/%s_1/' % split
        elif which_set == 'valid':
           self.bin_idx = np.loadtxt(data_path +'/%s_0/bins_idx.txt'%split)
           self.traj_idx = np.loadtxt(data_path+'/%s_0/trajs_idx.txt'%split)
           self.labels  = np.loadtxt(data_path + '/%s_0/labels.txt' % split)
           self.data_path = data_path + '/%s_0/' % split
        else:
           self.bin_idx = np.loadtxt(data_path +'/%s_2/bins_idx.txt'%split)
           self.traj_idx = np.loadtxt(data_path+'/%s_2/trajs_idx.txt'%split)
           self.labels  = np.loadtxt(data_path + '/%s_2/labels.txt' % split)
           self.data_path = data_path + '/%s_2/' % split


        print(self.labels.shape)

        self.nb_examples = self.labels.shape[0]
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
       h5file = tables.openFile(file_n, mode = "r")
       data = h5file.getNode('/', "denseFeat")

       print(lastIdx, self.bin_idx[lastIdx - 1, 1])
       assert(lastIdx - firstIdx == batch_size)
       assert(self.traj_idx[self.bin_idx[lastIdx - 1, 1] - 1, 1] == data.shape[0])


       ### Sample each videos
       for v in xrange(firstIdx, lastIdx):

           batchIdx = v - firstIdx
           
           ### Sample each each bins
           for b in xrange(int(self.bin_idx[v][0]),
                           int(self.bin_idx[v][1])):
               if b - self.bin_idx[v][0] >= self.nb_bins:
                   break
               ### Compute the trajectory start and end for the current bin
               start = self.traj_idx[b][0]
               end = self.traj_idx[b][1]
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


       h5file.close()
       print "Returned a minibatch"
       return x, y



    def iterator(self, mode=None, batch_size=None, num_batches=None, topo=None,
                 targets=False, data_specs=None, return_tuple=False, rng=None):

        # We ignore mode, topo and targets because screw them!
        return HMDBIterator(self, batch_size, num_batches,
                            data_specs, return_tuple, rng)

    def has_targets(self):
        return True

    def get_topo_batch_axis(self):
        """
        Returns the index of the axis that corresponds to different examples
        in a batch when using topological_view.
        """
        return self.axes.index('b')

    def get_design_matrix(self, topo=None):
        return self.labels


def load_list(filename):
    id_list = []
    with open(filename, 'r') as fd:
        for line in fd:
            id_list.append(line)
    return id_list


class HMDBIterator:

    stochastic = False

    def __init__(self, dataset=None, batch_size=None, num_batches=None,
                 data_specs=False, return_tuple=False, rng=None):

        self._dataset = dataset
        self._dataset_size = len(dataset.labels)

        # Validate the inputs
        assert dataset is not None
        if batch_size is None and num_batches is None:
            raise ValueError("Provide at least one of batch_size or num_batches")
        if batch_size is None:
            batch_size = int(np.ceil(self._dataset_size / float(num_batches)))
        if num_batches is None:
            num_batches = np.ceil(self._dataset_size / float(batch_size))

        max_num_batches = np.ceil(self._dataset_size / float(batch_size))
        if num_batches > max_num_batches:
            raise ValueError("dataset of %d examples can only provide "
                             "%d batches with batch_size %d, but %d "
                             "batches were requested" %
                             (self._dataset_size, max_num_batches,
                              batch_size, num_batches))

        if rng is None:
            self._rng = random.Random(1)
        else:
            self._rng = rng

        self._batch_size = batch_size
        self._num_batches = int(num_batches)
        self._next_batch_no = 0
        self._batch_order = range(self._num_batches)
        #self._rng.shuffle(self._batch_order)
        self._return_tuple = return_tuple
        self._data_specs = data_specs
        self.num_examples = self._dataset_size # Needed by Dataset interface
        self.num_example = self.num_examples
        print self.num_examples


    def __iter__(self):
        return self

    def next(self):
        if self._next_batch_no >= self._num_batches:
            print self.num_example
            print self._num_batches
            raise StopIteration()
        else:
            # Determine minibatch start and end idx
            first = self._batch_order[self._next_batch_no] * self._batch_size
            if first + self._batch_size > self._dataset_size:
                last = self._dataset_size
            else:
                last = first + self._batch_size
            data =  self._dataset.get_minibatch(first, last,
                                                self._batch_order[self._next_batch_no],
                                                self._batch_size,
                                                self._data_specs,
                                                self._return_tuple)
            self._next_batch_no += 1
            return data






class HMDBMixValidDataset(dataset.Dataset):

    nbTags = 51
    vidShape = [426, 1, 100]
    vidSize = vidShape[0] * vidShape[1] * vidShape[2]
    mapper = {'train': 0, 'valid': 1, 'test': 2}

    def __init__(self,
                 hmdb_path, hmdb_split,
                 label_transfer_file,
                 nb_samples = 100,
                 axes = ('b', 0, 1, 'c')):

        self.__dict__.update(locals())

        # Not much to do in this fake dataset
        self._hmdb = HMDBDataset(hmdb_path, hmdb_path, hmdb_split,
                                  'valid', nb_samples, axes)
        self._dvd =  AtousaDataset('valid', nb_samples, axes)

        self.label_transfer = np.loadtxt(label_transfer_file)



    def iterator(self, mode=None, batch_size=None, num_batches=None, topo=None,
                 targets=False, data_specs=None, return_tuple=False, rng=None):

        # We ignore mode, topo and targets because screw them!
        return HMDBMixValidIterator(self, batch_size, num_batches,
                                    data_specs, return_tuple, rng)

    def has_targets(self):
        return True

    def get_topo_batch_axis(self):
        """
        Returns the index of the axis that corresponds to different examples
        in a batch when using topological_view.
        """
        return self.axes.index('b')

    def get_design_matrix(self, topo=None):
        return self._hmdb.labels


class HMDBMixValidIterator:

    stochastic = False

    def __init__(self, dataset=None, batch_size=None, num_batches=None,
                 data_specs=False, return_tuple=False, rng=None):

        self._hmdb_iterator = HMDBIterator(dataset._hmdb, batch_size,
                                           num_batches,
                                           data_specs, return_tuple,
                                           rng)

        self._dvd_iterator = AtousaDatasetIterator(dataset._dvd, batch_size,
                                                   num_batches,
                                                   data_specs, return_tuple,
                                                   rng)
        self.num_examples = self._hmdb_iterator.num_examples + self._dvd_iterator.num_examples
        self.cur = 0
        self.batch_size = batch_size
        self.dataset = dataset
        self.hmdb_stop = False
        self.dvd_stop = False

    def __iter__(self):
        return self

    def next(self):
        print(self.hmdb_stop, self.cur, self.num_examples)
        if not self.hmdb_stop and self.cur % 2 == 0:
            print "herehmdb"
            try:
                x,y = self._hmdb_iterator.next()
            except:
                self.hmdb_stop = True
                if self.dvd_stop:
                    raise StopIteration()
                else:
                    return self.next()
                
        else:
            try:
                print "heredvd"
                x, y_tmp = self._dvd_iterator.next() 
            except:
                self.dvd_stop = True
                if self.hmdb_stop:
                    raise StopIteration()
                else:
                    return self.next()

            ### Adapt labels
            y = np.zeros((self.batch_size, 51)).astype(config.floatX)
            for i in xrange(0, self.batch_size):
                for j in xrange(0, len(self.dataset.label_transfer)):
                    y[i, self.dataset.label_transfer[j, 1]] = y_tmp[i, self.dataset.label_transfer[j, 0]]
                

        self.cur = self.cur + 1
        return x, y



