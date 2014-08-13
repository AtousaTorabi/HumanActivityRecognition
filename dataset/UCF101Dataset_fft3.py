import os
import math
import warnings
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

class UCF101Dataset(dataset.Dataset):


    def __init__(self, data_path, split, num_split, which_set,
                 batch_size = 64, axes = ('b', 'c', 't', 0, 1, )):

        ### Datasets parameters
        self.nbTags = 101
        self.nb_feats = 33
        self.nb_x = 20
        self.nb_y = 20
        self.nb_t = 12
        self.vidShape = [self.nb_x, self.nb_y, self.nb_t, self.nb_feats]
        #vidSize ('c' x 't' x 0 x 1) 
        self.vidSize = self.vidShape[0] * self.vidShape[1] * \
                       self.vidShape[2] * self.vidShape[3]
        self.mapper = {'train': 0, 'valid': 1, 'test': 2}
        self.axes = axes
        self.which_set = which_set

        self.data_path_base = data_path
        self.splits = split
        self.num_split = num_split
        self.batch_size = batch_size
        # determine label length
        if which_set == 'train':  
           path = self.data_path_base + '/split1/%s_1/' % self.splits
        elif which_set == 'valid':
           path = self.data_path_base + '/split1/%s_0/' % self.splits
        else: # test
           path = self.data_path_base + '/split1/%s_2/' % self.splits
        labels = np.loadtxt(os.path.join(path, 'labels.txt'))
        self.labelsLength = len(labels)
        if (self.labelsLength % self.batch_size != 0):
           to_add = self.batch_size - self.labelsLength % self.batch_size
           self.labelsLength = self.labelsLength + to_add
    def get_num_examples(self):
        
        
        data_num_examples = self.num_split*self.labelsLength
        return data_num_examples    

    def uploadSplit(self, s, Whichset):
	
        ### Load data
        if Whichset == 'train':  
          self.data_path = self.data_path_base + '/split%s' % str(s) + '/%s_1/' % self.splits
        elif Whichset == 'valid':
           self.data_path = self.data_path_base + '/split%s' % str(s) + '/%s_0/' % self.splits
        else: # test
           self.data_path = self.data_path_base + '/split%s' % str(s) + '/%s_2/' % self.splits
		   
        self.data = np.load(os.path.join(self.data_path, 'features.py.npy'))
        self.labels = np.loadtxt(os.path.join(self.data_path, 'labels.txt'))
        self.read_ids(os.path.join(self.data_path, 'files_lst.txt'))
        # number of videos examples in the dataset
        self.nb_examples = self.data.shape[0]
        ### add sample to be divisible by batch_size
        self.orig_nb_examples = self.data.shape[0]
        self.nb_examples = self.data.shape[0]
        if (self.nb_examples % self.batch_size != 0):
            to_add = self.batch_size - self.nb_examples % self.batch_size
            self.data = np.append(self.data, self.data[0:to_add, :], axis = 0)
            self.labels = np.append(self.labels, self.labels[0:to_add, :], axis = 0)
            for i in xrange(0, to_add):
                self.video_ids.append(self.video_ids[i])
		
        self.data = self.data.reshape(self.data.shape[0],
                                      self.nb_t,
                                      self.nb_x, 
                                      self.nb_y,
                                      self.nb_feats)

        ## Transform 'b', 't', 0, 1, 'c'  to  'b', 0, 1, 't', 'c' 
        self.data  = np.swapaxes(self.data, 1, 2) # 'b, 0, 't', 1, 'c'
        self.data  = np.swapaxes(self.data, 2, 3) # 'b, 0, 1, 't', 'c'
         
    def read_ids(self, filename):
        self.video_ids = []
        with open(filename) as fp:
            for line in fp:
                self.video_ids.append(line.strip())



    def get_minibatch(self, firstIdx, lastIdx, batch_size,
                      data_specs, return_tuple):
        
        x = np.zeros([lastIdx-firstIdx,] + self.vidShape , dtype="float32")
        y = np.zeros([lastIdx-firstIdx, self.nbTags],  dtype="float32")
     
	    
        # Return a batch ('b', 0, 1, 't','c')
        x[0:batch_size,:,:,:,:] = self.data[firstIdx:lastIdx]

        y[0:batch_size,:] = self.labels[firstIdx:lastIdx];
 
        print "Returned a minibatch", firstIdx, lastIdx
        return x, y
                                    
    def iterator(self, mode=None, batch_size=None, num_batches=None, topo=None,
                 targets=False, data_specs=None, return_tuple=False, rng=None,
                 shuffle_batch=True):
        return HMDBfftIterator(self, batch_size, num_batches, 
                               data_specs, return_tuple, rng,
                               shuffle_batch)
    
    
    def has_targets(self):
        return True
    def get_design_matrix(self, topo=None):
        return self.data
        
    def get_topo_batch_axis(self):
        """
        Returns the index of the axis that corresponds to different examples
        in a batch when using topological_view.
        """
        return self.axes.index('b')

class HMDBfftIterator:
    
    stochastic = False
    
    def __init__(self, dataset=None, batch_size=None, num_batches=None,
                 data_specs=False, return_tuple=False, rng=None,
                 shuffle_batch=True):
        
        self._dataset = dataset
        self._dataset_size = dataset.labelsLength

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
        if (shuffle_batch):
            self._rng.shuffle(self._batch_order)
        self._return_tuple = return_tuple
        self._data_specs = data_specs
        self.num_examples = self._dataset_size*self._dataset.num_split 
        print self.num_examples
		
        # first split
        self._split = 1
        # last split
        self._splitEnd = self._dataset.num_split
        
    def __iter__(self):
        return self
        
    def next(self):
        if (self._next_batch_no >= self._num_batches and self._split==self._splitEnd):
            print self.num_examples
            print self._num_batches
            raise StopIteration()
        #iterate on ith split
        else:
            # upload the ith split
            if (self._next_batch_no ==0):
                set = self._dataset.which_set
                spi = self._split
                #import pdb; pdb.set_trace()
                self._dataset.uploadSplit(spi, set)
            # Determine minibatch start and end idx
            first = self._batch_order[self._next_batch_no] * self._batch_size
            if first + self._batch_size > self._dataset_size:
                last = self._dataset_size
            else:
                last = first + self._batch_size
            data =  self._dataset.get_minibatch(first, last,
                                                self._batch_size,
                                                self._data_specs,
                                                self._return_tuple)
            self._next_batch_no += 1
            # check if it is end of split to read next split
            if (self._next_batch_no >= self._num_batches and self._split != self._splitEnd):
                self._split = self._split +1
                self._next_batch_no = 0
            return data
       
    
