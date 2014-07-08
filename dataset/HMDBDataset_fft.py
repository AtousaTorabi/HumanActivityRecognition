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
import numpy
from theano import config
from pylearn2.datasets import dataset
from pylearn2.utils.serial import load
from pylearn2.utils.string_utils import preprocess
from pylearn2.space import VectorSpace, CompositeSpace

class HMDBfftDataset(dataset.Dataset):

    def __init__(self, data_path, split, which_set, axes = ('b', 0, 1, 't','c')):

        ### Datasets parameters
        self.nbTags = 51
        self.nb_feats = 33
        self.nb_x = 20
        self.nb_y = 16 
        self.nb_t = 12
        self.vidShape = [nb_x, nb_y, nb_t, nb_feats]
        #vidSize : 20(0)x16(1)x12('t')x33('c') 
        self.vidSize = vidShape[0] * vidShape[1] * vidShape[2] * vidShape[3]
        self.mapper = {'train': 0, 'valid': 1, 'test': 2}
        self.axes = axes
        self.which_set = which_set

        ### Load data
        if which_set == 'train':
            self.data_path = data_path + '/%s_1/' % split
        elif which_set == 'valid':
           self.data_path = data_path + '/%s_0/' % split
        else: # test
           self.data_path = data_path + '/%s_2/' % split
            
        self.data = numpy.load(os.path.join(self.data_path, 'feature.npy'))
        self.labels = np.load(os.path.join(data_path, 'labels.txt'))
        self.nb_examples = self.labels.shape[0]

        # Reshape to (6755X20X16X33)
        self.data = datatmp.reshape(self.data.shape[0],
                                    self.nb_x, 
                                    self.nb_y, 
                                    self.nb_t,
                                    self.nb_feats)


    def get_minibatch(self, firstIdx, lastIdx, batch_size,
                      data_specs, return_tuple):
        
        x = numpy.zeros([lastIdx-firstIdx,] + self.vidShape , dtype="float32")
        y = numpy.zeros([lastIdx-firstIdx, self.nbTags],  dtype="float32")
     
	    
        # Return a batch ('b', 0, 1, 't','c')
        x[0:batch_size,:,:,:,:] = self.data[firstIdx:lastIdx]
        y[0:batch_size,:] = self.labels[firstIdx:lastIdx];
 
        print "Returned a minibatch"
        return x, y
                                    
    def iterator(self, mode=None, batch_size=None, num_batches=None, topo=None,
                 targets=False, data_specs=None, return_tuple=False, rng=None):
        return HMDBfftIterator(self, batch_size, num_batches, 
                                     data_specs, return_tuple, rng)
    
    def has_targets(self):
        return True
        
    def get_topo_batch_axis(self):
        """
        Returns the index of the axis that corresponds to different examples
        in a batch when using topological_view.
        """
        return self.axes.index('b')

class HMDBfftIterator:
    
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
        self._rng.shuffle(self._batch_order)
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
            
        
    
