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
from HumanActivityRecognition.space import Conv3DSpace

class UCF101Dataset(dataset.Dataset):


    def __init__(self, data_path, data_path2, split, dataSize, dataSize2, mutlipeAugment, num_split,nb_x, nb_y, nb_t, which_set,
                 batch_size = 48, axes = ('b', 'c', 't', 0, 1, )):

        ### Datasets parameters
		# UCF101
        self.nbTags = 101
		# HMDB51
        self.nbTags2 = 51
        self.nb_feats = 33
        self.nb_x = nb_x
        self.nb_y = nb_y
        self.nb_t = nb_t
        self.vidShape = [self.nb_x, self.nb_y, self.nb_t, self.nb_feats]
        #vidSize ('c' x 't' x 0 x 1) 
        self.vidSize = self.vidShape[0] * self.vidShape[1] * \
                       self.vidShape[2] * self.vidShape[3]
        self.mapper = {'train': 0, 'valid': 1, 'test': 2}
        self.axes = axes
        self.which_set = which_set

        self.data_path_base = data_path
        self.data_path_base2 = data_path2
        self.splits = split
        self.num_split = num_split
        self.batch_size = batch_size
		
		
        # path (UCF101)
        if which_set == 'train':  
           self.data_path = self.data_path_base + '/split%s' % str(num_split) + '/%s_1/' % self.splits
        elif which_set == 'valid':
           self.data_path = self.data_path_base + '/split%s' % str(num_split) + '/%s_0/' % self.splits
        else: # test
          self.data_path = self.data_path_base + '/split%s' % str(num_split) + '/%s_2/' % self.splits
		  
        # path (HMDB51)
        if which_set == 'train':  
           self.data_path2 = self.data_path_base2 + '/split%s' % str(num_split) + '/%s_1/' % self.splits
        elif which_set == 'valid':
           self.data_path2 = self.data_path_base2 + '/split%s' % str(num_split) + '/%s_0/' % self.splits
        else: # test
          self.data_path2 = self.data_path_base2 + '/split%s' % str(num_split) + '/%s_2/' % self.splits

		#UCF101  
        h5file1 = tables.openFile(os.path.join(self.data_path, 'features1.hdf'), mode = "r")
        self.data = h5file1.getNode('/', "denseCuboid")
        self.labels = np.loadtxt(os.path.join(self.data_path, 'labels1.txt'))
		#self.read_ids(os.path.join(self.data_path, 'files_lst.txt'))
        # number of videos examples in the dataset
        self.nb_examples = self.data.shape[0]
		
		#HMDB51
        h5file2 = tables.openFile(os.path.join(self.data_path2, 'features1.hdf'), mode = "r")
        self.data2 = h5file2.getNode('/', "denseFeat")
        self.labels2 = np.loadtxt(os.path.join(self.data_path2, 'labels1.txt'))
        #self.read_ids(os.path.join(self.data_path, 'files_lst.txt'))
        # number of videos examples in the dataset
        self.nb_examples2 = self.data2.shape[0]
		
        to_add2 =0 
        to_add = 0
        ### add sample to be divisible by batch_size (UCF101)
        self.orig_nb_examples = self.data.shape[0]
        if (self.nb_examples % self.batch_size != 0):
            to_add = self.batch_size - self.nb_examples % self.batch_size
            #self.data = np.append(self.data, self.data[0:to_add, :], axis = 0)
            #self.labels = np.append(self.labels, self.labels[0:to_add, :], axis = 0)
            if self.nb_examples == 19136: 
               self.labels = self.labels[0:19104, :]
           #for i in xrange(0, to_add):
            #    self.video_ids.append(self.video_ids[i])
        self.nb_examples = self.data.shape[0]
        print self.data.shape
		
        ### add sample to be divisible by batch_size
        self.orig_nb_examples2 = self.data2.shape[0] 
        if (self.nb_examples2 % self.batch_size != 0):
            to_add2 = self.batch_size - self.nb_examples2 % self.batch_size
            self.data2 = np.append(self.data2, self.data2[0:to_add2, :], axis = 0)
            self.labels2 = np.append(self.labels2, self.labels2[0:to_add2, :], axis = 0)
            #for i in xrange(0, to_add):
            #    self.video_ids.append(self.video_ids[i])
        self.nb_examples2 = self.data2.shape[0]
        print self.data2.shape
        #self.data = self.data.reshape(self.data.shape[0],
        #                              self.nb_t,
        #                              self.nb_x, 
        #                              self.nb_y,
        #                              self.nb_feats)

        ## Transform 'b', 't', 0, 1, 'c'  to  'b', 0, 1, 't', 'c' 
        #self.data  = np.swapaxes(self.data, 1, 2) # 'b, 0, 't', 1, 'c'
        #self.data  = np.swapaxes(self.data, 2, 3) # 'b, 0, 1, 't', 'c'		
		

        self.labelsLength = len(self.labels) + len(self.labels2)
        #if (self.labelsLength % self.batch_size != 0):
        #   to_add = self.batch_size - self.labelsLength % self.batch_size
        #   self.labelsLength = self.labelsLength + to_add
		
		# data_specs features (UCF101+ HMDB51), targets (101 classes UCF101), and second_targets (51 classes HMDB51)  
        #space_components = [Conv3DSpace(shape=(self.nb_x, self.nb_y, self.nb_t) , num_channels= self.nb_feats, axes=self.axes), VectorSpace(dim=self.nbTags), VectorSpace(dim=self.nbTags2)]
        #source_components = ['features', 'targets', 'second_targets']
        #space = CompositeSpace(space_components)
        #source = tuple(source_components)
        #self.data_specs = (space, source)
		
        
        
		
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
		   
        h5file1 = tables.openFile(os.path.join(self.data_path, 'features1.hdf'), mode = "r")
        self.data = h5file1.getNode('/', "denseCuboid")

        self.labels = np.loadtxt(os.path.join(self.data_path, 'labels1.txt'))
		   
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
        print self.data.shape
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



    def get_minibatch(self, firstIdxHMDB, lastIdxHMDB, firstIdxUCF, lastIdxUCF, batch_size,
                      data_specs, return_tuple):
 
        x = np.zeros([batch_size, self.data2.shape[1]] , dtype="float32")
        y = np.zeros([batch_size, self.nbTags],  dtype="float32")
        second_y = np.zeros([batch_size, self.nbTags2],  dtype="float32")
		
        #UCF
        dataTmp = self.data[firstIdxUCF:lastIdxUCF,0:1570800]
        # re-scale back to float32 (UCF101)
        dataTmp=dataTmp.astype("float32")
        dataTmp *= (1./255.)
        x[0:(lastIdxUCF -firstIdxUCF),:] = dataTmp
        #import pdb; pdb.set_trace()
        y[0:(lastIdxUCF -firstIdxUCF),:] = self.labels[firstIdxUCF:lastIdxUCF]
        second_y[0:(lastIdxUCF -firstIdxUCF),:] = 0.0	
        #HMDB
        dataTmp = self.data2[firstIdxHMDB:lastIdxHMDB]
        x[(lastIdxUCF -firstIdxUCF):,:] = dataTmp
        y[(lastIdxUCF -firstIdxUCF):,:] = 0.0;
        second_y[(lastIdxUCF -firstIdxUCF):,:] = self.labels2[firstIdxHMDB:lastIdxHMDB];
		
        y = np.concatenate((y, second_y), axis=1)
        x = x.reshape(batch_size, self.nb_t, self.nb_x, self.nb_y, self.nb_feats)
        x  = np.swapaxes(x, 1, 2) # 'b, 0, 't', 1, 'c'
        x  = np.swapaxes(x, 2, 3) # 'b, 0, 1, 't', 'c'
      
        print "Returned a minibatch", lastIdxHMDB, lastIdxUCF
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
        #if (shuffle_batch):
        #    self._rng.shuffle(self._batch_order)
        self._return_tuple = return_tuple
        self._data_specs = data_specs
        
        #import pdb; pdb.set_trace()
        self.num_examples = self._dataset_size*self._dataset.num_split 
        print self.num_examples
		
        # first split
        self._split = 1
        # last split
        self._splitEnd = self._dataset.num_split
        #self._timer14=time.time()
        self.lastHMDB = 0
        self.lastUCF = 0
        
    def __iter__(self):
        return self
        
    def next(self):
        #timer13 = 0
        if (self._next_batch_no >= self._num_batches and self._split==self._splitEnd):
            print self.num_examples
            print self._num_batches
            raise StopIteration()
            #print 'total  epochtime:' + str(time.time() - self._timer14)
        #iterate on ith split
        else:
            # upload the ith split
            #if (self._next_batch_no ==0):
            #    set = self._dataset.which_set
            #    spi = self._split
            #    #import pdb; pdb.set_trace()
            #    self._dataset.uploadSplit(spi, set)
            # Determine minibatch start and end idx
            if self._dataset.which_set =='train':
               if (self._next_batch_no < 289) :
                  firstHMDB = self.lastHMDB
                  lastHMDB = firstHMDB + 8
                  self.lastHMDB = lastHMDB
                  firstUCF = self.lastUCF
                  lastUCF = firstUCF + 40
                  self.lastUCF = lastUCF
               else :
                 firstHMDB = self.lastHMDB
                 lastHMDB = firstHMDB + 7
                 self.lastHMDB = lastHMDB
                 firstUCF = self.lastUCF
                 lastUCF = firstUCF + 41
                 self.lastUCF = lastUCF
            else:
              if (self._next_batch_no < 80) :
                 firstHMDB = self.lastHMDB
                 lastHMDB = firstHMDB + 14
                 self.lastHMDB = lastHMDB
                 firstUCF = self.lastUCF
                 lastUCF = firstUCF + 34
                 self.lastUCF = lastUCF
              else :
                 firstHMDB = self.lastHMDB
                 lastHMDB = firstHMDB + 13
                 self.lastHMDB = lastHMDB
                 firstUCF = self.lastUCF
                 lastUCF = firstUCF + 35
                 self.lastUCF = lastUCF
			
            data =  self._dataset.get_minibatch(firstHMDB, lastHMDB, firstUCF, lastUCF,
                                                self._batch_size,
                                                self._data_specs,
                                                self._return_tuple)
            #timer13 += time.time()
            #print 'computing 1 minibatch:' + str(timer13)
            self._next_batch_no += 1
            # check if it is end of split to read next split
            if (self._next_batch_no >= self._num_batches and self._split != self._splitEnd):
                self._split = self._split +1
                self._next_batch_no = 0
            #import pdb; pdb.set_trace() 
            return data
       
    
