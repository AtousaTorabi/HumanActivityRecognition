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

class AtousaDataset(dataset.Dataset):

    nbExamples = 1707
    #nbSamples=100
    nbTags = 12
    vidShape = [426, 1, 100]
    #vidSize = 1

    mapper = {'train': 0, 'valid': 1, 'test': 2}
    data_path = '/Tmp/torabi/ImprovedfeatureHOHA2HDF/'

    def __init__(self, which_set, data_path, nbSamples, axes = ('b', 0, 1, 'c')):

        # Not much to do in this fake dataset
        self.vidShape = [426, 1, nbSamples]
        self.vidSize = self.vidShape[0] * self.vidShape[1] * self.vidShape[2]
        self.which_set = which_set
        self.nbSamples = nbSamples
        self.axes = axes		
        if self.which_set =='train':
            self.data_path = data_path + '/train/'
            file_n = self.data_path +'train.txt'
            self.Index=numpy.loadtxt(file_n)
            self.nbExamples = len(self.Index) - (len(self.Index) % 64)
            
        elif self.which_set =='test':
            self.data_path = data_path + '/test/'
            file_n = self.data_path +'test.txt'
            self.Index=numpy.loadtxt(file_n)
            self.nbExamples = len(self.Index) - (len(self.Index) % 64)
        else :
            self.data_path = data_path + '/valid/'
            file_n = self.data_path +'valid.txt'
            self.Index=numpy.loadtxt(file_n)
            self.nbExamples = len(self.Index) - (len(self.Index) % 64)
            
        file_n2 = self.data_path + 'Index.txt'
        self.Interval = numpy.loadtxt(file_n2)
        file_n3 = data_path +'/HOHA2Tags.txt'
        self.labels = numpy.loadtxt(file_n3)

    def get_minibatch(self, firstIdx, lastIdx, data_specs, return_tuple):
        
        x = numpy.zeros(self.vidShape + [lastIdx-firstIdx,], dtype="float32")
        y = numpy.zeros([lastIdx-firstIdx, self.nbTags],  dtype="float32")
		
        timer10 = 0
        timer11 = 0
        timer12 = 0
		
        next_batch_no = lastIdx/64
        buf1 = "%07d" %next_batch_no
        file_n = self.data_path + str(buf1)+'.hdf'
        #timer10 -= time.time()
        self.h5file = tables.openFile(file_n, mode = "r")
        #timer10 += time.time()
        #timer11 -= time.time()
        data = self.h5file.getNode('/', "denseFeat")
        #timer11 += time.time()
        #timer12 -= time.time()
        for num1 in range(firstIdx, lastIdx):
            start = self.Interval[num1][0]
            end = self.Interval[num1][1]
            num=self.Index[num1]
            #subFolder = self.data_path + str(int(math.floor(num/10000))) + "/"
            s= end - start
            #print file_n
            #print firstIdx
            #print lastIdx
            batchIdx = num1 - firstIdx
            
            if s < self.nbSamples:
                datatmp = data[start:end,:]
                t= int(math.ceil(self.nbSamples/s))
                dd=datatmp
                for k in range(0,t):
                     datatmp = numpy.append(datatmp,dd,axis=0)
                numpy.random.shuffle(datatmp)       
                R= random.randint(1,(len(datatmp)-self.nbSamples)) 
                datatmp=datatmp[R:R+self.nbSamples, 10:436]
            else:
                try:
                   R= random.randint(1,(s-self.nbSamples))
                   datatmp=data[start+R:start+R+self.nbSamples, 10:436]
                except:
                   import pdb
                   pdb.set_trace()
            #try:				   
            x[:,:,:,batchIdx] = datatmp.transpose()[:,None,:]
            #except:
            #     import pdb
            #     pdb.set_trace()
            y[batchIdx,:] = self.labels[num-1];
        #timer12 += time.time()
        self.h5file.close()
            
        # reshape the matrix 100X427 to 427X1X100
        #x = numpy.zeros(self.vidShape + [lastIdx-firstIdx,], dtype="float32")
        #y = numpy.zeros([lastIdx-firstIdx, self.nbTags],  dtype="float32")
        print "Returned a minibatch"
        #print timer10, timer11, timer12
        return x, y
                                    
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
                batch_size = int(numpy.ceil(self._dataset_size / float(num_batches)))
            else:
                raise ValueError("Provide at least one of batch_size or num_batches")
        elif batch_size is not None:
            if num_batches is not None:
                max_num_batches = numpy.ceil(self._dataset_size / float(batch_size))
                if num_batches > max_num_batches:
                    raise ValueError("dataset of %d examples can only provide "
                                     "%d batches with batch_size %d, but %d "
                                     "batches were requested" %
                                     (self._dataset_size, max_num_batches,
                                      batch_size, num_batches))
            else:
                num_batches = numpy.ceil(self._dataset_size / float(batch_size))
        
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
            data =  self._dataset.get_minibatch(first, last, self._data_specs,
                                                self._return_tuple)
            timer13 += time.time()
            print self._next_batch_no
            #print timer13
            
            #print time.time() - self._timer14
            self._timer14 = time.time()
            self._next_batch_no += 1
            return data
            
        
    
