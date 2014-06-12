
import math
import sys
import warnings
import theano
import numpy as np
from theano import config
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as T

from pylearn2.costs.mlp import Default
from pylearn2.expr.probabilistic_max_pooling import max_pool_channels
from pylearn2.linear import conv2d
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
from pylearn2.expr.nnet import pseudoinverse_softmax_numpy
from pylearn2.space import CompositeSpace
from pylearn2.space import Conv2DSpace
from pylearn2.space import Space
from pylearn2.space import VectorSpace
from pylearn2.utils import function
from pylearn2.utils import py_integer_types
from pylearn2.utils import safe_union
from pylearn2.utils import safe_zip
from pylearn2.utils import sharedX
from pylearn2.models.mlp import Layer, Linear, MLP

warnings.warn("MLP changing the recursion limit.")
# We need this to be high enough that the big theano graphs we make
# when doing max pooling via subtensors don't cause python to complain.
# python intentionally declares stack overflow well before the stack
# segment is actually exceeded. But we can't make this value too big
# either, or we'll get seg faults when the python interpreter really
# does go over the stack segment.
# IG encountered seg faults on eos3 (a machine at LISA labo) when using
# 50000 so for now it is set to 40000.
# I think the actual safe recursion limit can't be predicted in advance
# because you don't know how big of a stack frame each function will
# make, so there is not really a "correct" way to do this. Really the
# python interpreter should provide an option to raise the error
# precisely when you're going to exceed the stack segment.
sys.setrecursionlimit(40000)

# This sigmoid support tagging option which the original pylearn2 repo doesn't
# It also fixes a bug where the cost_matrix() function returns a matrix
# where cost is computed with mean-squared-error instead of binary_crossentropy
class Sigmoid(Linear):
    """
    Implementation of the sigmoid nonlinearity for MLP.
    """

    def __init__(self,dim,
                 layer_name, 
                 monitor_style = 'detection',
                 targets_accumulator_file = "y_accumulator.pkl.temp",
                 outputs_accumulator_file = "yhat_accumulator.pkl.temp",
                 dummy_channels = [],
                 irange=None,
                 istdev=None,
                 sparse_init=None,
                 sparse_stdev=1.,
                 include_prob=1.0,
                 init_bias=0.,
                 W_lr_scale=None,
                 b_lr_scale=None,
                 mask_weights=None,
                 max_row_norm=None,
                 max_col_norm=None,
                 min_col_norm=None,
                 softmax_columns=False,
                 copy_input=0,
                 use_abs_loss=False,
                 use_bias=True):
        """
            monitor_style: a string, either 'detection' or 'classification'
                           'detection' by default

                           if 'detection':
                               get_monitor_from_state makes no assumptions about
                               target, reports info about how good model is at
                               detecting positive bits.
                               This will monitor precision, recall, and F1 score
                               based on a detection threshold of 0.5. Note that
                               these quantities are computed *per-minibatch* and
                               averaged together. Unless your entire monitoring
                               dataset fits in one minibatch, this is not the same
                               as the true F1 score, etc., and will usually
                               seriously overestimate your performance.
                            if 'classification':
                                get_monitor_from_state assumes target is one-hot
                                class indicator, even though you're training the model
                                as k independent sigmoids. gives info on how good
                                the argmax is as a classifier
        """
        self.__dict__.update(locals())
        del self.self
        #self.b = sharedX(np.array([-0.7500,0.7640551]), name=(layer_name + '_b'))
        self.b = sharedX(np.array([-2.43,-2.20,-2.97,-2.65,-2.71,-3.20,-2.47,-1.82,-1.62,-1.93,-3.50,-1.65]), name=(layer_name + '_b'))
        
        assert monitor_style in ['classification', 'detection', 'tagging']
        self.monitor_style = monitor_style
        
        # Define an accumulator for the targets and the predictions of the
        # model.
        self.targets_accumulator_file = targets_accumulator_file
        self.targets_accumulator = DumpAccumulator(self.targets_accumulator_file)
        self.outputs_accumulator_file = outputs_accumulator_file
        self.outputs_accumulator = DumpAccumulator(self.outputs_accumulator_file)
        
        # Set the list of "fake" monitoring channels that the class must
        # set in prep
        self.dummy_channels = dummy_channels

    def fprop(self, state_below):
        p = self._linear_part(state_below)
        p = T.nnet.sigmoid(p)
        return p

    def kl(self, Y, Y_hat):
        """
        Returns a batch (vector) of
        mean across units of KL divergence for each example
        KL(P || Q) where P is defined by Y and Q is defined by Y_hat
        Currently Y must be purely binary. If it's not, you'll still
        get the right gradient, but the value in the monitoring channel
        will be wrong.
        Y_hat must be generated by fprop, i.e., it must be a symbolic
        sigmoid.

        p log p - p log q + (1-p) log (1-p) - (1-p) log (1-q)
        For binary p, some terms drop out:
        - p log q - (1-p) log (1-q)
        - p log sigmoid(z) - (1-p) log sigmoid(-z)
        p softplus(-z) + (1-p) softplus(z)
        """
        # Pull out the argument to the sigmoid
        assert hasattr(Y_hat, 'owner')
        owner = Y_hat.owner
        assert owner is not None
        op = owner.op

        if not hasattr(op, 'scalar_op'):
            raise ValueError("Expected Y_hat to be generated by an Elemwise op, got "+str(op)+" of type "+str(type(op)))
        assert isinstance(op.scalar_op, T.nnet.sigm.ScalarSigmoid)
        z ,= owner.inputs

        term_1 = Y * T.nnet.softplus(-z)
        term_2 = (1 - Y) * T.nnet.softplus(z)

        total = term_1 + term_2

        assert total.ndim == 2

        return total

    def cost_matrix(self, Y, Y_hat):
        import theano.tensor as T
        return T.nnet.binary_crossentropy(Y_hat, Y)
        
    def cost(self, Y, Y_hat):
        import theano.tensor as T
        return T.nnet.binary_crossentropy(Y_hat, Y).mean()
    
    def get_detection_channels_from_state(self, state, target):

        rval = OrderedDict()
        y_hat = state > 0.5
        y = target > 0.5
        wrong_bit = T.cast(T.neq(y, y_hat), state.dtype)
        rval['01_loss'] = wrong_bit.mean()
        rval['kl'] = self.cost(Y_hat=state, Y=target)

        y = T.cast(y, state.dtype)
        y_hat = T.cast(y_hat, state.dtype)
        tp = (y * y_hat).sum()
        fp = ((1-y) * y_hat).sum()
        precision = tp / T.maximum(1., tp + fp)
        recall = tp / T.maximum(1., y.sum())
        rval['precision'] = precision
        rval['recall'] = recall
        rval['f1'] = 2. * precision * recall / T.maximum(1, precision + recall)

        tp = (y * y_hat).sum(axis=0)
        fp = ((1-y) * y_hat).sum(axis=0)
        precision = tp / T.maximum(1., tp + fp)

        rval['per_output_precision.max'] = precision.max()
        rval['per_output_precision.mean'] = precision.mean()
        rval['per_output_precision.min'] = precision.min()

        recall = tp / T.maximum(1., y.sum(axis=0))

        rval['per_output_recall.max'] = recall.max()
        rval['per_output_recall.mean'] = recall.mean()
        rval['per_output_recall.min'] = recall.min()

        f1 = 2. * precision * recall / T.maximum(1, precision + recall)

        rval['per_output_f1.max'] = f1.max()
        rval['per_output_f1.mean'] = f1.mean()
        rval['per_output_f1.min'] = f1.min()

        return rval

    def get_tagging_channels_from_state(self, state, target):
        
        # Before using the state and targets, log them with the accumulator
        state = self.outputs_accumulator(state)
        target = self.targets_accumulator(target)


        missingValuesFilter = T.neq(target, -1)

        rval = OrderedDict()
        y_hat = state > 0.5
        y = target > 0.5
        wrong_bit = T.cast(T.neq(y, y_hat), state.dtype) * missingValuesFilter
        rval['mistagging'] = T.cast(wrong_bit.sum() / missingValuesFilter.sum(),
                                 state.dtype)

        y = T.cast(y, state.dtype)
        y_hat = T.cast(y_hat, state.dtype)
        tp = (y * y_hat * missingValuesFilter).sum()
        fp = ((1-y) * y_hat * missingValuesFilter).sum()
        precision = tp / T.maximum(1., tp + fp)
        recall = tp / T.maximum(1., (y * missingValuesFilter).sum())
        rval['precision'] = precision
        rval['recall'] = recall
        rval['f1'] = 2. * precision * recall / T.maximum(1, precision + recall)

        tp = (y * y_hat * missingValuesFilter).sum(axis=0)
        fp = ((1-y) * y_hat * missingValuesFilter).sum(axis=0)
        precision = tp / T.maximum(1., tp + fp)

        rval['per_output_precision.max'] = precision.max()
        rval['per_output_precision.mean'] = precision.mean()
        rval['per_output_precision.min'] = precision.min()

        recall = tp / T.maximum(1., (y * missingValuesFilter).sum(axis=0))

        rval['per_output_recall.max'] = recall.max()
        rval['per_output_recall.mean'] = recall.mean()
        rval['per_output_recall.min'] = recall.min()

        f1 = 2. * precision * recall / T.maximum(1, precision + recall)

        rval['per_output_f1.max'] = f1.max()
        rval['per_output_f1.mean'] = f1.mean()
        rval['per_output_f1.min'] = f1.min()
        
        # Define dummy channels with dummy values that will eventually receive
        # meanAvgPrec values from the TrainExtension that computes it
        for dummy in self.dummy_channels:
            rval[dummy] = f1.max() # Use f1.max() because it's already been
                                   # computed so it costs nothing
        
        # Add computation of the mean average recision
        #from pylearn2_ICML2014 import meanAvgPrec
        #(rval['min_avg_prec'],
        # rval['mean_avg_prec'],
        # rval['max_avg_prec']) = meanAvgPrec.meanAveragePrecisionTheano(target, state)

        return rval
        
    def get_monitoring_channels_from_state(self, state, target=None):

        rval = super(Sigmoid, self).get_monitoring_channels_from_state(state, target)

        if target is not None:
            if self.monitor_style == 'detection':
                rval.update(self.get_detection_channels_from_state(state, target))
            elif self.monitor_style == 'tagging':
                rval.update(self.get_tagging_channels_from_state(state, target))
            else:
                assert self.monitor_style == 'classification'

                Y = T.argmax(target, axis=1)
                Y_hat = T.argmax(state, axis=1)
                rval['misclass'] = T.cast(T.neq(Y, Y_hat), state.dtype).mean()

        return rval
        

import cPickle as Pickle
from theano.gof import Op, Apply     
 
# This identity-like Op print has a binary logging side effect.
class DumpAccumulator(Op):

    view_map = {0: [0]}

    def __init__(self, dumpFile):
        self.dumpFile = dumpFile
        
        # Init the dumpFile to a clean state to avoid screwing the results
        f = open(self.dumpFile, "w")
        Pickle.dump([], f)
        f.close() 

    def make_node(self, xin):
        xout = xin.type.make_variable()
        return Apply(op=self, inputs=[xin], outputs=[xout])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin
        self.dumpAccumulate(xin)
        
    def dumpAccumulate(self, data):
        
        # Get the current state of the dump
        currentDumpContent = []
        try:
            f = open(self.dumpFile, "r")
            currentDumpContent = Pickle.load(f)
            f.close()
        except IOError:
            pass # The file doesn't exist. No sweat, we'll create it later.
            
        # Update the dump state and write it to disk
        currentDumpContent.append(data)
        f = open(self.dumpFile, "w")
        Pickle.dump(currentDumpContent, f)
        f.close()
            
    def grad(self, input, output_gradients):
        return output_gradients

    def R_op(self, inputs, eval_points):
        return [x for x in eval_points]

    def __eq__(self, other):
        return (type(self) == type(other) and 
                self.dumpFile == other.dumpFile)

    def __hash__(self):
        return hash(self.dumpFile)

    def __setstate__(self, dct):
        self.__dict__.update(dct)

    def c_code_cache_version(self):
        return (1,)
        

from pylearn2.train_extensions import TrainExtension
from pylearn2_ECCV2014.meanAvgPrec import everyAveragePrecisionTheano as MapTheano
import theano
import theano.tensor as T

class MeanAveragePrecisionExtension(TrainExtension):
    
    
    def __init__(self, outputsFile, targetsFile,
                 channelNameMin = None, 
                 channelNameMean = None, 
                 channelNameMax = None,
                 channelNamesIndividual = None):
        self.__dict__.update(locals())
        del self.self
        
        # Compute a theano function to compute the mean
        # average precision
        x = T.fmatrix()
        y = T.fmatrix()
        area = MapTheano(x, y)
        self.fct = theano.function([x, y], area)
        
    def on_monitor(self, model, dataset, algorithm):
        
        # Figure out the names and sizes of the monitoring datasets
        # used
        datasets = algorithm.monitoring_dataset
        datasets_names = datasets.keys()
        datasets_desc = {}
        for name in datasets_names:
            datasets_desc[name] = datasets[name].nbExamples
        
        # Load the predictions and the targets
        f = open(self.outputsFile, "r")
        predictions = self.collate(Pickle.load(f), datasets_desc)
        f.close()
        
        f = open(self.targetsFile, "r")
        targets = self.collate(Pickle.load(f), datasets_desc)
        f.close()
        
        # Compute average precisions and set the channels for every 
        # monitoring dataset
        for name in datasets_names:
            
            # Add a prefix consisting of the dataset name to channel names
            channelMin = self.formatChannels(self.channelNameMin, name)
            channelMean = self.formatChannels(self.channelNameMean, name)
            channelMax = self.formatChannels(self.channelNameMax, name)
            channelInd = self.formatChannels(self.channelNamesIndividual, name)
            
            import time
            start = time.time()
            MAPs = self.fct(targets[name], predictions[name])
            print time.time() - start
            
            channelKeys = model.monitor.channels.keys()
            
            if channelMin != None and channelMin in channelKeys:
                self.logChannel(channelMin, MAPs.min())
                model.monitor.channels[channelMin].val_record[-1] = MAPs.min()
                
            if channelMean != None and channelMean in channelKeys:
                self.logChannel(channelMean, MAPs.mean())
                model.monitor.channels[channelMean].val_record[-1] = MAPs.mean() 
                
            if channelMax != None and channelMax in channelKeys:
                self.logChannel(channelMax, MAPs.max())
                model.monitor.channels[channelMax].val_record[-1] = MAPs.max()
                
            if (channelInd != None and len(channelInd) == len(MAPs) and
                self.allXsAreInY(channelInd, channelKeys)):
                    
                for idx in range(len(MAPs)):
                    channelName = channelInd[idx]
                    self.logChannel(channelName, MAPs[idx])
                    model.monitor.channels[channelName].val_record[-1] = MAPs[idx]            
                
        # Reset the state of the outputsFile and the targetsFile
        # to make them ready for the next epoch
        f = open(self.outputsFile, "w")
        Pickle.dump([], f)
        f.close()
        
        f = open(self.targetsFile, "w")
        Pickle.dump([], f)
        f.close()
        
    def logChannel(self, channel, value):
        print "%s : %f" % (channel, value)
        
    def formatChannels(self, channels, datasetName):
        if channels == None:
            return channels
        elif isinstance(channels, str):
            return datasetName + "_" + channels
        elif isinstance(channels, list):
            return [self.formatChannels(c, datasetName) for c in channels]
        else:
            assert False
        
        
    def allXsAreInY(self, Xs, Y):
        nbXsInY = sum([x in Y for x in Xs])
        nbXs = len(Xs)
        return nbXsInY == nbXs
    
    def collate(self, data, datasets_desc):
        
        output = {}
        dataIdx = 0
        
        for name in datasets_desc.keys():
            nbCases = datasets_desc[name]
            nbClasses = data[0].shape[1]
            
            output[name] = np.zeros((nbCases, nbClasses), 
                                    dtype=theano.config.floatX)
            
            datasetIdx = 0
            keepGoing = True
            while datasetIdx != nbCases:
                start = datasetIdx
                try:
                  end = start + data[dataIdx].shape[0]
                except:
                 import pdb
                 pdb.set_trace()
                output[name][start:end] = data[dataIdx]
                
                datasetIdx = end
                dataIdx += 1
            
        return output