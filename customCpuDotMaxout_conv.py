"""
Multilayer Perceptron
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow", "David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import math
import sys
import warnings

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
from pylearn2.models.mlp import Layer
#from pylearn2.monitor import get_monitor_doc
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
from pylearn2.utils import wraps

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


class CpuConvMaxout(Layer):
    """
    .. todo::

        WRITEME
    """
    def __init__(self,
                 output_channels,
                 num_pieces,
                 kernel_shape,
                 pool_shape,
                 pool_stride,
                 layer_name,
                 irange=None,
                 border_mode='valid',
                 sparse_init=None,
                 include_prob=1.0,
                 init_bias=0.,
                 W_lr_scale=None,
                 b_lr_scale=None,
                 left_slope=0.0,
                 max_kernel_norm=None,
                 pool_type='max',
                 detector_normalization=None,
                 output_normalization=None,
                 kernel_stride=(1, 1)):
        """
        .. todo::

            WRITEME properly

         output_channels: The number of output channels the layer should have.
         kernel_shape: The shape of the convolution kernel.
         pool_shape: The shape of the spatial max pooling. A two-tuple of ints.
         pool_stride: The stride of the spatial max pooling. Also must be
                      square.
         layer_name: A name for this layer that will be prepended to
                     monitoring channels related to this layer.
         irange: if specified, initializes each weight randomly in
                 U(-irange, irange)
         border_mode: A string indicating the size of the output:
            full - The output is the full discrete linear convolution of the
                   inputs.
            valid - The output consists only of those elements that do not rely
                    on the zero-padding.(Default)
         include_prob: probability of including a weight element in the set
                       of weights initialized to U(-irange, irange). If not
                       included it is initialized to 0.
         init_bias: All biases are initialized to this number
         W_lr_scale: The learning rate on the weights for this layer is
                     multiplied by this scaling factor
         b_lr_scale: The learning rate on the biases for this layer is
                     multiplied by this scaling factor
         left_slope: **TODO**
         max_kernel_norm: If specifed, each kernel is constrained to have at
                          most this norm.
         pool_type: The type of the pooling operation performed the the
                    convolution. Default pooling type is max-pooling.
         detector_normalization, output_normalization:
              if specified, should be a callable object. the state of the
              network is optionally replaced with normalization(state) at each
              of the 3 points in processing:
                  detector: the maxout units can be normalized prior to the
                            spatial pooling
                  output: the output of the layer, after sptial pooling, can
                          be normalized as well
         kernel_stride: The stride of the convolution kernel. A two-tuple of
                        ints.
        """

        #super(ConvRectifiedLinear, self).__init__()

        if (irange is None) and (sparse_init is None):
            raise AssertionError("You should specify either irange or "
                                 "sparse_init when calling the constructor of "
                                 "ConvRectifiedLinear.")
        elif (irange is not None) and (sparse_init is not None):
            raise AssertionError("You should specify either irange or "
                                 "sparse_init when calling the constructor of "
                                 "ConvRectifiedLinear and not both.")

        self.__dict__.update(locals())
        del self.self

    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        self.input_space = space

        if not isinstance(space, Conv2DSpace):
            raise BadInputSpaceError("ConvRectifiedLinear.set_input_space "
                                     "expected a Conv2DSpace, got " +
                                     str(space) + " of type " +
                                     str(type(space)))

        rng = self.mlp.rng

        if self.border_mode == 'valid':
            output_shape = [(self.input_space.shape[0]-self.kernel_shape[0]) /
                            self.kernel_stride[0] + 1,
                            (self.input_space.shape[1]-self.kernel_shape[1]) /
                            self.kernel_stride[1] + 1]
        elif self.border_mode == 'full':
            output_shape = [(self.input_space.shape[0]+self.kernel_shape[0]) /
                            self.kernel_stride[0] - 1,
                            (self.input_space.shape[1]+self.kernel_shape[1]) /
                            self.kernel_stride[1] - 1]

        self.detector_space = Conv2DSpace(shape=output_shape,
                                          num_channels=self.output_channels,
                                          axes=('b', 'c', 0, 1))

        if self.irange is not None:
            assert self.sparse_init is None
            self.transformer = conv2d.make_random_conv2D(
                irange=self.irange,
                input_space=self.input_space,
                output_space=self.detector_space,
                kernel_shape=self.kernel_shape,
                batch_size=self.mlp.batch_size,
                subsample=self.kernel_stride,
                border_mode=self.border_mode,
                rng=rng)
        elif self.sparse_init is not None:
            self.transformer = conv2d.make_sparse_random_conv2D(
                num_nonzero=self.sparse_init,
                input_space=self.input_space,
                output_space=self.detector_space,
                kernel_shape=self.kernel_shape,
                batch_size=self.mlp.batch_size,
                subsample=self.kernel_stride,
                border_mode=self.border_mode,
                rng=rng)

        W, = self.transformer.get_params()
        W.name = 'W'
        self.b = sharedX(np.zeros(((self.num_pieces*self.output_channels),)) + self.init_bias)
        self.b.name = 'b'

        print 'Input shape: ', self.input_space.shape
        print 'Detector space: ', self.detector_space.shape

        assert self.pool_type in ['max', 'mean']

        dummy_batch_size = self.mlp.batch_size
        if dummy_batch_size is None:
            dummy_batch_size = 2
        dummy_detector = sharedX(
            self.detector_space.get_origin_batch(dummy_batch_size))
            
            
        #dummy_p = dummy_p.eval()
        self.output_space = Conv2DSpace(shape=[400, 1],num_channels=self.output_channels,axes=('b', 'c', 0, 1))
										
        W = rng.uniform(-self.irange,self.irange,(426, (self.num_pieces*self.output_channels)))
        W = sharedX(W)
        W.name = self.layer_name + "_w"
        self.transformer = MatrixMul(W)
		
        print 'Output space: ', self.output_space.shape

    @wraps(Layer.censor_updates)
    def censor_updates(self, updates):
        """
        .. todo::

            WRITEME
        """

        if self.max_kernel_norm is not None:
            W, = self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=(1)))
                desired_norms = T.clip(row_norms, 0, self.max_kernel_norm)
                scales = desired_norms / (1e-7 + row_norms)
                updates[W] = updated_W * scales.dimshuffle(0, 'x')

    @wraps(Layer.get_params)
    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        assert self.b.name is not None
        W, = self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * abs(W).sum()

    @wraps(Layer.set_weights)
    def set_weights(self, weights):

        W, = self.transformer.get_params()
        W.set_value(weights)

    @wraps(Layer.set_biases)
    def set_biases(self, biases):

        self.b.set_value(biases)

    @wraps(Layer.get_biases)
    def get_biases(self):

        return self.b.get_value()

    @wraps(Layer.get_weights_format)
    def get_weights_format(self):

        return ('v', 'h')

    @wraps(Layer.get_weights_topo)
    def get_weights_topo(self):

        outp, inp, rows, cols = range(4)
        raw = self.transformer._filters.get_value()

        return np.transpose(raw, (outp, rows, cols, inp))

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):

        W, = self.transformer.get_params()

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=(1)))

        return OrderedDict([('kernel_norms_min',  row_norms.min()),
                            ('kernel_norms_mean', row_norms.mean()),
                            ('kernel_norms_max',  row_norms.max()), ])

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        self.input_space.validate(state_below)
        axes = self.input_space.axes
        #z = self.transformer.lmul(state_below) + self.b
        state_below = state_below.dimshuffle(3,1,2,0)	
        z = self.transformer.lmul(state_below) +self.b
        z = z.dimshuffle(0,3,1,2)

        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

	    #ReLUs
        d = T.maximum(z, 0)
		
        # Max pooling between linear pieces
        # d = None
        # for i in xrange(self.num_pieces):
            # t = z[:,i::self.num_pieces,:,:]
            # if d is None:
                # d = t
            # else:
                # d = T.maximum(d, t)

        self.detector_space.validate(d)

        if not hasattr(self, 'detector_normalization'):
            self.detector_normalization = None

        if self.detector_normalization:
            d = self.detector_normalization(d)

        # NOTE : Custom pooling
        p = d.max(3)[:,:,:,None]

        self.output_space.validate(p)

        if not hasattr(self, 'output_normalization'):
            self.output_normalization = None

        if self.output_normalization:
            p = self.output_normalization(p)

        return p




class Pooling0(Layer):
    """
    .. todo::

        WRITEME
    """
    def __init__(self,
                 layer_name,
                 pool_type='max',
                 output_normalization=None):
        """
        .. todo::

            WRITEME properly

        """

        self.__dict__.update(locals())
        del self.self


    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        self.input_space = space

        if not isinstance(space, Conv2DSpace):
            raise BadInputSpaceError("ConvRectifiedLinear.set_input_space "
                                     "expected a Conv2DSpace, got " +
                                     str(space) + " of type " +
                                     str(type(space)))

        rng = self.mlp.rng

        print (self.input_space.shape)
        output_shape = [1, self.input_space.shape[1]]
        self.output_space = Conv2DSpace(shape=output_shape,
                                        num_channels = self.input_space.num_channels,
                                        axes=('b', 'c', 0, 1))
        assert self.pool_type in ['max', 'mean']


    @wraps(Layer.censor_updates)
    def censor_updates(self, updates):
        None

    @wraps(Layer.get_params)
    def get_params(self):
        rval = OrderedDict()
        return rval

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):
        return 0

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):
        return 0

    @wraps(Layer.set_weights)
    def set_weights(self, weights):
        None
    @wraps(Layer.set_biases)
    def set_biases(self, biases):
        None

    @wraps(Layer.get_biases)
    def get_biases(self):
        return 0

    @wraps(Layer.get_weights_format)
    def get_weights_format(self):
        return ('v', 'h')

    @wraps(Layer.get_weights_topo)
    def get_weights_topo(self):
        return None

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):
        return OrderedDict()

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        self.input_space.validate(state_below)
        axes = self.input_space.axes

        #p = state_below
        # NOTE : Custom pooling
        if (self.pool_type == 'mean'):
            p = state_below.average(2)[:,:,:,None]
        else:
            p = state_below.max(2)[:,:,:,None]

        self.output_space.validate(p)

        if not hasattr(self, 'output_normalization'):
            self.output_normalization = None

        if self.output_normalization:
            p = self.output_normalization(p)

        return p
