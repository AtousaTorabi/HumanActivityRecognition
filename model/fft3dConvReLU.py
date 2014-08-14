"""
Multilayer Perceptron
"""
#__authors__ = "Ian Goodfellow"
#__copyright__ = "Copyright 2012-2013, Universite de Montreal"
#__credits__ = ["Ian Goodfellow", "David Warde-Farley"]
#__license__ = "3-clause BSD"
#__maintainer__ = "Ian Goodfellow"

import math
import sys
import warnings
import functools


import numpy as np
from theano import config
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values
from theano.printing import Print
from theano.sandbox import cuda
import theano.tensor as T

from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
from pylearn2.models.mlp import Layer
from pylearn2.utils import sharedX

from pylearn2.space import CompositeSpace
from pylearn2.space import Space
from pylearn2.space import VectorSpace, Conv2DSpace
from pylearn2.utils import function
from pylearn2.utils import py_integer_types

from pylearn2.utils import sharedX

# setup detector layer for fft 3d convolution with axes bct01
from HumanActivityRecognition.space import Conv3DSpace

from HumanActivityRecognition.model.conv3d_bct01 import setup_detector_layer_bct01
from HumanActivityRecognition.linear.conv3d_b01tc import setup_detector_layer_b01tc
if cuda.cuda_available:
    from pylearn2.sandbox.cuda_convnet.pool import max_pool_c01b
from pylearn2.models.mlp import max_pool_c01b as temporal_max_pool_c01b
from pylearn2.linear import local_c01b
from pylearn2.sandbox.cuda_convnet import check_cuda

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


class fft3dConvReLUPool(Layer):

    def __init__(self,
                 num_channels,
                 num_pieces,
                 kernel_shape,
                 pool_shape,
                 pool_stride,
                 layer_name,
                 irange = None,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 pad = 0,
                 fix_pool_shape = False,
                 fix_pool_stride = False,
                 fix_kernel_shape = False,
                 partial_sum = 1,
                 tied_b = False,
                 max_kernel_norm = None,
                 input_normalization = None,
                 detector_normalization = None,
                 min_zero = False,
                 output_normalization = None,
                 kernel_stride = (1, 1, 1)):
        """
            layer_name: A name for this layer that will be prepended to
                        monitoring channels related to this layer.
            num_channels: The number of output channels the layer should have.
                          Note that it must internally compute num_channels * num_pieces
                          convolution channels.
            num_pieces:   The number of linear pieces used to make each maxout unit.
            kernel_shape: The shape of the convolution kernel.
            pool_shape:   The shape of the spatial max pooling. A three-tuple of ints.
            pool_stride:  The stride of the spatial and temporal max pooling.
            irange: if specified, initializes each weight randomly in
                U(-irange, irange)
            init_bias: All biases are initialized to this number
            W_lr_scale: The learning rate on the weights for this layer is
                multiplied by this scaling factor
            b_lr_scale: The learning rate on the biases for this layer is
                multiplied by this scaling factor
            pad: The amount of zero-padding to implicitly add to the boundary of the
                image when computing the convolution. Useful for making sure pixels
                at the edge still get to influence multiple hidden units.
            fix_pool_shape: If True, will modify self.pool_shape to avoid having
                pool shape bigger than the entire detector layer.
                If you have this on, you should probably also have
                fix_pool_stride on, since the pool shape might shrink
                smaller than the stride, even if the stride was initially
                valid.
                The "fix" parameters are useful for working with a hyperparameter
                optimization package, which might often propose sets of hyperparameters
                that are not feasible, but can easily be projected back into the feasible
                set.
            fix_kernel_shape: if True, will modify self.kernel_shape to avoid
            having the kernel shape bigger than the implicitly
            zero padded input layer

            partial_sum: a parameter that controls whether to prefer runtime savings
                        or memory savings when computing the gradient with respect to
                        the kernels. See pylearn2.sandbox.cuda_convnet.weight_acts.py
                        for details. The default is to prefer high speed.
                        Note that changing this setting may change the value of computed
                        results slightly due to different rounding error.
            tied_b: If true, all biases in the same channel are constrained to be the same
                    as each other. Otherwise, each bias at each location is learned independently.
            max_kernel_norm: If specifed, each kernel is constrained to have at most this norm.
            input_normalization, detector_normalization, output_normalization:
                if specified, should be a callable object. the state of the network is optionally
                replaced with normalization(state) at each of the 3 points in processing:
                    input: the input the layer receives can be normalized right away
                    detector: the maxout units can be normalized prior to the spatial pooling
                    output: the output of the layer, after sptial pooling, can be normalized as well
            kernel_stride: vertical,  horizontal and time pixel stride between each detector.
        """

        super(fft3dConvReLUPool, self).__init__()

        check_cuda(str(type(self)))

        detector_channels = num_channels * num_pieces

        self.__dict__.update(locals())
        del self.self

    @functools.wraps(Layer.get_lr_scalers)
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

    @functools.wraps(Layer.set_input_space)
    def set_input_space(self, space):
        """ Note: this resets parameters! """

        # setup_detector_layer_bct01(layer=self,
        #                            input_space=space,
        #                            rng=self.mlp.rng,
        #                            irange=self.irange)
        # Use theano conv3d instead
        setup_detector_layer_b01tc(layer=self,
                                   input_space=space,
                                   rng=self.mlp.rng,
                                   irange=self.irange)
        rng = self.mlp.rng
        detector_shape = self.detector_space.shape

        def handle_pool_shape(idx):
            if self.pool_shape[idx] < 1:
                raise ValueError("bad pool shape: " + str(self.pool_shape))
            if self.pool_shape[idx] > detector_shape[idx]:
                if self.fix_pool_shape:
                    assert detector_shape[idx] > 0
                    self.pool_shape[idx] = detector_shape[idx]
                else:
                    raise ValueError("Pool shape exceeds detector layer shape on axis %d" % idx)
        map(handle_pool_shape, [0, 1, 2])


        ### Check some precondition
        assert self.pool_shape[0] == self.pool_shape[1]
        assert self.pool_stride[0] == self.pool_stride[1]
        assert all(isinstance(elem, py_integer_types) for elem in self.pool_stride)
        for i in xrange(0, 2):
            assert self.pool_stride[i] <= self.pool_shape[i]
        assert all(isinstance(elem, py_integer_types) for elem in self.pool_stride)

        # Find space shape after convolution
        dummy_output_shape = [int(np.ceil((i_sh + 2. * self.pad - k_sh) / float(k_st))) + 1
                              for i_sh, k_sh, k_st in zip(self.input_space.shape,
                                                          self.kernel_shape,
                                                          self.kernel_stride)]
        dummy_output_sequence_length = dummy_output_shape[2]

        ### Find the space shape after spatial pooling
        dummy_output_shape = [dummy_output_shape[0],
                              dummy_output_shape[1]]
        dummy_detector_space = Conv2DSpace(shape=dummy_output_shape,
                                           num_channels = self.detector_channels,
                                           axes = ('c', 0, 1, 'b'))
        dummy_detector = sharedX(dummy_detector_space.get_origin_batch(2)[0:16, :, :, :])
        dummy_p = max_pool_c01b(c01b = dummy_detector,
                                pool_shape = self.pool_shape,
                                pool_stride = self.pool_stride)
        dummy_p = dummy_p.eval()

        ### Find the space shape after temporal pooling
        # (0*1,'t')
        dummy_temp_image = [dummy_p.shape[1] * dummy_p.shape[2] , dummy_output_sequence_length]
        self.temp_pool_input_shape = dummy_temp_image
        dummy_temp_space = Conv2DSpace(shape=dummy_temp_image,
                                       num_channels = self.detector_channels,
                                       axes = ('c', 0, 1, 'b'))
        temp_input = sharedX(dummy_temp_space.get_origin_batch(2)[0:16,:,:,:])
        dummy_temp_p = temporal_max_pool_c01b(c01b=temp_input,
                                              pool_shape = [1, self.pool_shape[2]],
                                              pool_stride = [1, self.pool_stride[2]],
                                              image_shape = dummy_temp_image)
        dummy_temp_p = dummy_temp_p.eval()
        self.output_space = Conv3DSpace(shape=[dummy_p.shape[1],
                                               dummy_p.shape[2],
                                               dummy_temp_p.shape[2]],
                                        num_channels = self.num_channels,
                                        axes = ('b', 0, 1,'t','c'))

        # Print spaces
        print "Input shape: ", self.input_space.shape
        print "Detector space: ", self.detector_space.shape
        print "Output space: ", self.output_space.shape


    @functools.wraps(Layer._modify_updates)
    def _modify_updates(self, updates):
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
                updates[W] = updated_W #* scales.dimshuffle(0, 'x')

    @functools.wraps(Layer.get_params)
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

    @functools.wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    @functools.wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * abs(W).sum()

    @functools.wraps(Layer.set_weights)
    def set_weights(self, weights):

        W, = self.transformer.get_params()
        W.set_value(weights)

    @functools.wraps(Layer.set_biases)
    def set_biases(self, biases):

        self.b.set_value(biases)

    @functools.wraps(Layer.get_biases)
    def get_biases(self):

        return self.b.get_value()

    @functools.wraps(Layer.get_weights_format)
    def get_weights_format(self):

        return ('v', 'h')

    @functools.wraps(Layer.get_weights_topo)
    def get_weights_topo(self):

        outp, inp, rows, cols = range(4)
        raw = self.transformer._filters.get_value()

        return np.transpose(raw, (outp, rows, cols, inp))

    @functools.wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):
        W, = self.transformer.get_params()


        assert W.ndim == 5

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=(1, 2, 3, 4)))

        rval = OrderedDict([('kernel_norms_min',  row_norms.min()),
                            ('kernel_norms_mean', row_norms.mean()),
                            ('kernel_norms_max',  row_norms.max()), ])

        if (state is not None) or (state_below is not None):
            if state is None:
                state = self.fprop(state_below)

            P = state

            vars_and_prefixes = [(P, '')]

            for var, prefix in vars_and_prefixes:
                assert var.ndim == 5
                v_max = var.max(axis=(1, 2, 3, 4))
                v_min = var.min(axis=(1, 2, 3, 4))
                v_mean = var.mean(axis=(1, 2, 3, 4))
                v_range = v_max - v_min

                # max_x.mean_u is "the mean over *u*nits of the max over
                # e*x*amples" The x and u are included in the name because
                # otherwise its hard to remember which axis is which when
                # reading the monitor I use inner.outer rather than
                # outer_of_inner or something like that because I want
                # mean_x.* to appear next to each other in the
                # alphabetical list, as these are commonly plotted
                # together
                for key, val in [('max_x.max_u',    v_max.max()),
                                 ('max_x.mean_u',   v_max.mean()),
                                 ('max_x.min_u',    v_max.min()),
                                 ('min_x.max_u',    v_min.max()),
                                 ('min_x.mean_u',   v_min.mean()),
                                 ('min_x.min_u',    v_min.min()),
                                 ('range_x.max_u',  v_range.max()),
                                 ('range_x.mean_u', v_range.mean()),
                                 ('range_x.min_u',  v_range.min()),
                                 ('mean_x.max_u',   v_mean.max()),
                                 ('mean_x.mean_u',  v_mean.mean()),
                                 ('mean_x.min_u',   v_mean.min())]:
                    rval[prefix+key] = val

        return rval

    @functools.wraps(Layer.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state, target=None):
        return self.get_monitoring_channels_from_state()



    @functools.wraps(Layer.fprop)
    def fprop(self, state_below):

        check_cuda(str(type(self)))
        self.input_space.validate(state_below)
        if not hasattr(self, 'input_normalization'):
            self.input_normalization = None

        if self.input_normalization:
            state_below = self.input_normalization(state_below)

        # fft 3d covolution
        z = self.transformer.lmul(state_below)

        # bias addition
        if not hasattr(self, 'tied_b'):
            self.tied_b = False
        if self.tied_b:
            b = self.b.dimshuffle(0, 'x', 'x', 'x', 'x')
        else:
            b = self.b.dimshuffle('x', 0, 1, 2, 3)
        z = z + self.b



        if self.layer_name is not None:
           z.name = self.layer_name + '_z'
        self.detector_space.validate(z)
        #assert self.detector_space.num_channels % 16 == 0

        #ReLUs
        z = T.maximum(z, 0)
        if self.output_space.num_channels % 16 != 0:
            raise NotImplementedError("num channles should always be dvisible by 16")

        # alex's max pool op only works when the number of channels
        # is divisible by 16. we can only do the cross-channel pooling
        # first if the cross-channel pooling preserves that property

        # Pooling
        # permute axes ['b', 0, 1,'t','c'] -> ['c', 0, 1, 't', 'b'] (axes required for pooling )
        z = z.dimshuffle(4, 1, 2, 3, 0)

        # spatial pooling x/y
        z_shape = z.shape
        z = z.reshape((z_shape[0], z_shape[1], z_shape[2], z_shape[3] * z_shape[4]))
        p = max_pool_c01b(c01b = z,
                          pool_shape = self.pool_shape[0:2],
                          pool_stride = self.pool_stride[0:2])
        p = p.reshape((p.shape[0], p.shape[1], p.shape[2], z_shape[3], z_shape[4]))


        # temporal pooling with overlap (t)
        p_shape = p.shape
        #['c', 0, 1, 't', 'b'] -> ['c',0*1,'t','b'] ('c',0, 1,'b') for max_pool_c01b
        p = p.reshape((p_shape[0], p_shape[1] * p_shape[2], p_shape[3] , p_shape[4]))
        t = temporal_max_pool_c01b(c01b = p,
                                   pool_shape = [1, self.pool_shape[2]],
                                   pool_stride = [1, self.pool_stride[2]],
                                   image_shape = self.temp_pool_input_shape)
        t_shape = t.shape
        t = t.reshape((t_shape[0], p_shape[1] , p_shape[2], t_shape[2] , t_shape[3]))
        # Permute back axes ['c', 0, 1, 't', 'b'] -> ['b', 0, 1, 't', 'c']
        t = t.dimshuffle(4, 1, 2, 3, 0)
        self.output_space.validate(t)

        if not hasattr(self, 'output_normalization'):
            self.output_normalization = None

        if self.output_normalization:
            t = self.output_normalization(t)

        return t

    def upward_pass(self, inputs):
        """
        Wrapper to fprop functions for PretrainedLayer class

        Parameters
        ----------
        inputs : WRITEME

        Returns
        -------
        WRITEME
        """
        return self.fprop(inputs)
