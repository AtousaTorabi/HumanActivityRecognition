"""
Multilayer Perceptron
"""
import math
import sys
import warnings
from functools import wraps


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
from pylearn2.space import VectorSpace , Conv2DSpace
from HumanActivityRecognition.space import Conv3DSpace
from pylearn2.utils import function
from pylearn2.utils import py_integer_types

from pylearn2.utils import sharedX
from theano.sandbox.cuda.basic_ops import gpu_contiguous
# setup detector layer for fft 3d convolution with axes bct01
from HumanActivityRecognition.linear.conv3d_btc01new3 import setup_detector_layer_btc01
#from HumanActivityRecognition.linear.conv3d_b01tc import setup_detector_layer_b01tc
#from HumanActivityRecognition.linear.conv3d_btc01 import setup_detector_layer_b01tc
if cuda.cuda_available:
    from pylearn2.sandbox.cuda_convnet.pool import max_pool_c01b
from theano.sandbox.cuda.dnn import dnn_pool

# used for temporal pooling with overlap
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

from theano.gof import Op, Apply
class Print(Op):
    """ This identity-like Op print as a side effect.

    This identity-like Op has the side effect of printing a message
    followed by its inputs when it runs. Default behaviour is to print
    the __str__ representation. Optionally, one can pass a list of the
    input member functions to execute, or attributes to print.

    @type message: String
    @param message: string to prepend to the output
    @type attrs: list of Strings
    @param attrs: list of input node attributes or member functions to print.
                  Functions are identified through callable(), executed and
                  their return value printed.

    :note: WARNING. This can disable some optimizations!
                    (speed and/or stabilization)

            Detailed explanation:
            As of 2012-06-21 the Print op is not known by any optimization.
            Setting a Print op in the middle of a pattern that is usually
            optimized out will block the optimization. for example, log(1+x)
            optimizes to log1p(x) but log(1+Print(x)) is unaffected by
            optimizations.

    """
    view_map = {0: [0]}

    def __init__(self, message="", attrs=("__str__",)):
        self.message = message
        self.attrs = tuple(attrs)  # attrs should be a hashable iterable

    def make_node(self, xin):
        xout = xin.type.make_variable()
        return Apply(op=self, inputs=[xin], outputs=[xout])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin
        print self.message, xin.shape

    def grad(self, input, output_gradients):
        return output_gradients

    def R_op(self, inputs, eval_points):
        return [x for x in eval_points]

    def __eq__(self, other):
        return (type(self) == type(other) and self.message == other.message
                and self.attrs == other.attrs)

    def __hash__(self):
        return hash(self.message) ^ hash(self.attrs)

    def __setstate__(self, dct):
        dct.setdefault('global_fn', _print_fn)
        self.__dict__.update(dct)

    def c_code_cache_version(self):
        return (1,)

class fft3dConvReLUPool(Layer):

    def __init__(self,
                 num_channels,
                 num_pieces,
                 kernel_shape,
                 kernel_stride,
                 pool_shape,
                 pool_stride,
                 layer_name,
                 irange = None,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 pad = [0, 0, 0],
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
                 ):
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
        check_cuda(str(type(self)))

        detector_channels = num_channels * num_pieces

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
        """ Note: this resets parameters! """

        # set up detector space and initialize transformer
        setup_detector_layer_btc01(layer=self,
                                   input_space=space,
                                   rng=self.mlp.rng,
                                   irange=self.irange)
        rng = self.mlp.rng
        detector_shape = self.detector_space.shape

        print "detector shape:", self.detector_space.shape

        #def handle_pool_shape(idx):
        #    if self.pool_shape[idx] < 1:
        #        raise ValueError("bad pool shape: " + str(self.pool_shape))
        #    if self.pool_shape[idx] > detector_shape[idx]:
        #        if self.fix_pool_shape:
        #            assert detector_shape[idx] > 0
        #            self.pool_shape[idx] = detector_shape[idx]
        #        else:
        #            raise ValueError("Pool shape exceeds detector layer shape on axis %d" % idx)
        #map(handle_pool_shape, [0, 1, 2])


        ### Check some precondition
        assert all(isinstance(elem, py_integer_types) for elem in self.pool_stride)
        for i in xrange(0, 3):
            assert self.pool_stride[i] <= self.pool_shape[i]
        assert all(isinstance(elem, py_integer_types) for elem in self.pool_stride)

        dummy_shape = [self.input_space.shape[0] , self.input_space.shape[1], self.input_space.shape[2]]
        dummy_output_shape = [int((i_sh + 2. * p_sh - k_sh) / float(k_st)) +1
                              for i_sh, p_sh, k_sh, k_st in zip(dummy_shape,
                                                                self.pad,
                                                                self.kernel_shape,
                                                                self.kernel_stride)]

        dummy_output_shape = [dummy_output_shape[0],
                              dummy_output_shape[1]]
        #print dummy_output_shape
        dummy_detector_space = Conv2DSpace(shape=dummy_output_shape,
                                           num_channels = self.detector_channels,
                                           axes = ('b', 'c', 0, 1))
        dummy_detector = sharedX(dummy_detector_space.get_origin_batch(2)[:, :, :,:])
        dummy_p = dnn_pool(img=dummy_detector,
                           ws=tuple(self.pool_shape[:2]),
                           stride=tuple(self.pool_stride[:2]))

        print "bshape", dummy_detector_space.get_origin_batch(2)[:, :, :,:].shape, self.pool_shape[:2], self.pool_stride[:2]
        dummy_p = dummy_p.eval()

        print "ashape", dummy_p.shape, [1, self.pool_shape[2]], [1, self.pool_stride[2]]
        dummy_temp_image = [(dummy_p.shape[2]*dummy_p.shape[3]) , self.detector_space.shape[2]]
        self.temp_pool_input_shape = dummy_temp_image
        dummy_temp_space = Conv2DSpace(shape=dummy_temp_image,
                                       num_channels = self.detector_channels,
                                       axes = ('b', 'c', 0, 1))
        temp_input = sharedX(dummy_temp_space.get_origin_batch(2)[:,:,:,:])
        dummy_temp_p = dnn_pool(img=temp_input,
                                ws=tuple([1, self.pool_shape[2]]),
                                stride=tuple([1, self.pool_stride[2]]))
        dummy_temp_p = dummy_temp_p.eval()
        print "ashape2:", dummy_temp_p.shape



        self.output_space = Conv3DSpace(shape=[dummy_p.shape[2],
                                               dummy_p.shape[3],
                                               dummy_temp_p.shape[3]],
                                        num_channels = self.num_channels,
                                        axes = ('b','c', 0, 1,'t'))


        # Print spaces
        print "Input shape: ", self.input_space.shape
        print "Detector space: ", self.detector_space.shape
        print "Output space: ", self.output_space.shape


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
                updates[W] = updated_W #* scales.dimshuffle(0, 'x')
    def _modify_updates(self, updates):
        """
        Replaces the values in `updates` if needed to enforce the options set
        in the __init__ method, including `mask_weights` and `max_col_norm`.

        Parameters
        ----------
        updates : OrderedDict
            A dictionary mapping parameters (including parameters not
            belonging to this model) to updated values of those parameters.
            The dictionary passed in contains the updates proposed by the
            learning algorithm. This function modifies the dictionary
            directly. The modified version will be compiled and executed
            by the learning algorithm.
        """
        if self.max_kernel_norm is not None:
            W, = self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=(1, 2, 3, 4)))
                desired_norms = T.clip(row_norms, 0, self.max_kernel_norm)
                scales = desired_norms / (1e-7 + row_norms)
                updates[W] = (updated_W * scales.dimshuffle(0, 'x', 'x', 'x', 'x'))

    @wraps(Layer.get_params)
    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        assert self.b.name is not None
        W, = self.transformer.get_params()
        #print W.name
        #W.name = "W"
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

    @wraps(Layer.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state, target=None):
        return self.get_monitoring_channels_from_state()


    @wraps(Layer.fprop)
    def fprop(self, state_below):

        check_cuda(str(type(self)))
        self.input_space.validate(state_below)
        if not hasattr(self, 'input_normalization'):
            self.input_normalization = None

        #state_below= Print("state_below")(state_below)

        if self.input_normalization:
            state_below = self.input_normalization(state_below)
        #import pdb; pdb.set_trace()
        # GPU 3d correlation
        z = self.transformer.lmul(state_below)

        # bias addition
        if not hasattr(self, 'tied_b'):
            self.tied_b = False
        if self.tied_b:
            b = self.b.dimshuffle('x', 0, 'x', 'x', 'x')
        else:
            b = self.b.dimshuffle('x', 0, 1, 2, 3)

        #z = Print('z')(z)
        #b = Print('b')(b)
        z = z + b



        if self.layer_name is not None:
           z.name = self.layer_name + '_z'
        self.detector_space.validate(z)
        #assert self.detector_space.num_channels % 16 == 0

        #ReLUs
        #z= Print("z")(z)
        z = T.maximum(z, 0)

        # Pooling
        if tuple(self.pool_shape) != (1, 1, 1):
            # Pooling on y, t
            z_shape = z.shape
            z = z.reshape((z_shape[0], z_shape[1] * z_shape[2],
                           z_shape[3], z_shape[4]))

            p = dnn_pool(img=z,
                         ws=tuple(self.pool_shape[1:]),
                         stride=tuple(self.pool_stride[1:]))

            p_shape = p.shape
            p = p.reshape((p_shape[0], z_shape[1],
                           z_shape[2], p_shape[2],
                           p_shape[3]))

            #p = Print("p")(p)

            # Pooling on x
            p_shape =p.shape
            p = p.reshape((p_shape[0], p_shape[1],
                           p_shape[2], p_shape[3] * p_shape[4]))

            t = dnn_pool(img=p,
                         ws=tuple([self.pool_shape[0], 1]),
                         stride=tuple([self.pool_stride[0], 1]))
            t_shape = t.shape
            #print (t_shape[0], t_shape[1], t_shape[2], p_shape[2] , p_shape[3])
            t = t.reshape((t_shape[0], t_shape[1],
                           t_shape[2], p_shape[3] , p_shape[4]))
        else:
            t = z

        self.output_space.validate(t)
        ## Gpu contiguous
        #t = gpu_contiguous(t)
        #t = Print("t")(t)

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
