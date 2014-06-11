"""
Multilayer Perceptron

Note to developers and code reviewers: when making any changes to this
module, ensure that the changes do not break pylearn2/scripts/papers/maxout.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow", "David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import logging
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
from pylearn2.models.mlp import max_pool, mean_pool
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

logger = logging.getLogger(__name__)


class BranchMLP(Layer):
    """
    A multilayer perceptron.
    Note that it's possible for an entire MLP to be a single
    layer of a larger MLP.
    """

    def __init__(self,
            layers,
            batch_size=None,
            input_space=None,
            nvis=None,
            seed=None,
            dropout_include_probs = None,
            dropout_scales = None,
            dropout_input_include_prob = None,
            dropout_input_scale = None,
            ):
        """
            layers: a list of MLP_Layers. The final layer will specify the
                    MLP's output space.
            batch_size: optional. if not None, then should be a positive
                        integer. Mostly useful if one of your layers
                        involves a theano op like convolution that requires
                        a hard-coded batch size.
            input_space: a Space specifying the kind of input the MLP acts
                        on. If None, input space is specified by nvis.
            dropout*: None of these arguments are supported anymore. Use
                      pylearn2.costs.mlp.dropout.Dropout instead.
        """

        locals_snapshot = locals()

        for arg in locals_snapshot:
            if arg.find('dropout') != -1 and locals_snapshot[arg] is not None:
                raise TypeError(arg+ " is no longer supported. Train using "
                        "an instance of pylearn2.costs.mlp.dropout.Dropout "
                        "instead of hardcoding the dropout into the model"
                        " itself. All dropout related arguments and this"
                        " support message may be removed on or after "
                        "October 2, 2013. They should be removed from the "
                        "SoftmaxRegression subclass at the same time.")

        if seed is None:
            seed = [2013, 1, 4]

        self.seed = seed
        self.setup_rng()

        assert isinstance(layers, list)
        assert all(isinstance(layer, Layer) for layer in layers[:-1])
        assert isinstance(layers[-1], list)
        assert len(layers) >= 1
        self.layer_names = set()
        for layer in layers:
            if isinstance(layer, list):
                for item in layer:
                    assert item.get_mlp() is None
                    assert item.layer_name not in self.layer_names
                    item.set_mlp(self)
                    self.layer_names.add(item.layer_name)
            else:
                assert layer.get_mlp() is None
                assert layer.layer_name not in self.layer_names
                layer.set_mlp(self)
                self.layer_names.add(layer.layer_name)

        self.layers = layers

        self.batch_size = batch_size
        self.force_batch_size = batch_size

        assert input_space is not None or nvis is not None
        if nvis is not None:
            input_space = VectorSpace(nvis)

        self.input_space = input_space

        self._update_layer_input_spaces()

        self.freeze_set = set([])

        def f(x):
            if x is None:
                return None
            return 1. / x

    def setup_rng(self):
        self.rng = np.random.RandomState(self.seed)

    def get_default_cost(self):
        return Default()

    def get_output_space(self):
        rval = []
        for item in self.layers[-1]:
            rval.append(item.get_output_space())
        return rval

    def _update_layer_input_spaces(self):
        """
            Tells each layer what its input space should be.
            Note: this usually resets the layer's parameters!
        """
        layers = self.layers
        layers[0].set_input_space(self.input_space)
        for i in xrange(1,len(layers)-1):
            layers[i].set_input_space(layers[i-1].get_output_space())

        maxout2_outputspace = layers[-2].get_output_space()
        for layer in layers[-1]:
            layer.set_input_space(maxout2_outputspace)


    def add_layers(self, layers):
        """
            Add new layers on top of the existing hidden layers
        """

        existing_layers = self.layers
        assert len(existing_layers) > 0
        for layer in layers:
            assert layer.get_mlp() is None
            layer.set_mlp(self)
            layer.set_input_space(existing_layers[-1].get_output_space())
            existing_layers.append(layer)
            assert layer.layer_name not in self.layer_names
            self.layer_names.add(layer.layer_name)

    def freeze(self, parameter_set):

        self.freeze_set = self.freeze_set.union(parameter_set)

    def get_monitoring_channels(self, data):
        """
        data is a flat tuple, and can contain features, targets, or both
        """
        X, Y, Y_ = data
        state = X
        rval = OrderedDict()

        for layer in self.layers[:-1]:
            ch = layer.get_monitoring_channels()
            for key in ch:
                rval[layer.layer_name+'_'+key] = ch[key]
            state = layer.fprop(state)
            args = [state]
            if layer is self.layers[-1]:
                args.append(Y)
            ch = layer.get_monitoring_channels_from_state(*args)
            if not isinstance(ch, OrderedDict):
                raise TypeError(str((type(ch), layer.layer_name)))
            for key in ch:
                rval[layer.layer_name+'_'+key]  = ch[key]

        if isinstance(state, tuple):
            states = state
        else:
            states = [state, state]
        """        
        import pdb
        pdb.set_trace()
        states = [state,state]
        """
        for layer, state in zip(self.layers[-1], states):
            if isinstance(layer, NestedMLP):
                if layer is self.layers[-1][0]:
                    ch = layer.get_monitoring_channels((state, Y))
                else:
                    ch = layer.get_monitoring_channels((state, Y_))
                for key in ch:
                    rval[layer.layer_name+'_'+key] = ch[key]
            else:
                ch = layer.get_monitoring_channels()
                for key in ch:
                    rval[layer.layer_name+'_'+key] = ch[key]
                state = layer.fprop(state)
                args = [state]
                if layer is self.layers[-1][0]:
                    args.append(Y)
                else:
                    args.append(Y_)
                ch = layer.get_monitoring_channels_from_state(*args)
                if not isinstance(ch, OrderedDict):
                    raise TypeError(str((type(ch), layer.layer_name)))
                for key in ch:
                    rval[layer.layer_name+'_'+key]  = ch[key]

        return rval

    def get_monitoring_data_specs(self):
        """
        Return the (space, source) data_specs for self.get_monitoring_channels.

        In this case, we want the inputs and targets.
        """
        space = [self.get_input_space()]
        space += self.get_output_space()
        space = CompositeSpace(space)
        source = (self.get_input_source(), self.get_target_source(), 'second_targets')
        return (space, source)


    def get_params(self):

        rval = []
        for layer in self.layers[:-1]:
            for param in layer.get_params():
                if param.name is None:
                    print type(layer)
            layer_params = layer.get_params()
            assert not isinstance(layer_params, set)
            for param in layer_params:
                if param not in rval:
                    rval.append(param)
        for layer in self.layers[-1]:
            for param in layer.get_params():
                if param.name is None:
                    print type(layer)
            layer_params = layer.get_params()
            assert not isinstance(layer_params, set)
            for param in layer_params:
                if param not in rval:
                    rval.append(param)


        rval = [elem for elem in rval if elem not in self.freeze_set]

        assert all([elem.name is not None for elem in rval])

        return rval

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.force_batch_size = batch_size

        for layer in self.layers:
            layer.set_batch_size(batch_size)


    def censor_updates(self, updates):
        for layer in self.layers[:-1]:
            layer.censor_updates(updates)
        for layer in self.layers[-1]:
            layer.censor_updates(updates)

    def get_lr_scalers(self):
        rval = OrderedDict()

        params = self.get_params()

        for layer in self.layers[:-1]:
            contrib = layer.get_lr_scalers()

            assert isinstance(contrib, OrderedDict)
            # No two layers can contend to scale a parameter
            assert not any([key in rval for key in contrib])
            # Don't try to scale anything that's not a parameter
            assert all([key in params for key in contrib])

            rval.update(contrib)

        for layer in self.layers[-1]:
            contrib = layer.get_lr_scalers()

            assert isinstance(contrib, OrderedDict)
            # No two layers can contend to scale a parameter
            assert not any([key in rval for key in contrib])
            # Don't try to scale anything that's not a parameter
            assert all([key in params for key in contrib])

            rval.update(contrib)

        assert all([isinstance(val, float) for val in rval.values()])

        return rval

    def get_weights(self):
        return self.layers[0].get_weights()

    def get_weights_view_shape(self):
        return self.layers[0].get_weights_view_shape()

    def get_weights_format(self):
        return self.layers[0].get_weights_format()

    def get_weights_topo(self):
        return self.layers[0].get_weights_topo()

    def dropout_fprop(self, state_below, default_input_include_prob=0.5,
                      input_include_probs=None, default_input_scale=2.,
                      input_scales=None, per_example=True):
        """
        state_below: The input to the MLP

        Returns the output of the MLP, when applying dropout to the input and intermediate layers.
        Each input to each layer is randomly included or excluded
        for each example. The probability of inclusion is independent for each input
        and each example. Each layer uses "default_input_include_prob" unless that
        layer's name appears as a key in input_include_probs, in which case the input
        inclusion probability is given by the corresponding value.

        Each feature is also multiplied by a scale factor. The scale factor for each
        layer's input scale is determined by the same scheme as the input probabilities.

        per_example : bool, optional
            Sample a different mask value for every example in
            a batch. Default is `True`. If `False`, sample one
            mask per mini-batch.
        """

        warnings.warn("dropout doesn't use fixed_var_descr so it won't work with "
                "algorithms that make more than one theano function call per batch,"
                " such as BGD. Implementing fixed_var descr could increase the memory"
                " usage though.")

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self._validate_layer_names(list(input_include_probs.keys()))
        self._validate_layer_names(list(input_scales.keys()))

        theano_rng = MRG_RandomStreams(self.rng.randint(2 ** 15))

        for layer in self.layers[:-1]:
            layer_name = layer.layer_name

            if layer_name in input_include_probs:
                include_prob = input_include_probs[layer_name]
            else:
                include_prob = default_input_include_prob

            if layer_name in input_scales:
                scale = input_scales[layer_name]
            else:
                scale = default_input_scale

            state_below = self.apply_dropout(
                state=state_below,
                include_prob=include_prob,
                theano_rng=theano_rng,
                scale=scale,
                mask_value=layer.dropout_input_mask_value,
                input_space=layer.get_input_space(),
                per_example=per_example
            )
            state_below = layer.fprop(state_below)


        """ 
        import pdb
        pdb.set_trace()
        rvals = []
        state = state_below
        self.inputToBranchDropout = state_below
        for layer in self.layers[-1]:
        """
        
        if isinstance(state_below,tuple):
            states = state_below
        else:
            states = (state_below, state_below)
        self.inputToBranchDropout = state_below
        
        rvals = []
        for (state, layer) in zip(states, self.layers[-1]):       
            layer_name = layer.layer_name

            if layer_name in input_include_probs:
                include_prob = input_include_probs[layer_name]
            else:
                include_prob = default_input_include_prob

            if layer_name in input_scales:
                scale = input_scales[layer_name]
            else:
                scale = default_input_scale

            state_below = self.apply_dropout(
                state=state,
                include_prob=include_prob,
                theano_rng=theano_rng,
                scale=scale,
                mask_value=layer.dropout_input_mask_value,
                input_space=layer.get_input_space(),
                per_example=per_example
            )
            rvals.append(layer.fprop(state_below))

        return rvals

    def _validate_layer_names(self, layers):
        if any(layer not in self.layer_names for layer in layers):
            unknown_names = [layer for layer in layers
                                if layer not in self.layer_names]
            raise ValueError("MLP has no layer(s) named %s" %
                                ", ".join(unknown_names))

    def get_total_input_dimension(self, layers):
        """
        Get the total number of inputs to the layers whose

        names are listed in `layers`. Used for computing the
        total number of dropout masks.
        """
        self._validate_layer_names(layers)
        total = 0
        for layer in self.layers:
            if layer.layer_name in layers:
                total += layer.get_input_space().get_total_dimension()
        return total

    def fprop(self, state_below, return_all = False):
        rval_list = []

        rval = self.layers[0].fprop(state_below)
        rval_list.append(rval)


        for layer in self.layers[1:-1]:
            rval = layer.fprop(rval)
            rval_list.append(rval)
                    
        if isinstance(rval,tuple) or isinstance(rval,list):
            rvals = []
            for val, item in zip(rval, self.layers[-1]):
                rvals.append(item.fprop(val, True))
            rval_list.append(rvals)
        else:
            rvals = []
            for item in self.layers[-1]:
                rvals.append(item.fprop(rval, True))
            rval_list.append(rvals)
        
        if return_all:
            return rval_list
        else:
            return rvals

    def apply_dropout(self, state, include_prob, scale, theano_rng,
                      input_space, mask_value=0, per_example=True):

        """
        Parameters
        ----------
        ...

        per_example : bool, optional
            Sample a different mask value for every example in
            a batch. Default is `True`. If `False`, sample one
            mask per mini-batch.
        """
        if include_prob in [None, 1.0, 1]:
            return state
        assert scale is not None
        if isinstance(state, tuple):
            return tuple(self.apply_dropout(substate, include_prob,
                                            scale, theano_rng, mask_value)
                         for substate in state)
        # TODO: all of this assumes that if it's not a tuple, it's
        # a dense tensor. It hasn't been tested with sparse types.
        # A method to format the mask (or any other values) as
        # the given symbolic type should be added to the Spaces
        # interface.
        if per_example:
            mask = theano_rng.binomial(p=include_prob, size=state.shape,
                                       dtype=state.dtype)
        else:
            batch = input_space.get_origin_batch(1)
            mask = theano_rng.binomial(p=include_prob, size=batch.shape,
                                       dtype=state.dtype)
            rebroadcast = T.Rebroadcast(*zip(xrange(batch.ndim),
                                             [s == 1 for s in batch.shape]))
            mask = rebroadcast(mask)
        if mask_value == 0:
            return state * mask * scale
        else:
            return T.switch(mask, state * scale, mask_value)

    def cost(self, Ys, Y_hats):
        assert isinstance(Y_hats, list)
        assert isinstance(Ys, list)
        assert len(Ys) == len(Y_hats)
        assert len(self.layers[-1]) == len(Ys)

        total_cost = 0
        for i in xrange(len(Ys)):
            total_cost += self.layers[-1][i].cost(Ys[i], Y_hats[i])
        return total_cost

    def cost_matrix(self, Y, Y_hat):
        return self.layers[-1].cost_matrix(Y, Y_hat)

    def cost_from_cost_matrix(self, cost_matrix, cost_matrix_):
        return self.layers[-1][0].cost_from_cost_matrix(cost_matrix) + self.layers[-1][1].cost_from_cost_matrix(cost_matrix_)

    def cost_from_X(self, data):
        """
        Computes self.cost, but takes data=(X, Y) rather than Y_hat as an argument.
        This is just a wrapper around self.cost that computes Y_hat by
        calling Y_hat = self.fprop(X)
        """
        self.cost_from_X_data_specs()[0].validate(data)
        X, Y = data
        Y_hats= self.fprop(X)
        return self.cost(Y, Y_hats)

    def cost_from_X_data_specs(self):
        """
        Returns the data specs needed by cost_from_X.

        This is useful if cost_from_X is used in a MethodCost.
        """
        space = [self.get_input_space()]
        space.append(selg.get_output_space())
        space = CompositeSpace(sapce)
        source = (self.get_input_source(), self.get_target_source(), 'second_targets')
        return (space , source)


class NestedMLP(MLP):
    def __init__(self,
            layers,
            layer_name,
            batch_size=None,
            input_space=None,
            nvis=None,
            seed=None,
            dropout_include_probs = None,
            dropout_scales = None,
            dropout_input_include_prob = None,
            dropout_input_scale = None,
            ):

        super(NestedMLP, self).__init__(layers = layers,
            batch_size= batch_size,
            input_space= input_space,
            nvis=nvis,
            seed=seed,
            dropout_include_probs = dropout_include_probs,
            dropout_scales = dropout_scales,
            dropout_input_include_prob = dropout_input_include_prob,
            dropout_input_scale = dropout_input_scale)

        self.layer_name = layer_name


    def set_input_space(self, space):
        pass

 # This sigmoid support tagging option which the original pylearn2 repo doesn't
class Sigmoid(Linear):
    """
    Implementation of the sigmoid nonlinearity for MLP.
    """
	
    def __init__(self,
                 dim,
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
        ave = total.mean(axis=1)
        assert ave.ndim == 1

        return ave


    def cost(self, Y, Y_hat):
        """
        mean across units, mean across batch of KL divergence
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

        total = self.kl(Y=Y, Y_hat=Y_hat)

        ave = total.mean()

        return ave
        
    def cost_matrix(self, Y, Y_hat):
        """
        mean across units, mean across batch of KL divergence
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
        return total

    def cost_from_cost_matrix(self, cost_matrix):
        return cost_matrix.mean()

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
        #APs = meanAvgPrec.everyAveragePrecisionTheano(target, state)
        #rval['min_avg_prec'] = APs.min()
        #rval['mean_avg_prec'] = APs.mean()
        #rval['max_avg_prec'] = APs.max()
        #rval['avg_prec_AnswerPhone'] = APs[0]
        #rval['avg_prec_DriveCar'] = APs[1]
        #rval['avg_prec_Eat'] = APs[2]
        #rval['avg_prec_FightPerson'] = APs[3]
        #rval['avg_prec_GetOutCar'] = APs[4]
        #rval['avg_prec_HandShake'] = APs[5]
        #rval['avg_prec_HugPerson'] = APs[6]
        #rval['avg_prec_Kiss'] = APs[7]
        #rval['avg_prec_Run'] = APs[8]
        #rval['avg_prec_SitDown'] = APs[9]
        #rval['avg_prec_SitUp'] = APs[10]
        #rval['avg_prec_StandUp'] = APs[11]

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

class Conv1DReLU(Layer):
    """
    Parameters
    ----------
    output_channels : int
        The number of output channels the layer should have.
    kernel_shape : tuple
        The shape of the convolution kernel.
    pool_shape : tuple
        The shape of the spatial max pooling. A two-tuple of ints.
    pool_stride : tuple
        The stride of the spatial max pooling. Also must be square.
    layer_name : str
        A name for this layer that will be prepended to monitoring channels
        related to this layer.
    irange : float, optional
        if specified, initializes each weight randomly in
        U(-irange, irange)
    border_mode : str, optional
        A string indicating the size of the output:
        - "full" : The output is the full discrete linear convolution of the
        inputs.
        - "valid" : The output consists only of those elements that do not
        rely on the zero-padding. (Default)
    sparse_init : WRITEME
    include_prob : float, optional
        probability of including a weight element in the set of weights
        initialized to U(-irange, irange). If not included it is initialized
        to 1.0.
    init_bias : float, optional
        All biases are initialized to this number. Default is 0.
    W_lr_scale : float or None
        The learning rate on the weights for this layer is multiplied by this
        scaling factor
    b_lr_scale : float or None
        The learning rate on the biases for this layer is multiplied by this
        scaling factor
    max_kernel_norm : float or None
        If specified, each kernel is constrained to have at most this norm.
    pool_type : str or None
        The type of the pooling operation performed the convolution.
        Default pooling type is max-pooling.
    tied_b : bool, optional
        If true, all biases in the same channel are constrained to be the
        same as each other. Otherwise, each bias at each location is
        learned independently. Default is true.
    detector_normalization : callable or None
        See `output_normalization`.
        If pooling argument is not provided, detector_normalization
        is not applied on the layer.
    output_normalization : callable or None
        if specified, should be a callable object. the state of the
        network is optionally replaced with normalization(state) at each
        of the 3 points in processing:

        - detector: the maxout units can be normalized prior to the
        spatial pooling
        - output: the output of the layer, after sptial pooling, can
        be normalized as well
        kernel_stride : 2-tuple of ints, optional
        The stride of the convolution kernel. Default is (1, 1).
    """
    def __init__(self,
                 output_channels,
                 kernel_shape,
                 layer_name,
                 irange=None,
                 border_mode='valid',
                 sparse_init=None,
                 include_prob=1.0,
                 init_bias=0.,
                 W_lr_scale=None,
                 b_lr_scale=None,
                 max_kernel_norm=None,
                 pool_type=None,
                 pool_shape=None,
                 pool_stride=None,
                 tied_b=None,
                 detector_normalization=None,
                 output_normalization=None,
                 kernel_stride=(1, 1),
                 monitor_style="classification"):

        if (irange is None) and (sparse_init is None):
            raise AssertionError("You should specify either irange or "
                                 "sparse_init when calling the constructor of "
                                 "ConvElemwise.")
        elif (irange is not None) and (sparse_init is not None):
            raise AssertionError("You should specify either irange or "
                                 "sparse_init when calling the constructor of "
                                 "ConvElemwise and not both.")

        if pool_type is not None:
            assert pool_shape is not None, ("You should specify the shape of "
                                           "the spatial %s-pooling." % pool_type)
            assert pool_stride is not None, ("You should specify the strides of "
                                            "the spatial %s-pooling." % pool_type)

        self.__dict__.update(locals())
        assert monitor_style in ['classification',
                            'detection'], ("%s.monitor_style"
                            "should be either detection or classification"
                            % self.__class__.__name__)
        del self.self

    def initialize_transformer(self, rng):
        """
        This function initializes the transformer of the class. Re-running
        this function will reset the transformer.

        Parameters
        ----------
        rng : object
        random number generator object.
        """
        if self.irange is not None:
            assert self.sparse_init is None
            self.transformer = conv2d.make_random_conv2D(
                    irange=self.irange,
                    input_space=self.input_space,
                    output_space=self.detector_space,
                    kernel_shape=self.kernel_shape,
                    subsample=self.kernel_stride,
                    border_mode=self.border_mode,
                    rng=rng)
        elif self.sparse_init is not None:
            self.transformer = conv2d.make_sparse_random_conv2D(
                    num_nonzero=self.sparse_init,
                    input_space=self.input_space,
                    output_space=self.detector_space,
                    kernel_shape=self.kernel_shape,
                    subsample=self.kernel_stride,
                    border_mode=self.border_mode,
                    rng=rng)

    def initialize_output_space(self):
        
        """
        Initializes the output space of the ConvElemwise layer by taking
        pooling operator and the hyperparameters of the convolutional layer
        into consideration as well.
        """
        dummy_batch_size = self.mlp.batch_size

        if dummy_batch_size is None:
            dummy_batch_size = 2
        dummy_detector =\
                sharedX(self.detector_space.get_origin_batch(dummy_batch_size))

        if self.pool_type is not None:
            assert self.pool_type in ['max', 'mean']
            if self.pool_type == 'max':
                dummy_p = max_pool(bc01=dummy_detector,
                                   pool_shape=self.pool_shape,
                                   pool_stride=self.pool_stride,
                                   image_shape=self.detector_space.shape)
            elif self.pool_type == 'mean':
                dummy_p = mean_pool(bc01=dummy_detector,
                                    pool_shape=self.pool_shape,
                                    pool_stride=self.pool_stride,
                                    image_shape=self.detector_space.shape)
            dummy_p = dummy_p.eval()
            self.output_space = Conv2DSpace(shape=[dummy_p.shape[2],
                                                   dummy_p.shape[3]],
                                            num_channels=
                                                self.output_channels,
                                            axes=('b', 'c', 0, 1))
        else:
            dummy_detector = dummy_detector.eval()
            self.output_space = Conv2DSpace(shape=[dummy_detector.shape[2],
                                            dummy_detector.shape[3]],
                                            num_channels=self.output_channels,
                                            axes=('b', 'c', 0, 1))
                                            
        print ('Output space: {0}'.format(self.output_space.shape))

    def set_input_space(self, space):
        """ Note: this function will reset the parameters! """

        self.input_space = space

        if not isinstance(space, Conv2DSpace):
            raise BadInputSpaceError(self.__class__.__name__ +
                                     ".set_input_space "
                                     "expected a Conv2DSpace, got " +
                                     str(space) + " of type " +
                                     str(type(space)))

        rng = self.mlp.rng

        if self.border_mode == 'valid':
            output_shape = [(self.input_space.shape[0] - self.kernel_shape[0])
                            / self.kernel_stride[0] + 1,
                            (self.input_space.shape[1] - self.kernel_shape[1])
                            / self.kernel_stride[1] + 1]
        elif self.border_mode == 'full':
            output_shape = [(self.input_space.shape[0] + self.kernel_shape[0])
                            / self.kernel_stride[0] - 1,
                            (self.input_space.shape[1] + self.kernel_shape[1])
                            / self.kernel_stride_stride[1] - 1]

        self.detector_space = Conv2DSpace(shape=output_shape,
                                          num_channels=self.output_channels,
                                          axes=('b', 'c', 0, 1))

        
        self.initialize_transformer(rng)
        W, = self.transformer.get_params()
        W.name = self.layer_name + '_W'

        if self.tied_b:
            self.b = sharedX(np.zeros((self.detector_space.num_channels)) +
                             self.init_bias)
        else:
            self.b = sharedX(self.detector_space.get_origin() + self.init_bias)

        self.b.name = self.layer_name + '_b'

        print('Input shape: {0}'.format(self.input_space.shape))
        print('Detector space: {0}'.format(self.detector_space.shape))
        
        self.initialize_output_space()


    def _modify_updates(self, updates):
        if self.max_kernel_norm is not None:
            W, = self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=(1, 2, 3)))
                desired_norms = T.clip(row_norms, 0, self.max_kernel_norm)
                updates[W] = updated_W * (desired_norms /
                        (1e-7 + row_norms)).dimshuffle(0, 'x', 'x', 'x')

    def get_params(self):
        assert self.b.name is not None
        W, = self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    def get_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    def get_l1_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * abs(W).sum()

    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

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

    def get_weights_topo(self):
        outp, inp, rows, cols = range(4)
        raw = self.transformer._filters.get_value()
        return np.transpose(raw, (outp, rows, cols, inp))

    def get_monitoring_channels(self):
        return self.get_layer_monitoring_channels()

    def get_monitoring_channels_from_state(self, state, target=None):
        
        mx = state.max(axis=0)
        mean = state.mean(axis=0)
        mn = state.min(axis=0)
        rg = mx - mn

        rval = OrderedDict()
        rval['range_x_max_u'] = rg.max()
        rval['range_x_mean_u'] = rg.mean()
        rval['range_x_min_u'] = rg.min()

        rval['max_x_max_u'] = mx.max()
        rval['max_x_mean_u'] = mx.mean()
        rval['max_x_min_u'] = mx.min()

        rval['mean_x_max_u'] = mean.max()
        rval['mean_x_mean_u'] = mean.mean()
        rval['mean_x_min_u'] = mean.min()

        rval['min_x_max_u'] = mn.max()
        rval['min_x_mean_u'] = mn.mean()
        rval['min_x_min_u'] = mn.min()
        
        return rval

    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):
        return OrderedDict()

        W, = self.transformer.get_params()
        assert W.ndim == 4

        sq_W = T.sqr(W)
        row_norms = T.sqrt(sq_W.sum(axis=(1, 2, 3)))
        return OrderedDict([
                           ('kernel_norms_min', row_norms.min()),
                           ('kernel_norms_mean', row_norms.mean()),
                           ('kernel_norms_max', row_norms.max()),
                           ])

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        z = self.transformer.lmul(state_below)
        if not hasattr(self, 'tied_b'):
            self.tied_b = False

        if self.tied_b:
            b = self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            b = self.b.dimshuffle('x', 0, 1, 2)

        z = z + b
        
        d = z * (z > 0)

        if self.layer_name is not None:
            d.name = self.layer_name + '_z'
            self.detector_space.validate(d)

        if self.pool_type is not None:
            if not hasattr(self, 'detector_normalization'):
                self.detector_normalization = None

            if self.detector_normalization:
                d = self.detector_normalization(d)

            assert self.pool_type in ['max', 'mean'], ("pool_type should be"
                                                      "either max or mean"
                                                      "pooling.")

            if self.pool_type == 'max':
                p = max_pool(bc01=d, pool_shape=self.pool_shape,
                        pool_stride=self.pool_stride,
                        image_shape=self.detector_space.shape)
            elif self.pool_type == 'mean':
                p = mean_pool(bc01=d, pool_shape=self.pool_shape,
                        pool_stride=self.pool_stride,
                        image_shape=self.detector_space.shape)

            self.output_space.validate(p)
        else:
            p = d

        if not hasattr(self, 'output_normalization'):
           self.output_normalization = None

        if self.output_normalization:
           p = self.output_normalization(p)

        return p

    def cost(self, Y, Y_hat):
        import pdb
        pdb.set_trace()

