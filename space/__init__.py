"""
Classes that define how vector spaces are formatted

Most of our models can be viewed as linearly transforming
one vector space to another. These classes define how the
vector spaces should be represented as theano/numpy
variables.

For example, the VectorSpace class just represents a
vector space with a vector, and the model can transform
between spaces with a matrix multiply. The Conv2DSpace
represents a vector space as an image, and the model
can transform between spaces with a 2D convolution.

To make models as general as possible, models should be
written in terms of Spaces, rather than in terms of
numbers of hidden units, etc. The model should also be
written to transform between spaces using a generic
linear transformer from the pylearn2.linear module.

The Space class is needed so that the model can specify
what kinds of inputs it needs and what kinds of outputs
it will produce when communicating with other parts of
the library. The model also uses Space objects internally
to allocate parameters like hidden unit bias terms in
the right space.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import functools, warnings
import numpy as np
import theano
import theano.sparse
from theano import tensor
from theano.tensor import TensorType
from theano.gof.op import get_debug_values
from theano.sandbox.cuda.type import CudaNdarrayType

from pylearn2.utils import py_integer_types, safe_zip, sharedX, wraps
from pylearn2.format.target_format import OneHotFormatter

if theano.sparse.enable_sparse:
    # We know scipy.sparse is available
    import scipy.sparse

### FIXME do the right import in pylearn2.space


class Conv3DSpace(SimplyTypedSpace):
    """
    A space whose points are 4-D tensors representing (potentially
    multi-channel) 2D video with a fix temporal duration.

    Parameters
    ----------
    shape : sequence, length 3
        The shape of a single image, i.e. (rows, cols, time).
    num_channels : int (synonym: channels)
        Number of channels in the image, i.e. 3 if RGB.
    axes : tuple
        A tuple indicating the semantics of each axis, containing the
        following elements in some order:

            - 'b': this axis is the batch index of a minibatch.
            - 't': time axis
            - 0  : topological axis 0 (rows)
            - 1  : topological axis 1 (columns)
            - 'c': this axis tcnhe channel index of a minibatch.

    dtype : str
        A numpy dtype string (e.g. 'float32') indicating this space's
        dtype, or None for a dtype-agnostic space.
    kwargs : dict
        Passed on to superclass constructor
    """


    default_axes = ('b', 't', 0, 1, 'c')

    def __init__(self,
                 shape,
                 channels=None,
                 num_channels=None,
                 axes=None,
                 dtype='floatX',
                 **kwargs):

        super(Conv3DSpace, self).__init__(dtype, **kwargs)

        assert (channels is None) + (num_channels is None) == 1
        if num_channels is None:
            num_channels = channels

        assert isinstance(num_channels, py_integer_types)

        if not hasattr(shape, '__len__'):
            raise ValueError("shape argument for Conv3DSpace must have a "
                             "length. Got %s." % str(shape))

        if len(shape) != 3:
            raise ValueError("shape argument to Conv3DSpace must be length 3, "
                             "not %d" % len(shape))
        self.sequence_length = shape[2]

        assert all(isinstance(elem, py_integer_types) for elem in shape)
        assert all(elem > 0 for elem in shape)
        assert isinstance(num_channels, py_integer_types)
        assert num_channels > 0
        # Converts shape to a tuple, so it can be hashable, and self can be too
        self.shape = tuple(shape)
        self.num_channels = num_channels
        if axes is None:
            axes = self.default_axes
        assert len(axes) == 5
        self.axes = tuple(axes)

    def __str__(self):
        """Return a string representation"""
        return ("%s(shape=%s, num_channels=%d, axes=%s, dtype=%s)" %
                (self.__class__.__name__,
                 str(self.shape),
                 self.num_channels,
                 str(self.axes),
                 self.dtype))

    def __eq__(self, other):
        """
        Returns true iff
        space.format_as(batch, self) and
        space.format_as(batch, other) return the same formatted batch.
        """
        assert isinstance(self.axes, tuple)
        if isinstance(other, Conv3DSpace):
            assert isinstance(other.axes, tuple)
        return (type(self) == type(other) and
                self.shape == other.shape and
                self.num_channels == other.num_channels and
                self.axes == other.axes and
                self.dtype == other.dtype)

    def __hash__(self):
        """
        .. todo::

            WRITEME
        """
        return hash((type(self),
                     self.shape,
                     self.num_channels,
                     self.axes,
                     self.dtype))

    @functools.wraps(Space.get_batch_axis)
    def get_batch_axis(self):
        return self.axes.index('b')

    @functools.wraps(Space.get_origin)
    def get_origin(self):
        dims = {0: self.shape[0],
                1: self.shape[1],
                'c': self.num_channels,
                't': self.shape[2]}
        shape = [dims[elem] for elem in self.axes if elem != 'b']
        return np.zeros(shape, dtype = self.dtype)

    @functools.wraps(Space.get_origin_batch)
    def get_origin_batch(self, batch_size, dtype=None):
        if dtype is None:
            dtype = theano.config.floatX

        if not isinstance(batch_size, py_integer_types):
            raise TypeError("Conv3DSpace.get_origin_batch expects an int, "
                            "got %s of type %s" % (str(batch_size),
                                                   type(batch_size)))
        assert batch_size > 0
        dims = {'b': batch_size,
                0: self.shape[0],
                1: self.shape[1],
                't': self.shape[2],
                'c': self.num_channels}
        shape = [dims[elem] for elem in self.axes]
        return np.zeros(shape, dtype=dtype)


    @functools.wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        if dtype is None:
            dtype = theano.config.floatX

        broadcastable = [False] * 5
        broadcastable[self.axes.index('c')] = (self.num_channels == 1)
        broadcastable[self.axes.index('b')] = (batch_size == 1)
        broadcastable = tuple(broadcastable)

        rval = TensorType(dtype=dtype,
                          broadcastable=broadcastable)(name=name)
        if theano.config.compute_test_value != 'off':
            if batch_size == 1:
                rval.tag.test_value = self.get_origin_batch(batch_size=1)
            else:
                rval.tag.test_value = self.get_origin_batch(batch_size=4)
        return rval


    @functools.wraps(Space._batch_size_impl)
    def _batch_size_impl(self, is_numeric, batch):
        return batch.shape[self.axes.index('b')]

    @staticmethod
    def convert(tensor, src_axes, dst_axes):
        """
        Returns a view of tensor using the axis semantics defined
        by dst_axes. (If src_axes matches dst_axes, returns
        tensor itself)

        Useful for transferring tensors between different
        Conv3DSpaces.

        Parameters
        ----------
        tensor : tensor_like
            A 4-tensor representing a batch of images
        src_axes : tuple
            Axis semantics of tensor
        dst_axes : tuple
            Axis semantic of the destination space
        """
        src_axes = tuple(src_axes)
        dst_axes = tuple(dst_axes)
        assert len(src_axes) == 5
        assert len(dst_axes) == 5

        if src_axes == dst_axes:
           return tensor
        shuffle = [src_axes.index(elem) for elem in dst_axes]
        if is_symbolic_batch(tensor):
            return tensor.dimshuffle(*shuffle)
        else:
            return tensor.transpose(*shuffle)

    @staticmethod
    def convert_numpy(tensor, src_axes, dst_axes):
        """
        .. todo::

            WRITEME
        """
        return Conv3DSpace.convert(tensor, src_axes, dst_axes)

    @functools.wraps(Space.get_total_dimension)
    def get_total_dimension(self):
        # Patch old pickle files
        if not hasattr(self, 'num_channels'):
            self.num_channels = self.nchannels
        assert len(self.shape) == 3
        return self.shape[0] * self.shape[1] * self.shape[2] * self.num_channels

    @functools.wraps(Space._validate_impl)
    def _validate_impl(self, is_numeric, batch):
        # checks batch.type against self.dtype
        super(Conv3DSpace, self)._validate_impl(is_numeric, batch)

        if isinstance(batch, theano.gof.Variable):
            if isinstance(batch, theano.sparse.SparseVariable):
                raise TypeError("Conv3DSpace cannot use SparseVariables, "
                                "since as of this writing (8 May 2014), "
                                "there is not yet a SparseVariable type with "
                                "5 dimensions")

            if not isinstance(batch, theano.gof.Variable):
                raise TypeError("Conv3DSpace batches must be theano "
                                "Variables, got " + str(type(batch)))

            if not isinstance(batch.type, (theano.tensor.TensorType,
                                           CudaNdarrayType)):
                raise TypeError('Expected TensorType or CudaNdArrayType, got '
                                '"%s"' % type(batch.type))

            if batch.ndim != 5:
                raise ValueError("The value of a Conv3DSpace batch must be "
                                 "5D, got %d dimensions for %s." %
                                 (batch.ndim, batch))

            for val in get_debug_values(batch):
                self.np_validate(val)
        else:
            if scipy.sparse.issparse(batch):
                raise TypeError("Conv3DSpace cannot use sparse batches, since "
                                "scipy.sparse does not support 5 dimensional "
                                "tensors currently (8 May 2014).")

            if (not isinstance(batch, np.ndarray)) \
               and type(batch) != 'CudaNdarray':
                raise TypeError("The value of a Conv3DSpace batch should be a "
                                "numpy.ndarray, or CudaNdarray, but is %s."
                                % str(type(batch)))

            if batch.ndim != 5:
                raise ValueError("The value of a Conv3DSpace batch must be "
                                 "5D, got %d dimensions for %s." %
                                 (batch.ndim, batch))

            d = self.axes.index('c')
            actual_channels = batch.shape[d]
            if actual_channels != self.num_channels:
                raise ValueError("Expected axis %d to be number of channels "
                                 "(%d) but it is %d" %
                                 (d, self.num_channels, actual_channels))
            assert batch.shape[self.axes.index('c')] == self.num_channels

            for coord in [0, 1]:
                d = self.axes.index(coord)
                actual_shape = batch.shape[d]
                expected_shape = self.shape[coord]
                if actual_shape != expected_shape:
                    raise ValueError("Conv3DSpace with shape %s and axes %s "
                                     "expected dimension %s of a batch (%s) "
                                     "to have length %s but it has %s"
                                     % (str(self.shape),
                                        str(self.axes),
                                        str(d),
                                        str(batch),
                                        str(expected_shape),
                                        str(actual_shape)))

    @functools.wraps(Space._format_as_impl)
    def _format_as_impl(self, is_numeric, batch, space):
        if isinstance(space, VectorSpace):
            # We need to ensure that the resulting batch will always be
            # the same in `space`, no matter what the axes of `self` are.
            if self.axes != self.default_axes:
                # The batch index goes on the first axis
                assert self.default_axes[0] == 'b'
                batch = batch.transpose(*[self.axes.index(axis)
                                          for axis in self.default_axes])
            result = batch.reshape((batch.shape[0],
                                    self.get_total_dimension()))
            if space.sparse:
                result = _dense_to_sparse(result)

        elif isinstance(space, Conv3DSpace):
            result = Conv3DSpace.convert(batch, self.axes, space.axes)
        else:
            raise NotImplementedError("%s doesn't know how to format as %s"
                                      % (str(self), str(space)))

        return _cast(result, space.dtype)

