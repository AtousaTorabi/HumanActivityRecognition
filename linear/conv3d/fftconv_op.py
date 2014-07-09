import numpy as np
import theano
import theano.tensor as T

import theano.sandbox.cuda as cuda
from theano.misc.pycuda_utils import to_gpuarray

import scikits.cuda
from scikits.cuda import fft
from scikits.cuda import linalg
from scikits.cuda import cublas

import pycuda.gpuarray

import theano.misc.pycuda_init
import string

import HumanActivityRecognition.linear.conv3d.fftconv as conv

linalg.init()




class Conv3DFFT(cuda.GpuOp):

    def __init__(self, input_shape, filter_shape):
        self.input_shape = input_shape
        self.filter_shape = filter_shape

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False] * (inp.type.ndim))

    def make_node(self, inp, filters):

        inp = cuda.basic_ops.gpu_contiguous(
            cuda.basic_ops.as_cuda_ndarray_variable(inp))
        filters = cuda.basic_ops.gpu_contiguous(
            cuda.basic_ops.as_cuda_ndarray_variable(filters))

        assert inp.dtype == "float32"
        return theano.Apply(self, [inp, filters], [self.output_type(inp)()])



    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():
            inp = inputs[0][0]
            filters = inputs[1][0]

            # output_shape = self.input_shape
            # output_shape[-1] = (output_shape[-1] - 1) * 2 # restore full signal length
            # output_shape = tuple(output_shape)

            z = outputs[0]
            # batch size, input channels, input dim 0, input dim 1
            b, ic, i0, i1, i2 = self.input_shape 
            # output channels, input channels, filter dim 0, filter dim 1
            oc, ic_, f0, f1, f2 = self.filter_shape 
            # Output shape
            output_shape = [b, oc, i0 - f0 + 1, i1 - f1 + 1, i2 - f2 + 1]

            # only allocate if there is no previous allocation of the right size.
            if z[0] is None or z[0].shape != output_shape:
                 z[0] = cuda.CudaNdarray.zeros(output_shape)

            z = conv.conv3d_fft(inp, filters,
                                self.input_shape,
                                self.filter_shape)

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk



        
    def grad(self, inputs, dout):
        print "here"

        inputs, filters = inputs
        # if 'Cuda' not in str(type(images)):
        #     raise TypeError("inputs must be cuda")
        # if 'Cuda' not in str(type(filters)):
        #     raise TypeError("filters must be cuda")

        dout, = dout
        #dout = cuda.basic_ops.gpu_contiguous(
        #    cuda.basic_ops.as_cuda_ndarray_variable(dout))
        #dout = gpu_contiguous(dout)
        # if 'Cuda' not in str(type(dout)):
        #     raise TypeError("output gradients must be cuda")


        #filters = filters.dimshuffle(1, 0, 2, 3, 4)
        #inputs = inputs.dimshuffle(1, 0, 2, 3, 4)
        #print inputs.shape, filters.shape, dout.shape
        d_inputs = conv.conv3d_fft(dout, inputs, dout.shape, inputs.shape)
        d_filters = conv.conv3d_fft(dout, filters.T, dout.shape, filters.T.shape)
        
        return d_inputs, filters
