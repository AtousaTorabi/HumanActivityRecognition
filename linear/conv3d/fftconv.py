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

linalg.init()


# TODO define gradient!!!!!!!


# TODO: implement __eq__ and __hash__ correctly
# TODO: Find out if scikits.cuda.fft.fft is destructive - if so we need to specify a destroy_map
# TODO: investigate FFTW compatibility modes. Can probably set this to the fastest setting.
# TODO: investigate the effect of enabling fastmath on FFT performance (how can it be enabled?).


class ScikitsCudaOp(cuda.GpuOp): # base class for shared code between scikits.cuda-based ops
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def output_type(self, inp):
        raise NotImplementedError

    def make_node(self, inp):
        inp = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp))

        assert inp.dtype == "float32"

        return theano.Apply(self, [inp], [self.output_type(inp)()])



class CuFFTOp(ScikitsCudaOp):
    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False] * (inp.type.ndim + 1)) # add one extra dim for real/imag

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        plan_input_shape = [None]
        plan = [None]

        def thunk():
            input_shape = inputs[0][0].shape


            # construct output shape
            output_shape = list(input_shape)
            output_shape[-1] = output_shape[-1] // 2 + 1 # DFT of real input is symmetric, no need to store redundant coefficients
            output_shape += [2] # extra dimension with length 2 for real/imag
            output_shape = tuple(output_shape)

            print "fft:", input_shape, output_shape


            z = outputs[0]

            # only allocate if there is no previous allocation of the right size.
            if z[0] is None or z[0].shape != output_shape:
                z[0] = cuda.CudaNdarray.zeros(output_shape)

            input_pycuda = to_gpuarray(inputs[0][0])
            # I thought we'd need to change the type on output_pycuda so it is complex64,
            # but as it turns out scikits.cuda.fft doesn't really care either way and
            # treats the array as if it is complex64 anyway.
            output_pycuda = to_gpuarray(z[0])

            # only initialise plan if necessary
            if plan[0] is None or plan_input_shape[0] != input_shape:
                plan_input_shape[0] = input_shape
                plan[0] = fft.Plan(input_shape[1:], np.float32, np.complex64, batch=input_shape[0])

            fft.fft(input_pycuda, output_pycuda, plan[0])

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk



class CuIFFTOp(ScikitsCudaOp):
    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False] * (inp.type.ndim - 1)) # remove extra real/imag dim

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        plan_input_shape = [None]
        plan = [None]

        def thunk():
            input_shape = inputs[0][0].shape

            # construct output shape
            output_shape = list(input_shape[:-1]) # chop off the extra length-2 dimension for real/imag
            output_shape[-1] = (output_shape[-1] - 1) * 2 # restore full signal length
            output_shape = tuple(output_shape)

            print "ifft:", input_shape, output_shape

            z = outputs[0]

            # only allocate if there is no previous allocation of the right size.
            if z[0] is None or z[0].shape != output_shape:
                z[0] = cuda.CudaNdarray.zeros(output_shape)

            input_pycuda = to_gpuarray(inputs[0][0])
            # input_pycuda is a float32 array with an extra dimension, but will be
            # interpreted by scikits.cuda as a complex64 array instead.
            output_pycuda = to_gpuarray(z[0])

            # only initialise plan if necessary
            if plan[0] is None or plan_input_shape[0] != input_shape:
                plan_input_shape[0] = input_shape
                plan[0] = fft.Plan(output_shape[1:], np.complex64, np.float32, batch=output_shape[0])

            fft.ifft(input_pycuda, output_pycuda, plan[0]) # , True)
            # strangely enough, enabling rescaling here makes it run very, very slowly.
            # so do this rescaling manually afterwards!

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk

    def grad(self, inp, output_gradients):
        return inp, output_gradients




def to_complex_gpuarray(x, copyif=False):
    """
    adapted version of theano.misc.pycuda_utils.to_gpuarray that takes an array with an extra trailing
    dimension of length 2 for real/imaginary parts, and turns it into a complex64 PyCUDA GPUArray.
    """
    if not isinstance(x, cuda.CudaNdarray):
        raise ValueError("We can transfer only CudaNdarray to pycuda.gpuarray.GPUArray")
    else:
        # Check if trailing dimension has length 2
        assert x.shape[-1] == 2

        # check if dtype is float32
        assert x.dtype == 'float32'

        # Check if it is c contiguous
        size = 1
        c_contiguous = True
        for i in range(x.ndim-1, -1, -1):
            if x.shape[i] == 1:
                continue
            if x._strides[i] != size:
                c_contiguous = False
                break
            size *= x.shape[i]
        if not c_contiguous:
            if copyif:
                x = x.copy()
            else:
                raise ValueError("We were asked to not copy memory, but the memory is not c contiguous.")

        # Now x is always c contiguous
        px = pycuda.gpuarray.GPUArray(x.shape[:-1], np.complex64, base=x, gpudata=x.gpudata)
        return px


def bptrs(a):
    """
    Pointer array when input represents a batch of matrices.

    taken from scikits.cuda tests/test_cublas.py
    """

    return pycuda.gpuarray.arange(a.ptr,a.ptr+a.shape[0]*a.strides[0],a.strides[0],
                dtype=cublas.ctypes.c_void_p)


def sc_complex_dot_batched(bx_gpu, by_gpu, bc_gpu, transa='N', transb='N', handle=None):
    """
    uses cublasCgemmBatched to compute a bunch of complex dot products in parallel
    """
    if handle is None:
        handle = scikits.cuda.misc._global_cublas_handle

    assert len(bx_gpu.shape) == 3
    assert len(by_gpu.shape) == 3
    assert len(bc_gpu.shape) == 3
    assert bx_gpu.dtype == np.complex64
    assert by_gpu.dtype == np.complex64
    assert bc_gpu.dtype == np.complex64

    # Get the shapes of the arguments
    bx_shape = bx_gpu.shape
    by_shape = by_gpu.shape

    # Perform matrix multiplication for 2D arrays:
    alpha = np.complex64(1.0)
    beta = np.complex64(0.0)

    transa = string.lower(transa)
    transb = string.lower(transb)

    if transb in ['t', 'c']:
        N, m, k = by_shape
    elif transb in ['n']:
        N, k, m = by_shape
    else:
        raise ValueError('invalid value for transb')

    if transa in ['t', 'c']:
        N2, l, n = bx_shape
    elif transa in ['n']:
        N2, n, l = bx_shape
    else:
        raise ValueError('invalid value for transa')

    if l != k:
        raise ValueError('objects are not aligned')

    if N != N2:
        raise ValueError('batch sizes are not the same')

    if transb == 'n':
        lda = max(1, m)
    else:
        lda = max(1, k)

    if transa == 'n':
        ldb = max(1, k)
    else:
        ldb = max(1, n)

    ldc = max(1, m)

    # construct pointer arrays needed for cublasCgemmBatched
    bx_arr = bptrs(bx_gpu)
    by_arr = bptrs(by_gpu)
    bc_arr = bptrs(bc_gpu)

    cublas.cublasCgemmBatched(handle, transb, transa, m, n, k, alpha, by_arr.gpudata,
                lda, bx_arr.gpudata, ldb, beta, bc_arr.gpudata, ldc, N)



class BatchedComplexDotOp(ScikitsCudaOp):
    """
    This version uses cublasCgemmBatched under the hood, instead of
    doing multiple cublasCgemm calls.
    """
    def make_node(self, inp1, inp2):
        inp1 = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp1))
        inp2 = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp2))

        assert inp1.dtype == "float32"
        assert inp2.dtype == "float32"
        assert inp1.ndim == 4 # (batch, a, b, real/imag)
        assert inp2.ndim == 4

        return theano.Apply(self, [inp1, inp2], [self.output_type(inp1)()])

    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False] * inp.type.ndim)

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():

            print "jheeeeeeeeere"
            bx = inputs[0]
            by = inputs[1]

            input_shape_x = bx[0].shape # (batch, a, b, 2)
            input_shape_y = by[0].shape # (batch, b, c, 2)

            output_shape = (input_shape_x[0], input_shape_x[1], input_shape_y[2], 2) # (batch, a, c, 2)

            bz = outputs[0]

            # only allocate if there is no previous allocation of the right size.
            if bz[0] is None or bz[0].shape != output_shape:
                bz[0] = cuda.CudaNdarray.zeros(output_shape)

            input_bx_pycuda = to_complex_gpuarray(bx[0])
            input_by_pycuda = to_complex_gpuarray(by[0])
            output_b_pycuda = to_complex_gpuarray(bz[0])

            # fancy native batched version
            print "jheeeeeeeeere2"
            sc_complex_dot_batched(input_bx_pycuda, input_by_pycuda, output_b_pycuda)
            print "jheeeeeeeeere3"
            print input_shape_x, input_shape_y
            print input_by_pycuda.shape, input_by_pycuda.shape, output_b_pycuda.shape

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk





cufft = CuFFTOp()
cuifft = CuIFFTOp()
batched_complex_dot = BatchedComplexDotOp()


def mult_and_reduce(input_fft_v, filters_fft_v, input_shape=None, filter_shape=None):
    """
    input_fft_v is (b, ic, i0, i1//2 + 1, 2)
    filters_fft_v is (oc, ic, i0, i1//2 + 1, 2)
    """

    if input_shape is None:
        input_shape =  input_fft_v.shape # symbolic

    if filter_shape is None:
        filter_shape = filters_fft_v.shape # symbolic

    b, ic, i0, i1_f, _ = input_shape
    oc = filter_shape[0]


    # reshape to flatten the dimensions that are multiplied elemwise
    input_r = input_fft_v.reshape((b, ic, i0 * i1_f, 2))
    filters_r = filters_fft_v.reshape((oc, ic, i0 * i1_f, 2))

    # shuffle for batched dot product
    input_s = input_r.dimshuffle(2, 0, 1, 3) # (i0 * i1_f, b, ic, 2)
    filters_s = filters_r.dimshuffle(2, 1, 0, 3) # (i0 * i1_f, ic, oc, 2)

    output_s = batched_complex_dot(input_s, filters_s)

    # shuffle again
    output_r = output_s.dimshuffle(1, 2, 0, 3)

    # reshape to unflatten
    output = output_r.reshape((b, oc, i0, i1_f, 2))

    return output


def mult_and_reduce3d(input_fft_v, filters_fft_v, input_shape=None, filter_shape=None):
    """
    input_fft_v is (b, ic, i0, i1, i2 //2 + 1, 2)
    filters_fft_v is (oc, ic, i0, i1, i2 //2 + 1, 2)
    """

    if input_shape is None:
        input_shape =  input_fft_v.shape # symbolic

    if filter_shape is None:
        filter_shape = filters_fft_v.shape # symbolic

    b, ic, i0, i1_f, i2_f, _ = input_shape
    oc = filter_shape[0]

    # reshape to flatten the dimensions that are multiplied elemwise
    input_r = input_fft_v.reshape((b, ic, i0 * i1_f * i2_f, 2))
    filters_r = filters_fft_v.reshape((oc, ic, i0 * i1_f * i2_f, 2))

    # shuffle for batched dot product
    input_s = input_r.dimshuffle(2, 0, 1, 3) # (i0 * i1_f, b, ic, 2)
    filters_s = filters_r.dimshuffle(2, 1, 0, 3) # (i0 * i1_f, ic, oc, 2)

    output_s = batched_complex_dot(input_s, filters_s)

    # shuffle again
    output_r = output_s.dimshuffle(1, 2, 0, 3)

    # reshape to unflatten
    output = output_r.reshape((b, oc, i0, i1_f, i2_f, 2))

    return output





def conv2d_fft(input, filters, image_shape=None, filter_shape=None):
    """
    expects bc01 input
    performs a valid convolution

    input: (b, ic, i0, i1)
    filters: (oc, ic, f0, f1)
    """

    # use symbolic shapes to compute shape info at runtime if not specified
    if image_shape is None:
        image_shape = input.shape

    if filter_shape is None:
        filter_shape = filters.shape

    b, ic, i0, i1 = image_shape # batch size, input channels, input dim 0, input dim 1
    oc, ic_, f0, f1 = filter_shape # output channels, input channels, filter dim 0, filter dim 1

    # pad filters to input shape
    filters_padded = T.zeros((oc, ic, i0, i1))
    filters_padded = T.set_subtensor(filters_padded[:, :, :f0, :f1], filters)

    # reshape for FFT
    input_flat = input.reshape((b * ic, i0, i1))
    filters_flat = filters_padded.reshape((oc * ic, i0, i1))

    # perform FFT
    input_fft_flat = cufft(input_flat) # (b * ic, i0, i1//2 + 1, 2)
    filters_fft_flat = cufft(filters_flat) # (oc * ic, i0, i1//2 + 1, 2)

    # unfold ic dimension
    input_fft_v_shape = (b, ic, i0, i1//2 + 1, 2)
    filters_fft_v_shape = (oc, ic, i0, i1//2 + 1, 2)
    input_fft_v = input_fft_flat.reshape(input_fft_v_shape)
    filters_fft_v = filters_fft_flat.reshape(filters_fft_v_shape)

    output_fft_s = mult_and_reduce(input_fft_v, filters_fft_v,
                            input_shape=input_fft_v_shape, filter_shape=filters_fft_v_shape) # (b, oc, i0, i1//2 + 1, 2)

    # reshape for IFFT
    output_fft_flat = output_fft_s.reshape((b * oc, i0, i1//2 + 1, 2))

    # perform IFFT
    output_flat = cuifft(output_fft_flat) # (b * oc, i0, i1)

    # reshape
    output_circ = output_flat.reshape((b, oc, i0, i1)) # circular!

    # slice because the convolution was circular, we need it to be valid
    output = output_circ[:, :, f0 - 1:, f1 - 1:]

    # rescale manually
    output = (1.0 / T.cast(i0 * i1, theano.config.floatX)) * output # allow for the scale factor to move to the gpu

    # output should now be the result of a batched valid convolution of the input with the filters.
    return output


class ShapePrint(ScikitsCudaOp):
    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False] * (inp.type.ndim))

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():
            input_shape = inputs[0][0].shape
            print input_shape
            outputs[0] = inputs[0]

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False
        return thunk


shapeprint = ShapePrint()
def conv3d_fft(input, output,
               filters, image_shape, filter_shape):
    """
    expects bc01 input
    performs a valid convolution

    input: (b, ic, i0, i1)
    filters: (oc, ic, f0, f1)
    """

    b, ic, i0, i1, i2 = image_shape # batch size, input channels, input dim 0, input dim 1
    oc, ic_, f0, f1, f2 = filter_shape # output channels, input channels, filter dim 0, filter dim 1


    input = shapeprint(input)

    # pad filters to input shape
    filters_padded = T.zeros((oc, ic, i0, i1, i2))
    filters_padded = T.set_subtensor(filters_padded[:, :, :f0, :f1, :f2], filters)

    # reshape for FFT
    input_flat = input.reshape((b * ic, i0, i1, i2))
    filters_flat = filters_padded.reshape((oc * ic, i0, i1, i2))

    # perform FFT
    # FIXME what is actually the return type?
    input_fft_flat = cufft(input_flat) # (b * ic, i0, i1, i2 //2 + 1, 2)
    filters_fft_flat = cufft(filters_flat) # (oc * ic, i0, i1, i2 //2 + 1, 2)


    # unfold ic dimension
    input_fft_v_shape = (b, ic, i0, i1, i2 // 2 + 1, 2)
    filters_fft_v_shape = (oc, ic, i0, i1, i2 // 2 + 1, 2)
    input_fft_v = input_fft_flat.reshape(input_fft_v_shape)
    filters_fft_v = filters_fft_flat.reshape(filters_fft_v_shape)

    output_fft_s = mult_and_reduce3d(input_fft_v, filters_fft_v,
                                     input_shape = input_fft_v_shape, 
                                     filter_shape = filters_fft_v_shape) # (b, oc, i0, i1, i2 //2 + 1, 2)

    # reshape for IFFT
    output_fft_flat = output_fft_s.reshape((b * oc, i0, i1, i2 //2 + 1, 2))

    # perform IFFT
    output_flat = cuifft(output_fft_flat) # (b * oc, i0, i1, i2)

    # reshape
    output_circ = output_flat.reshape((b, oc, i0, i1, i2)) # circular!

    # slice because the convolution was circular, we need it to be valid
    output = output_circ[:, :, f0 - 1:, f1 - 1:, f2 -1:] # b, oc, 

    # rescale manually
    output = (1.0 / T.cast(i0 * i1, theano.config.floatX)) * output # allow for the scale factor to move to the gpu


    # output should now be the result of a batched valid convolution of the input with the filters.
#    return output



