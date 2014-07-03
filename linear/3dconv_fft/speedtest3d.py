import time
import cPickle as pickle
import numpy as np
import theano
import theano.tensor as T

from theano.sandbox.cuda.basic_ops import gpu_from_host

# FFT-based convolution implementation
import fftconv

target_path = "speedtest_data.pkl"

num_runs = 10 # number of times each convolution is run,
# running time is averaged across these runs.

atol = 1e-3
rtol = 1e-5
std = 0.1

shapes_list = [
    # (input_shape, filter_shape)
    # ((minibatch_size, num_input_channels, time, image_width, image_height),
    #  (num_filters, num_input_channels, time, filter_width, filter_height))
    ((64, 128, 12, 32, 32), (64, 128, 3, 8, 8)),
    ((64, 33, 12, 20, 16), (128, 33, 3, 2, 2)),
]


x = theano.shared(np.zeros((1,1,1,1,1), dtype=theano.config.floatX))
w = theano.shared(np.zeros((1,1,1,1,1), dtype=theano.config.floatX))


def estimate_running_time(func):
    start_time = time.time()
    for _ in xrange(num_runs):
        func()
    duration = time.time() - start_time
    return duration / float(num_runs)


results = {}


for shape_x, shape_w in shapes_list:
    print
    print "X: %s" % str(shape_x)
    print "W: %s" % str(shape_w)
    print

    x_val = np.random.randn(*shape_x).astype(theano.config.floatX) * std
    w_val = np.random.randn(*shape_w).astype(theano.config.floatX) * std

    x.set_value(x_val)
    w.set_value(w_val)

    y_fft = fftconv.conv3d_fft(x, w, image_shape=shape_x, filter_shape=shape_w)

    print "  compiling: FFT"
    f_fft = theano.function([], gpu_from_host(y_fft)) # don't transfer to host

    print

    #print "  verifying accuracy"
    # wrapping the function output in np.array causes a transfer to the host.
    #out_fft = np.array(f_fft())
    #assert np.allclose(out_fft, atol=atol, rtol=rtol)

    print

    print "  running time: FFT\t\t",
    t_fft = estimate_running_time(f_fft)
    print "%.5f s" % t_fft

    print

    results_run = {
        'fft': t_fft,
    }

    results[(shape_x, shape_w)] = results_run

    # memory cleanup
    del f_fft



