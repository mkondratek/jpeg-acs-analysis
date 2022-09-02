import numpy as np
import tensorflow as tf
import torchjpeg.codec
from progress.bar import Bar


def load_data(fname, *, transpose):

    dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients(fname)
    arr = tf.concat([Y_coefficients, CbCr_coefficients], 0)
    height = Y_coefficients.shape[1]
    width = Y_coefficients.shape[2]
    data = np.zeros([3, height, width, 8, 8], dtype=int)

    with Bar('Loading & transposing blocks...', max=3 * height * width) as bar:
        for c in range(3):
            for y in range(height):
                for x in range(width):
                    data[c][y][x] = np.reshape(arr[c][y][x], (8, 8))
                    if transpose:
                        data[c][y][x] = data[c][y][x].T
                    bar.next()

    return data