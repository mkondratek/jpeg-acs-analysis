import numpy as np
import scipy
import tensorflow as tf
import torchjpeg.codec
from progress.bar import Bar
import cv2

rgb2lms = np.array([
    [17.8824, 43.5161, 4.1194],
    [3.4557, 27.1554, 3.8671],
    [0.0300, 0.1843, 1.4671]
])

lms2xyb = np.array([
    [1, -1, 0],
    [1, 1, 0],
    [0, 0, 1]
], dtype=np.float64)

rgb2xyb = np.dot(lms2xyb, rgb2lms)


def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def load_xyb(fname, *, transpose):
    rgb = cv2.imread(fname)
    h, w, _ = rgb.shape
    xyb = np.zeros((h, w, 3))
    for y in range(h):
        for x in range(w):
            xyb[y][x] = np.dot(rgb2xyb, rgb[y][x])

    xyb = np.array([xyb[:, :, 0], xyb[:, :, 1], xyb[:, :, 2]])

    for c in [0, 1, 2]:
        for y in range(0, h, 8):
            for x in range(0, w, 8):
                xyb[c][y:(y + 8), x:(x + 8)] = dct2(xyb[c][y:(y + 8), x:(x + 8)])

    for y in range(h):
        for x in range(w):
            print(f'bx={x} by={y}:')
            for a in range(8):
                for b in range(8):
                    print(f'{xyb[y][x][b][a]}', end=' ')
                print()


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
