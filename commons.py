import numpy as np
import scipy
import tensorflow as tf
import torchjpeg.codec
from progress.bar import Bar
import cv2
from jpeg import parse

points = [0, 1, 8, 9]

points_len = len(points)


def load_jpeg(fname, *, transpose=False):
    arr = parse(fname, normalize=False, quality=100, subsampling='4:4:4', upsample=False, stack=True)
    height = arr.shape[1]
    width = arr.shape[2]
    data = np.zeros([3, height, width, 8, 8], dtype=int)
    c_map = {0: 1, 1: 0, 2: 2}

    with Bar('Loading & transposing blocks...', max=3 * height * width) as bar:
        for c in range(3):
            for y in range(height):
                for x in range(width):
                    data[c][y][x] = np.reshape(arr[c_map.get(c)][y][x], (8, 8))
                    if transpose:
                        data[c][y][x] = data[c][y][x].T

                    # The dct[0][0] (idc) cannot be used in libjxl ac prediction for some reason.
                    # Let's do not include it as a factor here then.
                    data[c][y][x][0][0] = 0
                    bar.next()

    return data


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


def derive_dataset(data):
    height = data.shape[0]
    width = data.shape[1]
    dataset = np.zeros((height - 1, width - 1, points_len, 16), dtype=int)
    values = np.zeros((height - 1, width - 1, points_len), dtype=int)

    derive_dataset_impl(data, dataset, values)

    m = (height - 1) * (width - 1) * points_len
    return (dataset.reshape(m, 16), values.reshape(m))


def derive_dataset_impl(data, dataset, values):
    height = data.shape[0]
    width = data.shape[1]

    for y in range(1, height):
        for x in range(1, width):
            for k, p in enumerate(points):
                j = p // 8
                i = p % 8
                left_acs = data[y, x - 1, j]
                top_acs = data[y - 1, x, :, i]
                dataset[y - 1, x - 1, k] = np.concatenate((left_acs, top_acs), axis=None)
                values[y - 1, x - 1, k] = data[y, x, j, i]


def write_cffs_as_cpp_array(wd, cffs):
    with open(f'{wd}/cpp_array.txt', 'w') as f:
        out = [f'const float coeffs[3][{points_len}][16] = {{']
        for c in range(3):
            out.append('\t{')
            for i in range(points_len):
                cffs_str = ', '.join(map(lambda x: f'{x:.9f}', cffs[c][i]))
                out.append(f'\t\t{ {cffs_str} },'.replace('\'', ''))
            out.append('\t},')
        out.append('};')
        f.write('\n'.join(out))


def write_cffs_as_plain_numbers(wd, jpg, cffs):
    with open(f'{wd}/{jpg.replace(".", "_")}.txt', 'w') as f:
        out = []
        for c in range(3):
            for i in range(points_len):
                cffs_str = ' '.join(map(lambda x: f'{x:.9f}', cffs[c][i]))
                out.append(cffs_str.replace('\'', ''))
        f.write('\n'.join(out))


def print_acs(data):
    height = data.shape[1]
    width = data.shape[2]
    for c in range(3):
        for y in range(height):
            for x in range(width):
                print(f'(c={c}, bx={x}, by={y}) block values:')
                for j in range(8):
                    for i in range(8):
                        print(f'{data[c][y][x][j][i]:8}', end='')
                    print()
