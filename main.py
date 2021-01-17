import time

import matplotlib.pyplot as plt
import numpy as np
import threading
from jpeg import parse
from progress.bar import Bar


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


def load_data(fname, *, transpose):
    arr = parse(fname, normalize=True, quality=100, subsampling='keep', upsample=True, stack=True)
    height = arr.shape[1]
    width = arr.shape[2]
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
    height = data.shape[1]
    width = data.shape[2]
    dataset = np.zeros((3, height - 1, width - 1, 8, 8, 16), dtype=int)
    threads = []

    for c in range(3):
        thread = threading.Thread(target=derive_dataset_impl, args=(data, dataset, c))
        threads.append(thread)
        thread.start()

    for c in range(3):
        threads[c].join()

    m = (height - 1) * (width - 1) * 8 * 8
    return (dataset.reshape(3, m, 16), data[:, 1:, 1:].reshape(3, m))


def derive_dataset_impl(data, dataset, c):
    height = data.shape[1]
    width = data.shape[2]

    for y in range(1, height):
        for x in range(1, width):
            for j in range(8):
                left_acs = data[c, y, x - 1, j]
                for i in range(8):
                    top_acs = data[c, y - 1, x, :, i]
                    dataset[c, y - 1, x - 1, j, i] = np.concatenate((left_acs, top_acs), axis=None)


points = [0, 1, 8, 9]

points_len = len(points)


def derive_cffs(dataset, values):
    threads = []
    cffs_arr = np.zeros((3 * points_len, 16))

    for c in range(3):
        thread = threading.Thread(target=derive_cffs_impl, args=(dataset, values, cffs_arr, c))
        threads.append(thread)
        thread.start()

    for c in range(3):
        threads[c].join()

    return cffs_arr


def derive_cffs_impl(dataset, values, cffs_arr, c):
    for i in range(points_len):
        p = points[i]
        args = dataset[c][p::64]
        vals = values[c][p::64]
        res = np.linalg.lstsq(args, vals, rcond=None)
        cffs_arr[c * points_len + i] = res[0]


def print_cffs_as_cpp_array(cffs_arr):
    out = [f'const float coeffs[3][{points_len}][16] = {{']
    for c in range(3):
        out.append('\t{')
        for i in range(points_len):
            cffs_str = ', '.join(map(lambda x: f'{x:.9f}', cffs_arr[c * points_len + i]))
            out.append(f'\t\t{ {cffs_str} },'.replace('\'', ''))
        out.append('\t},')
    out.append('};')
    return '\n'.join(out)


if __name__ == '__main__':
    # data = load_data('StockSnap_7QH4K6AESO.jpg', transpose=True)
    data = load_data('other.jpg', transpose=True)

    (dataset, values) = derive_dataset(data)

    cffs = derive_cffs(dataset, values)

    out = print_cffs_as_cpp_array(cffs)
    time_seconds = time.time()
    with open(f'cpp_array_{time_seconds}.txt', 'w') as f:
        f.write(out)
        f.close()

    for i in range(points_len):
        p = points[i]
        _ = plt.plot(cffs[i], 'k', label=f'Point {p}; color 0 (Y)')
        _ = plt.plot(cffs[i + points_len], 'b', label=f'Point {p}; color 1 (Cb)')
        _ = plt.plot(cffs[i + 2 * points_len], 'r', label=f'Point {p}; color 2 (Cr)')
        _ = plt.legend()
        plt.savefig(f'ac{p}_{time_seconds}.png')
        plt.clf()
