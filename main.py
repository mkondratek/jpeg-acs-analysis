import os
from os.path import basename

import matplotlib.pyplot as plt
import numpy as np
import threading

from commons import load_data

points = [0, 1, 8, 9]

points_len = len(points)


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


def derive_dataset(data):
    height = data.shape[1]
    width = data.shape[2]
    dataset = np.zeros((3, height - 1, width - 1, points_len, 16), dtype=int)
    values = np.zeros((3, height - 1, width - 1, points_len), dtype=int)
    threads = []

    for c in range(3):
        thread = threading.Thread(target=derive_dataset_impl, args=(data, dataset, values, c))
        threads.append(thread)
        thread.start()

    for c in range(3):
        threads[c].join()

    m = (height - 1) * (width - 1) * points_len
    return (dataset.reshape(3, m, 16), values.reshape(3, m))


def derive_dataset_impl(data, dataset, values, c):
    height = data.shape[1]
    width = data.shape[2]

    for y in range(1, height):
        for x in range(1, width):
            for k, p in enumerate(points):
                j = p // 8
                i = p % 8
                left_acs = data[c, y, x - 1, j]
                top_acs = data[c, y - 1, x, :, i]
                dataset[c, y - 1, x - 1, k] = np.concatenate((left_acs, top_acs), axis=None)
                values[c, y - 1, x - 1, k] = data[c, y, x, j, i]


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
    for i, p in enumerate(points):
        args = dataset[c][i::points_len]
        vals = values[c][i::points_len]
        np.savetxt(f'args_{i}.csv', args, delimiter=',')
        np.savetxt(f'vals_{i}.csv', vals, delimiter=',')
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


def run(jpg):
    wd = basename(jpg).replace('.', '_')
    if os.path.exists(wd):
        return
    os.mkdir(wd)
    data = load_data(jpg, transpose=True)
    (dataset, values) = derive_dataset(data)
    cffs = derive_cffs(dataset, values)
    out = print_cffs_as_cpp_array(cffs)
    with open(f'{wd}/cpp_array.txt', 'w') as f:
        f.write(out)
    for i in range(points_len):
        p = points[i]
        _ = plt.plot(cffs[i], 'k', label=f'Point {p}; color 0 (Y)')
        _ = plt.plot(cffs[i + points_len], 'b', label=f'Point {p}; color 1 (Cb)')
        _ = plt.plot(cffs[i + 2 * points_len], 'r', label=f'Point {p}; color 2 (Cr)')
        _ = plt.legend()
        plt.savefig(f'{wd}/ac{p}.png')
        plt.clf()


def main():
    for jpg in os.listdir('input'):
        run(f'input/{jpg}')


if __name__ == '__main__':
    main()
