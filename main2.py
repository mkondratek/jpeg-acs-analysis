import os
import threading
from os.path import basename

import matplotlib.pyplot as plt
import numpy as np

from commons import load_jpeg, points_len, points, write_cffs_as_plain_numbers, write_cffs_as_cpp_array, derive_dataset


def derive_cffs(dataset, values, cffs):
    for i, p in enumerate(points):
        args = dataset[i::points_len]
        vals = values[i::points_len]
        res = np.linalg.lstsq(args, vals, rcond=None)
        cffs[i] = res[0]


def add_dataset_and_values(channel, set_path, dataset, values):
    listdir = os.listdir(set_path)

    for f in listdir:
        print(f'Processing ({channel}) {f}...')
        data = load_jpeg(f'{set_path}/{f}', transpose=True)
        (_dataset, _values) = derive_dataset(data[channel])
        dataset[channel] = np.vstack((dataset[channel], _dataset))
        values[channel] = np.hstack((values[channel], _values))


def main():
    # 15min
    sets = [
        'sets/more_small/',
        'sets/more_big/',
        'sets/more_all/',
        'sets/set0_small/',
        'sets/set0_big/',
        'sets/set0_all/'
    ]

    for s in sets:
        print(s)
        set_path = s

        set_path = set_path.removesuffix('/')
        wd = f'total/{basename(set_path)}'
        if not os.path.exists(wd):
            os.mkdir(wd)

        cffs = np.zeros((3, points_len, 16))
        threads = []

        dataset0 = np.zeros((1, 16))
        values0 = np.zeros((1))
        dataset = [dataset0, dataset0, dataset0]
        values = [values0, values0, values0]

        for c in [0, 1, 2]:
            thread = threading.Thread(target=add_dataset_and_values, args=(c, set_path, dataset, values))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

        for c in [0, 1, 2]:
            derive_cffs(dataset[c][1:], values[c][1:], cffs[c])

        write_cffs_as_cpp_array(wd, cffs)
        write_cffs_as_plain_numbers(wd, wd.replace('/', '_'), cffs)

        for i, p in enumerate(points):
            _ = plt.plot(cffs[0][i], 'k', label=f'Point {p}; color 0 (Y)')
            _ = plt.plot(cffs[1][i], 'b', label=f'Point {p}; color 1 (Cb)')
            _ = plt.plot(cffs[2][i], 'r', label=f'Point {p}; color 2 (Cr)')
            _ = plt.legend()
            plt.savefig(f'{wd}/ac{p}.png')
            plt.clf()


if __name__ == '__main__':
    main()
