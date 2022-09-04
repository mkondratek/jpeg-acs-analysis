import os
import threading
from os.path import basename

import matplotlib.pyplot as plt
import numpy as np

from commons import load_jpeg, derive_dataset, write_cffs_as_cpp_array, write_cffs_as_plain_numbers

points = [0, 1, 8, 9]

points_len = len(points)


def derive_cffs(dataset, values, cffs):
    for i, p in enumerate(points):
        args = dataset[i::points_len]
        vals = values[i::points_len]
        res = np.linalg.lstsq(args, vals, rcond=None)
        cffs[i] = res[0]


def run(jpg):
    wd = basename(jpg).replace('.', '_')
    wd = f'coeffs_2/{wd}'
    if os.path.exists(wd):
        return
    os.mkdir(wd)

    data = load_jpeg(jpg, transpose=True)
    cffs = np.zeros((3, points_len, 16))
    threads = []

    for c in [0, 1, 2]:
        def run(channel):
            (dataset, values) = derive_dataset(data[channel])
            derive_cffs(dataset, values, cffs[channel])

        thread = threading.Thread(target=run, args=(c,))
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()

    write_cffs_as_cpp_array(wd, cffs)
    write_cffs_as_plain_numbers(wd, basename(jpg), cffs)

    for i, p in enumerate(points):
        _ = plt.plot(cffs[0][i], 'k', label=f'Point {p}; color 0 (Y)')
        _ = plt.plot(cffs[1][i], 'b', label=f'Point {p}; color 1 (Cb)')
        _ = plt.plot(cffs[2][i], 'r', label=f'Point {p}; color 2 (Cr)')
        _ = plt.legend()
        plt.savefig(f'{wd}/ac{p}.png')
        plt.clf()


def main():
    for jpg in os.listdir('input'):
        print(f'Processing input/{jpg}...')
        run(f'input/{jpg}')


if __name__ == '__main__':
    main()
