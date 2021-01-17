import matplotlib.pyplot as plt
import numpy as np
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

    with Bar('Deriving dataset...', max=3 * (height - 1) * (width - 1) * 64) as bar:
        for c in range(3):
            for y in range(1, height):
                for x in range(1, width):
                    for j in range(8):
                        left_acs = data[c, y, x - 1, j]
                        for i in range(8):
                            top_acs = data[c, y - 1, x, :, i]
                            dataset[c, y - 1, x - 1, j, i] = np.concatenate((left_acs, top_acs), axis=None)
                            bar.next()
    m = (height - 1) * (width - 1) * 8 * 8
    return (dataset.reshape(3, m, 16), data[:, 1:, 1:].reshape(3, m))


points = [0, 1, 8, 9]

points_len = len(points)


def derive_cffs(dataset, values):
    cffs_arr = np.zeros((3 * points_len, 16))

    with Bar('Deriving least-squares solutions...', max=3 * points_len) as bar:
        for c in range(3):
            for i in range(points_len):
                p = points[i]
                args = dataset[c][p::64]
                vals = values[c][p::64]
                res = np.linalg.lstsq(args, vals)
                cffs_arr[c * points_len + i] = res[0]
                bar.next()

    return cffs_arr


def print_cffs_as_cpp_array(cffs_arr):
    print('const float coeffs[3][4][16] = {')
    for c in range(3):
        print('\t{')
        for i in range(points_len):
            cffs_str = ', '.join(map(lambda x: f'{x:.9f}', cffs_arr[c * points_len + i]))
            print(f'\t\t{ {cffs_str} },'.replace('\'', ''))
        print('\t},')
    print('};')


if __name__ == '__main__':
    data = load_data('sunset.jpg', transpose=True)
    # print_acs(data)

    (dataset, values) = derive_dataset(data)

    cffs = derive_cffs(dataset, values)
    print_cffs_as_cpp_array(cffs)

    for i in range(points_len):
        p = points[i]
        _ = plt.plot(cffs[i], 'k', label=f'Point {p}; color 0 (Y)')
        _ = plt.plot(cffs[i + points_len], 'b', label=f'Point {p}; color 1 (Cb)')
        _ = plt.plot(cffs[i + 2 * points_len], 'r', label=f'Point {p}; color 2 (Cr)')
        _ = plt.legend()
        plt.show()
