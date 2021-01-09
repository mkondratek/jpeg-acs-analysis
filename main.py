import numpy as np
from jpeg import parse

# todo: all colors
c = 0


def print_acs(data):
    height = data.shape[1]
    width = data.shape[2]
    for y in range(height):
        for x in range(width):
            print(f"(c={c}, bx={x}, by={y}) block values:")
            for j in range(8):
                for i in range(8):
                    print(f"{data[c][y][x][j][i]:8}", end='')
                print()


def load_data(fname):
    arr = parse(fname, normalize=True, quality=100, subsampling='keep', upsample=True, stack=True)
    height = arr.shape[1]
    width = arr.shape[2]
    data = np.zeros([3, height, width, 8, 8], dtype=int)

    for y in range(height):
        for x in range(width):
            print(f"(c={c}, bx={x}, by={y}) block - transposing...")
            data[c][y][x] = np.transpose(np.reshape(arr[c][y][x], (8, 8)))

    return data


def derive_dataset(data):
    height = data.shape[1]
    width = data.shape[2]

    dataset = np.zeros((3, height, width, 8, 8, 17), dtype=int)

    for y in range(1, height):
        for x in range(1, width):
            for j in range(8):
                for i in range(8):
                    left_acs = data[c][y][x - 1][j]
                    top_acs = data[c][y - 1][x][:, i]
                    dataset[c, y, x, j, i] = np.concatenate((left_acs, top_acs, data[c, y, x, j, i]), axis=None)
                    print(f"x={x}, y={y}, j={j}, i={i}, ds={dataset[c, y, x, j, i]}")
    return data


if __name__ == '__main__':
    data = load_data('some_2.jpg')

    derive_dataset(data)

    # print_acs()
