import numpy as np
from sklearn.cross_decomposition import CCA

from commons import load_data


def perform(arrs):
    blocks_cnt = sum([arr.shape[1] * arr.shape[2] for arr in arrs])
    X = np.zeros((blocks_cnt, 16))
    Y = np.zeros((blocks_cnt, 64))
    for c in range(3):
        for i, arr in enumerate(arrs):
            height = arr.shape[1]
            width = arr.shape[2]
            for y in range(height):
                for x in range(width):
                    X[y * width + x] = np.hstack([arr[c][y][x - 1][:][-1], arr[c][y - 1][x][-1][:]])
                    Y[y * width + x] = arr[c][y][x].ravel()

        X_mc = (X - X.mean()) / (X.std())
        Y_mc = (Y - Y.mean()) / (Y.std())

        ca = CCA(n_components=1)
        ca.fit(X_mc, Y_mc)

        print(f'\nColor {c}:')
        weights = ca.x_weights_.ravel()
        print(weights.shape)
        print(', '.join(map(lambda a: str(a), weights)))
        print(ca.n_iter_)


if __name__ == '__main__':
    images = [
        # 'other.jpg',
        # 'sunset.jpg',
        'StockSnap_7QH4K6AESO.jpg'
    ]
    data = [load_data(img, transpose=True) for img in images]
    perform(data)
