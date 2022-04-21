import numpy as np
import matplotlib.pyplot as plt

def generate_toy_data(plot=False):
    X = [[1, 3],
        [2, 4],
        [3, 1],
        [5, 1],
        [5, 4],
        [6, 2]]
    X = np.array(X)

    y = [0, 0, 1, 1, 1, 1]
    y = np.array(y)

    if plot:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
        plt.xlim(0.0, 7.0)
        plt.ylim(0.0, 5.0)
        plt.show()
        plt.close()

    np.save('toy_X.npy', X)
    np.save('toy_y.npy', y)

    X = np.load('toy_X.npy')
    y = np.load('toy_y.npy')
    print(X.shape, y.shape)

if __name__ == '__main__':
    generate_toy_data(plot=True)