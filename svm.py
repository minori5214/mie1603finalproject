from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

class SVM():
    def __init__(self, kernel='linear'):
        self.clf = svm.SVC(kernel=kernel)
    
    def fit(self, X, y):
        self.clf.fit(X, y)
    
    def predict(self, X):
        return self.clf.predict(X)

if __name__ == '__main__':
    plot = True

    X = np.load('toy_X.npy')
    y = np.load('toy_y.npy')

    svm = SVM()
    svm.fit(X, y)
    y_pred = svm.predict(X)

    if plot:
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='rainbow')
        plt.xlim(0.0, 7.0)
        plt.ylim(0.0, 5.0)
        plt.show()
        plt.close()