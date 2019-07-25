import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]
              ])

plt.scatter(X[:, 0], X[:, 1], s=150)
plt.show()

colors = 10*["g.", "r.", "c.", "b.", "k."]


class KMeans:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = {}

    def fit(self, data):
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            classifications = {}
            for j in range(self.k):
                classifications[j] = []

    def predict(self, data):
        pass


a = 5
