import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
from sklearn import preprocessing
from Clustering.via_skLearn.KMeansClustering_Titanic_skLearn import TitanicPreProcessing
style.use('ggplot')


class KMeans:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = {}
        self.classifications = {}

    def fit(self, data):
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            # {centroid: [featuresets closest to that centroid]}
            self.classifications = {}
            for j in range(self.k):
                self.classifications[j] = []

            for featureset in data:
                # Array containing the distances between this coordinate and each centroid
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                # Relocate this iteration's centroid to the average of its featureset pool
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for cKey in self.centroids:
                prev_centroid = prev_centroids[cKey]
                current_centroid = self.centroids[cKey]
                if abs(np.sum(current_centroid - prev_centroid/prev_centroid*100.0)) > self.tol:
                    optimized = False
                    break

            if optimized:
                break

    def predict(self, data):
        min_centroid: tuple = (0, np.infty)
        for cKey in self.centroids:
            curr_distance = abs(np.linalg.norm(data - self.centroids[cKey]))
            if curr_distance < min_centroid[1]:
                min_centroid = (cKey, curr_distance)
        return min_centroid[0]


# X = np.array([[1, 2],
#               [1.5, 1.8],
#               [5, 8],
#               [8, 8],
#               [1, 0.6],
#               [9, 11]
#               ])
#
# plt.scatter(X[:, 0], X[:, 1], s=150)
#
# colors = 10*["g", "r", "c", "b", "k"]
#
# clf = KMeans()
# clf.fit(X)
#
# for centroid in clf.centroids:
#     plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o", color='k',
#                 s=150, linewidths=5)
#
# for csf in clf.classifications:
#     color = colors[csf]
#     for fst in clf.classifications[csf]:
#         plt.scatter(fst[0], fst[1], marker='x', color=color, s=150, linewidths=5)
#
# unknowns = np.array([[1, 3],
#                      [8, 9],
#                      [0, 3],
#                      [5, 4],
#                      [6, 4]])
#
# for unknown in unknowns:
#     csf = clf.predict(unknown)
#     plt.scatter(unknown[0], unknown[1], marker="*", color=colors[csf], s=150, linewidths=5)
# plt.show()

# Testing on titanic data --------------------------------------------

df = pd.read_excel('../../training_data/titanicData.xls')
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
a = TitanicPreProcessing()
df = a.handle_non_numerical_data(df)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array((df['survived']))

clf = KMeans()
clf.fit(X)

correct = 0
for i in range(len(X)):
    predictEntry = np.array(X[i].astype(float)).reshape(-1, len(X[i]))
    prediction = clf.predict(predictEntry)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))
