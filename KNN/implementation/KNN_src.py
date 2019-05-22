import numpy as np
import warnings
import heapq
from sklearn import model_selection
import pandas as pd

class KNN:
    def __init__(self):
        self.features = None
        self.labels = None
        self.kVal = 0

    def fit(self, dataX, dataY, k=3):
        self.features = dataX
        self.labels = dataY
        self.kVal = k

    def accuracy(self, testFeatures, testLabels):
        accuracySum = 0
        for testIndex in range(len(testFeatures)):
            if self.predict(testFeatures[testIndex].astype(int)) == testLabels[testIndex]:
                accuracySum += 1

        return accuracySum / float(len(testFeatures))

    def predict(self,predict):
        if self.kVal % 2 == 0:
            warnings.warn("K is even! This could lead to unwanted results!")
        distances = []
        groups_track = {}
        for i in range(len(self.features)):
            heapq.heappush(distances, (np.linalg.norm(np.array(self.features[i])-np.array(predict)), self.labels[i]))
            try:
                groups_track[self.labels[i]] += 1
            except KeyError:
                groups_track[self.labels[i]] = 1
        if self.kVal < len(groups_track.keys()):
            warnings.warn('K is set to a value less than total voting groups! This could lead to unwanted results!')

        vote_track = {}
        for i in range(self.kVal):
            curMin = heapq.heappop(distances)
            try:
                vote_track[curMin[1]] += 1
            except KeyError:
                vote_track[curMin[1]] = 1

        vote_result = ['error', -9999999]
        for key in vote_track:
            curVotes = vote_track[key]
            if curVotes > vote_result[1]:
                vote_result[0] = key
                vote_result[1] = curVotes
        return vote_result[0]

df = pd.read_csv('..\\training_data\\breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X,y,test_size=0.2)

#for i in range(len(Xtrain)):
#    print(Xtrain[i].astype(int), ytrain[i])

clf = KNN()
clf.fit(Xtrain.astype(int),ytrain)
print (clf.accuracy(Xtest,ytest))

example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
print(example_measures)
#example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)
print(prediction)



#dataset = {'a': [[1, 2], [2, 3], [3, 1]], 'b': [[6, 5], [7, 7], [8, 6]]}
#new_features = [5, 7]
#print(kNN(dataset, new_features))