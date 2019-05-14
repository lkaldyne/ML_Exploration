import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd
import pickle

df = pd.read_csv('..\\training_data\\breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X,y,test_size=0.2)

# clf = neighbors.KNeighborsClassifier()
# clf.fit(Xtrain,ytrain)
#
# with open('KNN_BCA.pickle','wb') as f:
#     pickle.dump(clf, f)

pickle_in = open('KNN_BCA.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(Xtest, ytest)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
print(example_measures)
#example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)
print(prediction)