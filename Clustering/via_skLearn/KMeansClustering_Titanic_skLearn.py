import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
import numpy as np
from sklearn import preprocessing
import pandas as pd
import sys
style.use('ggplot')


class TitanicKMeansCluster:
    def __init__(self):
        self.text_to_digit_map = {}

    def convert_to_digit(self, textval):
        return self.text_to_digit_map[textval]

    def handle_non_numerical_data(self, dataframe):
        columns = dataframe.columns.values

        for column in columns:
            if dataframe[column].dtype not in [np.int64, np.float64]:
                unique_column_entries = set(dataframe[column].values.tolist())

                incrementor = 0
                for entry in unique_column_entries:
                    if entry not in self.text_to_digit_map:
                        self.text_to_digit_map[entry] = incrementor
                        incrementor += 1

                dataframe[column] = list(map(self.convert_to_digit, dataframe[column]))

        return dataframe


df = pd.read_excel('../../training_data/titanicData.xls')
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
a = TitanicKMeansCluster()
df = a.handle_non_numerical_data(df)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array((df['survived']))

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predictEntry = np.array(X[i].astype(float)).reshape(-1, len(X[i]))
    prediction = clf.predict(predictEntry)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))





