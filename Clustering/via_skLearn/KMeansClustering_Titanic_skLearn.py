import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
import numpy as np
from sklearn import preprocessing, model_selection
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
                    # else:
                    #     print('Cannot preprocess text data. There is a clash in names between two columns!')
                    #     sys.exit(1)

                dataframe[column] = list(map(self.convert_to_digit, dataframe[column]))

        return dataframe


df = pd.read_excel('../../training_data/titanicData.xls')
df.drop(['body', 'name', 'ticket'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
a = TitanicKMeansCluster()
df = a.handle_non_numerical_data(df)
print(df.head())




