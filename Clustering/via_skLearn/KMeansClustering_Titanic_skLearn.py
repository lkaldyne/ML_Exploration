import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
import numpy as np
from sklearn import preprocessing, model_selection
import pandas as pd
style.use('ggplot')

df = pd.read_excel('../../training_data/titanicData.xls')
print(df.head)
