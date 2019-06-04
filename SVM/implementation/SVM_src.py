import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization = False):
        self.data = None
        self.min_feature_value = None
        self.max_feature_value = None
        self.w = None
        self.b = None
        self.visualization = visualization
        self.colors = {1:'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}

        transforms = [ [1,1],
                       [-1,1],
                       [-1,-1],
                       [1,-1] ]

        self.max_feature_value = -np.infty
        self.min_feature_value = np.infty

        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    if feature < self.min_feature_value:
                        self.min_feature_value = feature
                    if feature > self.max_feature_value:
                        self.max_feature_value = feature

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001
                      ]

        # extremely expensive
        b_range_multiple = 5
        # We don't need to take as small of steps with b as we do w
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(-1*self.max_feature_value*b_range_multiple,
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True

                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi * w + b) >= 1
                        for yi in self.data:
                            for xi in self.data[yi]:
                                if not yi * (np.dot(xi ,w_t) + b) >= 1:
                                    found_option = False
                                    break
                            if not found_option:
                                break

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]




    def predict(self, features):
        # sign(x.w+b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        return classification

data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8]]),
             1: np.array([[5,1],
                          [6,-1],
                          [7,3]])
             }
a = Support_Vector_Machine(True)

a.fit(data_dict)