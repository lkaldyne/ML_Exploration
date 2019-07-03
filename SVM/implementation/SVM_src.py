import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization = False, debug = False):
        self.data = None
        self.min_feature_value = None
        self.max_feature_value = None
        self.w = None
        self.b = None
        self.visualization = visualization
        self.debug = debug
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
                if w[0] < 0:
                    optimized = True
                    print("Optimized a step.")
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2

        if self.debug:
            for yi in self.data:
                for xi in self.data[yi]:
                    print(xi,': ',yi*(np.dot(self.w,xi) + self.b))

    def predict(self, features):
        # sign(x.w+b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x . w + b
        # v = x . w + b
        # v prospects, usually, 0, 1, -1
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w . x + b = 1)
        # Positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1) #First 'y' value for hyperplane
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)  # Second 'y' value for hyperplane
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k') #hyperplane plot

        # (w . x + b = -1)
        # Negative support vector hyperplane
        ngf1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        ngf2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [ngf1, ngf2], 'k')

        # (w . x + b = 0)
        # Decision hyperplane
        dec1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        dec2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [dec1, dec2], 'y--')

        plt.show()


data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8]]),
             1: np.array([[5,1],
                          [6,-1],
                          [7,3]])
             }
clf = Support_Vector_Machine(True, True)

clf.fit(data=data_dict)
print (clf.predict([4,7.5]))
clf.visualize()