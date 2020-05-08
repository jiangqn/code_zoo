import numpy as np
from typing import List, Tuple
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def regression_analysis(pairs: List[tuple], labels: List[str] = None):

    """
    This is a function to do linear regression analysis.
    Input pairs: [(x_1, y_1, label_1=None), ..., (x_n, y_n, label_n=None)]
    Input labels: [xlabel, ylabel]
    """

    # add support to show chinese
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    
    # check labels
    if labels != None:
        assert len(labels) == 2
        xlabel, ylabel = labels
    
    for pair in pairs:
        
        # check pair
        assert len(pair) in (2, 3)
        if len(pair) == 2:
            x, y = pair
            assert type(x) == np.ndarray and type(y) == np.ndarray and x.shape == y.shape
            label = None
        else:
            x, y, label = pair
            assert type(x) == np.ndarray and type(y) == np.ndarray and x.shape == y.shape and type(label) == str
        
        # calculate lower_bound and upper_bound
        xp = np.asarray([[x.min()], [x.max()]])
        
        # plot scatter diagram
        plt.scatter(x, y, label=label)

        # fit with linear_regression
        x = x[:, np.newaxis]
        model = LinearRegression()
        model.fit(x, y)

        # predict on lower_bound and upper_bound
        yp = model.predict(xp)
        xp = xp[:, 0]

        # plot line diagram
        plt.plot(xp, yp)
        
        # add xlabel and ylabel
        if labels != None:
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
        
        # show labels
        if label != None:
            plt.legend()
            
    plt.show()
