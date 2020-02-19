"""
Author      : Arjun Natarjan, Daniel Sealand
Class       : HMC CS 158
Date        : 18 Feb 2020
Description : Perceptron
"""

# This code was adapted course material by Tommi Jaakola (MIT).

# utilities
import util

# numpy libraries
import numpy as np

# scikit-learn libraries
from sklearn.svm import SVC

# matplotlib libraries
import matplotlib.pyplot as plt

######################################################################
# functions
######################################################################

def load_simple_dataset() :
    """Simple dataset of four points."""
    
    #  dataset
    #     i    x^{(i)}        y^{(i)}
    #     1    ( 1,    1)^T   -1
    #     2    ( 0.5, -1)^T    1
    #     3    (-1,   -1)^T    1
    #     4    (-1,    1)^T    1
    #   if outlier is set, x^{(3)} = (12, 1)^T
    
    # data set
    data = util.Data()
    data.X = np.array([[ 1,    1],
                       [ 0.5, -1],
                       [-1,   -1],
                       [-1,    1]])
    data.y = np.array([-1, 1, 1, 1])
    return data


def plot_perceptron(data, clf, plot_data=True, axes_equal=False, **kwargs) :
    """Plot decision boundary and data."""
    assert isinstance(clf, Perceptron)
    
    # plot options
    if "linewidths" not in kwargs :
        kwargs["linewidths"] = 2
    if "colors" not in kwargs :
        kwargs["colors"] = 'k'
    label = None
    if "label" in kwargs :
        label = kwargs["label"]
        del kwargs["label"]
    
    # plot data
    if plot_data : data.plot()
    
    # hack to skip theta of all zeros
    if len(np.nonzero(clf.coef_)) == 0: return
    
    # axes limits and properties
    xmin, xmax = data.X[:, 0].min() - 1, data.X[:, 0].max() + 1
    ymin, ymax = data.X[:, 1].min() - 1, data.X[:, 1].max() + 1
    if axes_equal :
        xmin = ymin = min(xmin, ymin)
        xmax = ymax = max(xmax, ymax)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
    
    # create a mesh to plot in
    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h),
                         np.arange(ymin, ymax, h))
    
    # plot decision boundary
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    zz = z.reshape(xx.shape)
    cs = plt.contour(xx, yy, zz, [0], **kwargs)
    
    # legend
    if label :
        cs.collections[0].set_label(label)
        plt.legend()


######################################################################
# classes
######################################################################

class Perceptron :
    
    def __init__(self) :
        """
        Perceptron classifier that keeps track of mistakes made on each data point.
        
        Attributes
        --------------------
            coef_     -- numpy array of shape (d,), feature weights
            mistakes_ -- numpy array of shape (n,), mistakes per data point
        """
        self.coef_ = None
        self.mistakes_ = None
    
    def fit(self, X, y, coef_init=None,
            verbose=False, plot=False) :
        """
        Fit the perceptron using the input data.
        
        Parameters
        --------------------
            X         -- numpy array of shape (n,d), features
            y         -- numpy array of shape (n,), targets
            coef_init -- numpy array of shape (d,), initial feature weights
            verbose   -- boolean, for debugging purposes
            plot      -- boolean, for debugging purposes
        
        Returns
        --------------------
            self      -- an instance of self
        """
        # get dimensions of data
        n,d = X.shape
        
        # initialize weight vector to all zeros
        if coef_init is None :
            self.coef_ = np.zeros(d)
        else :
            self.coef_ = coef_init
        
        # record number of mistakes we make on each data point
        self.mistakes_ = np.zeros(n)
        
        # debugging
        if verbose :
            print(f'\ttheta^{{(0)}} = {self.coef_}')
        if plot :
            # set up colors
            colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
            cndx = 1
            
            # plot
            data = util.Data(X, y)
            plot_perceptron(data, self, plot_data=True, axes_equal=True,
                            colors=colors[0],
                            label=r"$\theta^{(0)}$")
            
            # pause
            plt.gca().annotate('press key to continue\npress mouse to quit',
                               xy=(0.99,0.01), xycoords='figure fraction', ha='right')
            plt.draw()
            keypress = plt.waitforbuttonpress(0) # True if key, False if mouse
            if not keypress :
                plot = False
        
        # part a: implement perceptron algorithm
        # cycle until all examples are correctly classified
        # do NOT shuffle examples on each iteration
        # on a mistake, be sure to update self.mistakes_
        # professor's solution: 10 lines

        # update weights
        incorrectlyClassified = True
        while incorrectlyClassified:
            # checks if an example is classified incorrectly and if so, updates the weights
            oneIncorrect = False
            for i in range(n):
                if y[i] * np.dot(self.coef_, X[i]) <= 0:
                    self.coef_ = self.coef_ + y[i] * X[i]
                    oneIncorrect = True
                    self.mistakes_[i] += 1

                    # indent the following debugging code to execute every time you update
                    # you can include code both before and after this block
                    mistakes = int(sum(self.mistakes_))
                    if verbose :
                        print(f'\ttheta^{{({mistakes:d})}} = {self.coef_}')
                    if plot :
                        plot_perceptron(data, self, plot_data=False, axes_equal=True,
                                        colors=colors[cndx],
                                        label=rf"$\theta^{{({mistakes:d})}}$")
                        
                        # set next color
                        cndx += 1
                        if cndx == len(colors) :
                            cndx = 0
                        
                        # pause
                        plt.draw()
                        keypress = plt.waitforbuttonpress(0) # True if key, False if mouse
                        if not keypress :
                            plot = False

            # if all examples classified correctly, stop                    
            if not oneIncorrect:
                incorrectlyClassified = False
        
        return self
    
    def predict(self, X) :
        """
        Predict labels using perceptron.
        
        Parameters
        --------------------
            X         -- numpy array of shape (n,d), features
        
        Returns
        --------------------
            y_pred    -- numpy array of shape (n,), predictions
        """
        return np.sign(np.dot(X, self.coef_))


######################################################################
# main
######################################################################

def main() :
    
    #========================================
    # test part a
    
    # simple data set (from class)
    # coef = [ -1.5, -1], mistakes = 7
    data = load_simple_dataset()
    clf = Perceptron()
    clf.fit(data.X, data.y, coef_init=np.array([1,0]),
            verbose=True, plot=True)
    print(f'simple data\n\tcoef = {clf.coef_}, mistakes = {int(sum(clf.mistakes_)):d}')
    
    #========================================
    # perceptron data set
    
    train_data = util.load_data("perceptron_data.csv")
    
    # part b: compare different initializations
    # professor's solution: 4 lines

    clf.fit(train_data.X, train_data.y, coef_init=np.array([0,0]), verbose=True, plot=True)
    print(f'train_data, init [0,0]\n\tcoef = {clf.coef_}, mistakes = {int(sum(clf.mistakes_)):d}')
    clf.fit(train_data.X, train_data.y, coef_init=np.array([1,0]), verbose=True, plot=True)
    print(f'train_data, init [1,0]\n\tcoef = {clf.coef_}, mistakes = {int(sum(clf.mistakes_)):d}')
        



    print('perceptron bound')
    
    # you do not have to understand this code -- we will cover it when we discuss SVMs
    # compute gamma^2 using hard-margin SVM (SVM with large C)
    clf = SVC(kernel='linear', C=1e10)
    clf.fit(train_data.X, train_data.y)
    gamma = 1./np.linalg.norm(clf.coef_, 2)
    
    # part c: compare perceptron bound to number of mistakes
    # professor's solution: 4 lines
    
    n,d = np.shape(train_data.X)
    # compute R^2
    maxR = 0
    for i in range(n):
        maxR = max(np.linalg.norm(train_data.X[i]), maxR)
    
    # compute perceptron bound (R / gamma)^2
    bound = (maxR / gamma)**2
    print(bound)

if __name__ == "__main__" :
    main()