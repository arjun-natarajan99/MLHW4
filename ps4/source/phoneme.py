"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2020 Feb 11
Description : Perceptron vs Logistic Regression on a Phoneme Dataset
"""

# utilities
import util

# numpy libraries
import numpy as np

# scipy libraries
from scipy import stats

# scikit-learn libraries
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Perceptron, LogisticRegression

# matplotlib libraries
import matplotlib.pyplot as plt

######################################################################
# functions
######################################################################

def get_classifier(clf_str) :
    """
    Initialize and return a classifier represented by the given string.
    
    Parameters
    --------------------
        clf_str     -- string, classifier name
                       implemented
                           "dummy"
                           "perceptron"
                           "logistic regression"
    
    Returns
    --------------------
        clf         -- scikit-learn classifier
        param_grid  -- dict, hyperparameter grid
                           key = string, name of hyperparameter
                           value = list, parameter settings to search
    """
    
    if clf_str == "dummy" :
        clf = DummyClassifier(strategy='stratified')
        param_grid = {}
    elif clf_str == "perceptron" :
        ### ========== TODO : START ========== ###
        # part b: modify two lines below to set parameters for perceptron
        # classifier parameters
        #   estimate intercept, use L2-regularization, set max iterations of 10k
        # parameter search space
        #   find the parameter for tuning regularization strength
        #   let search values be [1e-5, 1e-4, ..., 1e5] (hint: use np.logspace)
        
        clf = Perceptron()
        param_grid = {}
        ### ========== END : START ========== ###
    elif clf_str == "logistic regression" :
        ### ========== TODO : START ========== ###
        # part b: modify two lines below to set parameters for logistic regression
        # classifier parameters
        #     estimate intercept, use L2-regularization and lbfgs solver, set max iterations of 10k
        # parameter search space
        #    find the parameter for tuning regularization strength
        #    let search values be [1e-5, 1e-4, ..., 1e5] (hint: use np.logspace)
        
        clf = LogisticRegression()
        param_grid = {}
        ### ========== END : START ========== ###
    
    return clf, param_grid


def get_performance(clf, param_grid, X, y, ntrials=100) :
    """
    Estimate performance used nested 5x2 cross-validation.
    
    Parameters
    --------------------
        clf             -- scikit-learn classifier
        param_grid      -- dict, hyperparameter grid
                               key = string, name of hyperparameter
                               value = list, parameter settings to search
        X               -- numpy array of shape (n,d), features values
        y               -- numpy array of shape (n,), target classes
        ntrials         -- integer, number of trials
        
    Returns
    --------------------
        train_scores    -- numpy array of shape (ntrials,), average train scores across cv splits
                           scores computed via clf.score(X,y), which measures accuracy
        test_scores     -- numpy array of shape (ntrials,), average test scores across cv splits
                           scores computed via clf.score(X,y), which measures accuracy
    """
    
    train_scores = np.zeros(ntrials)
    test_scores = np.zeros(ntrials)
    
    ### ========== TODO : START ========== ###
    # part c: compute average performance using 5x2 cross-validation
    # hint: use StratifiedKFold, GridSearchCV, and cross_validate
    #       set idd=False for GridSearchCV to prevent DeprecationWarnings
    # professor's solution: 6 lines
    
    
    
    ### ========== TODO : END ========== ###
    
    return train_scores, test_scores


def plot(train_scores, test_scores, clf_strs) :
    """Plot performance."""
    
    labels = ["training", "testing"]
    ind = np.arange(len(labels))    # x locations for groups
    width = 1 / (len(clf_strs) + 1) # width of the bars
    
    # text annotation
    def autolabel(rects) :
        """Attach a text label above each bar displaying its height"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}", xy=(rect.get_x() + rect.get_width() / 2., height),
                        xytext=(0, 3), textcoords='offset points', # 3 points vertical offset
                        ha='center', va='bottom')
    
    # bar plot with error bars
    fig, ax = plt.subplots()
    for i, clf_str in enumerate(clf_strs) :
        means = (train_scores[clf_str].mean(), test_scores[clf_str].mean())
        stds = (train_scores[clf_str].std(), test_scores[clf_str].std())
        
        rects = ax.bar(ind + width * i, means, width, yerr=stds, label=clf_str)
        autolabel(rects)
    
    # axes
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)
    ax.set_title('Nested Cross-Validation Performance')
    ax.set_xticks(ind + width * (len(clf_strs) - 1) / 2.)
    ax.set_xticklabels(labels)
    ax.legend(loc=4)    # lower right
    fig.tight_layout()
    
    plt.show()


######################################################################
# main
######################################################################

def main() :
    np.random.seed(1234)
    
    # load data
    phoneme = util.load_data("phoneme_train.csv")
    X, y = phoneme.X, phoneme.y
    
    ### ========== TODO : START ========== ###
    # part a: is data linearly separable?
    # hints: be sure to set parameters for Perceptron
    #        an easy parameter to miss is tol=None, a much stricter stopping criterion than default
    # professor's solution: 5 lines
    
    
    
    ### ========== TODO : END ========== ###
    
    print()
    
    #========================================
    # part d: compare classifiers
    
    # setup
    ntrials = 10
    train_scores_all = {}
    test_scores_all = {}
    
    clf_strs = ["dummy", "perceptron", "logistic regression"]
    
    # nested CV to estimate performance
    for clf_str in clf_strs :
        clf, param_grid = get_classifier(clf_str)
        train_scores, test_scores = get_performance(clf, param_grid, X, y, ntrials)
        train_scores_all[clf_str], test_scores_all[clf_str] = train_scores, test_scores
        
        print(f"{clf_str}")
        print(f"\ttraining accuracy: {np.mean(train_scores):.3g} +/- {np.std(train_scores):.3g}")
        print(f"\ttest accuracy:     {np.mean(test_scores):.3g} +/- {np.std(test_scores):.3g}")
    print()
    
    # plot
    plot(train_scores_all, test_scores_all, clf_strs)
    
    ### ========== TODO : START ========== ###
    # part e: compute significance using t-test
    # professor's solution: 7 lines
    
    print("significance tests")
    
    
    
    ### ========== TODO : END ========== ###

if __name__ == "__main__" :
    main()