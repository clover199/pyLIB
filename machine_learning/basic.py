from __future__ import absolute_import, division, print_function
import numpy as np
import logging

"""
This file defines some useful functions for machine learning problems.
"""


def covariance(data):
    """
    Calculate covariance of features
    input:  data    2D array of data with rows as samples and columns as features
    returns         a covariance matrix
    """
    logger = logging.getLogger(__name__)
    logger.info("------ covariance(data) ------")
    n = data.shape[0]
    return np.mean((data-np.mean(data, axis=0))**2) * n / (n-1)


def RMSE(y, y_hat):
    """
    Root Mean Square Error
    input:  y       true values
            y_hat   predicted values
    y and y_hat should be of the same shape or one is a number
    """
    logger = logging.getLogger(__name__)
    logger.info("------ RMSE(y, y_hat) -------")
    return np.sqrt(np.mean((y-y_hat)**2))


def accuracy(y, y_hat):
    """
    Accuracy, the percentage of correct predictions
    input:  y       true values
            y_hat   predicted values
    y and y_hat should be of the same shape or one is a number
    """
    logger = logging.getLogger(__name__)
    logger.info("------ accuracy(y, y_hat) ------")
    return np.mean(y==y_hat)


def cross_entropy(y, y_hat, tol=1e-10):
    """
    Cross entropy E_y[log y_hat]
    input:  y       non-negative 2D array, true probability
            y_hat   non-negative 2D array, predicted probability
            tol     tolerance, default 1e-10
    y and y_hat should be of the same shape
    row sum of both y and y_hat should be 1
    """
    logger = logging.getLogger(__name__)
    logger.info("------ cross_entropy(y, y_hat, tol=1e-10) ------")
    return -np.mean(np.sum(y*np.log(y_hat+tol), axis=1))


def soft_max(x):
    """
    Calculate soft max probability
    input:  x   2D array of data with rows as samples
    returns     2D array of probabilities with row sum as one
    """
    logger = logging.getLogger(__name__)
    logger.info("------ soft_max(x) ------")
    sm = np.exp(x-np.max(x, axis=1, keepdims=True))
    sm = sm/np.sum(sm, axis=1, keepdims=True)
    return sm


def label_to_proba(label, p=None):
    """
    Convert label array to probability matrix
    input:  label   1D array of labels. Labels should be consecutive integers starting from 0
            p       number of classes, default None.
                    If None, calculate from label
    returns         2D array of probabilities with row sum as one
    """
    logger = logging.getLogger(__name__)
    logger.info("------ label_to_proba(label, p=None) ------")
    if label.ndim!=1:
        raise ValueError("input of labels should be 1d array")
    levels = len(np.unique(label))
    if p is not None:
        if p<levels:
            logger.warning("Warning: more levels found {:d}>{:d}".format(levels, p))
        else:
            levels = p
    n = label.shape[0]
    proba = np.zeros([n,levels])
    proba[np.arange(n), label] = 1
    return proba


def proba_to_label(proba):
    """
    Convert probability matrix to label array
    input:  proba   2D array of probabilities
    returns         1D array of labels. Labels are consecutive integers starting from 0
    """
    logger = logging.getLogger(__name__)
    logger.info("------ proba_to_label(proba) ------")
    return np.argmax(proba, axis=1).ravel()


def generate_dummy(data, names=None):
    """
    create dummy variables for input series data
    input:  data    pandas Series
            names   a list of unique names, default None
                    If None, use all unique values in data
    returns         pandas DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.info("------ generate_dummy(data, names=None) ------")
    n = data.shape[0]
    if names is None:
        names = np.unique(data)
    names = list(names)
    dummies = np.zeros([n, len(names)], dtype=int)
    dummies[range(n), data.map(lambda x: names.index(x))] = 1
    return pd.DataFrame(dummies, columns=names)


def select_feature(data, y, tol=0, min_bin=0.05, regress=True):
    """
    Use random forest to select features from data set with both numerical and
    categorical features.
    input:  data    2D array of the independent variables
            y       1D array of the respons variable
            tol     the smallest feature importance to keep, default 0
            min_bin the minimum percentage of one class in categorical feature,
                    default 0.05
            regress indicate whether the problem is regression or classification
    """
    logger = logging.getLogger(__name__)
    logger.info("------ select_feature(data, y, tol=0, min_bin=0.05, regress=True) ------")
    from time import time
    begin = time()
    cat_cols = data.columns[data.dtypes=='object']
    num_cols = data.columns[data.dtypes!='object']
    logger.info("{} numerical and {} categorical features out of {} features".format(
        len(num_cols), len(cat_cols), data.shape[1]))
    results = []  # score, feature name, feature importance

    # for numerical variables
    beg = time()
    if regress:
        from sklearn.ensemble import RandomForestRegressor as RandomForest
    else:
        from sklearn.ensemble import RandomForestClassifier as RandomForest
    rf = RandomForest(n_estimators=500,
                      max_depth=int(np.log(len(num_cols))/np.log(2))+1,
                      max_features=0.33)
    logger.info("max_depth", int(np.log(len(num_cols))/np.log(2))+1)
    rf = rf.fit(data[num_cols], y)
    print("Model with numerical variables fitted, time used {:.2f} min".format((time()-beg)/60))
    beg = time()
    score = rf.score(data[num_cols], y)
    results.append([score, num_cols, rf.feature_importances_])
    selected = num_cols[rf.feature_importances_ > tol]
    print("Model score {:.2f}, time used {:.2f} min".format(score, (time()-beg)/60))

    # for categorical variables
    dummies = pd.DataFrame()
    print("-"*len(cat_cols), end='\r', flush=True)
    for i, col in enumerate(cat_cols):
        beg = time()
        c, d = np.unique(data[col], return_counts=True)
        d = d/sum(d)
        for cls in c[d>min_bin]:
            dummies[str(col)+'.'+str(cls)] = (data[col]==cls).astype(int)
        rf = RandomForest(n_estimators=500,
                          max_depth=int(np.log(len(selected)+dummies.shape[1])/np.log(2))+1,
                          max_features=0.33)
        temp = pd.concat([data[selected], dummies], axis=1)
        logger.info("max_depth",int(np.log(len(selected)+dummies.shape[1])/np.log(2))+1, temp.shape[1])
        rf = rf.fit(temp, y)
        score = rf.score(temp, y)
        results.append([score, temp.columns, rf.feature_importances_])
        selected = selected[rf.feature_importances_[:len(selected)] > tol]
        dummies.drop(dummies.columns[rf.feature_importances_[len(selected):] < tol], axis=1, inplace=True)
        print("*"*(i+1) + "-"*(len(cat_cols)-i-1) + " score {:.2f}, time {:.2f} min".format(score, (time()-beg)/60),
              end='\n', flush=True)
    print("\ntotal time used: {:.2f} min".format((time()-begin)/60))
    return results


if __name__=="__main__":
    import argparse
    import basic
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Basic functions for machine learning")
    parser.add_argument('--list', type=bool, nargs='?', const=1, default=0, \
                        help='list all available modules')
    parser.add_argument('--doc', type=str, \
                        help='print documents of given function')
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.list:
        for x in getmembers(basic, isfunction):
            print(x[0])
        exit()
    if not FLAGS.doc is None:
        print(getattr(basic, FLAGS.doc).__doc__)
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))
