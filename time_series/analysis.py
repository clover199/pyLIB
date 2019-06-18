from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import logging

"""
This file defines functions for time series analysis
"""


def calculate_summary(ts, digits=2, name=None):
    """
    Calculate some simple statistics of the input time sereise data
    Return a DataFrame with 9 columns
    input:  ts      pandas Series
            digits  number of digits to keep, default 2
            name    name of the series, default None
    """
    logger = logging.getLogger(__name__)
    df = pd.DataFrame()
    if name is None:
        name = ts.name
    if name is None:
        name = 0
    df.at[name, 'Start'] = min(ts.index)
    df.at[name, 'End'] = max(ts.index)
    n = ts.shape[0]
    mean = np.mean(ts)
    df.at[name, 'Size'] = n
    df.at[name, 'Mean'] = np.round(mean, digits)
    df.at[name, 'Std'] = np.round(np.sqrt( np.mean((ts-mean)**2) * n / (n-1) ), digits)
    df.at[name, 'Skew'] = np.round( np.mean((ts-mean)**3) / np.mean((ts-mean)**2)**1.5, digits)
    df.at[name, 'Kurtosis'] = np.round( np.mean((ts-mean)**4) / np.mean((ts-mean)**2)**2 - 3, digits)
    data = np.sort(ts.values).flatten()
    df.at[name, 'min'] = data[0]
    for p in [0.25, 0.5, 0.75]:
        i = int(n*p)
        ratio = np.abs(n*p - i - p)
        df.at[name, "{:.0%}".format(p)] = ratio * data[i-1] + (1-ratio) * data[i]
    df.at[name, 'max'] = data[n-1]
    df = df.astype({'Size':int})
    return df


def auto_correlation(data, lag=10, plotit=False, alpha=0.95):
    """
    Calculate sample auto correlation for given data
    input:  data    Series or numpy array of data
            lag     positive integer, default 10, all lags smaller than the value is calculated
            plotit  indicate whether to make plots or not, default False
            alpha   a number from 0 to 1, confidence level, default 0.95
    return a numpy array
    """
    logger = logging.getLogger(__name__)
    try:
        data = data.dropna().values
    except:
        pass
    mean = np.mean(data)
    var = np.sum( (data-mean)**2 )
    ac = np.ones(lag+1)
    for i in range(1, lag+1):
        ac[i] = np.sum( (data[i:]-mean) * (data[:-i]-mean) ) / var
    if plotit:
        import matplotlib.pyplot as plt
        from scipy import stats
        plt.stem(ac)
        std = np.zeros(ac.shape)
        for i in range(1, std.shape[0]):
            std[i] = std[i-1] + ac[i-1]**2 / len(data)
        std = np.sqrt(std)[1:]
        sig = -stats.t.ppf(0.5-alpha/2, len(data))
        plt.fill_between(range(1,lag+1), sig*std, -sig*std, color='grey', alpha='0.5')
        plt.xlim(0,lag)
    return ac


def partial_auto_correlation(data, lag=10, plotit=False, alpha=0.95):
    """
    Calculate sample partial auto correlation for given data
    Week stationary and zero mean assumption is applied to speed up the calculation
    input:  data    Series numpy array of data
            lag     positive integer, all lags smaller than the value is calculated
                    default 10
            plotit  indicate whether to make plots or not, default False
            alpha   a number from 0 to 1, confidence level, default 0.95
    return a list of numbers
    """
    logger = logging.getLogger(__name__)
    from functions import inverse, block_inverse
    try:
        data = data.dropna().values
    except:
        pass

    ac = auto_correlation(data, lag=lag)
    XX = np.zeros([lag,lag])
    for i in range(lag):
        XX[i,i:] = ac[:lag-i]
    for i in range(1,lag):
        XX[i,:i] = ac[1:1+i][::-1]
    XY = ac[1:]

    pac = np.ones(lag+1)
    pac[1] = ac[1]
    inv = np.array([[1]])
    std = np.zeros(lag) + np.sqrt( (1-XY[0]*pac[1]) / (len(data)-2))
    YY = np.mean((data-np.mean(data))**2)
    for i in range(1,lag):
        inv = block_inverse(inv, XX[:i,i], XX[i,:i], XX[i,i])
        beta = inv.dot(XY[:i+1])
        pac[i+1] = beta[-1]
        std[i] = np.sqrt( inv[-1,-1] * (1-XY[:i+1].dot(beta)) / (len(data)-2*i-2))
    if plotit:
        import matplotlib.pyplot as plt
        from scipy import stats
        plt.stem(pac)
        sig = -stats.t.ppf(0.5-alpha/2, len(data))
        plt.fill_between(range(1,lag+1), sig*std, -sig*std, color='grey', alpha='0.5')
        plt.xlim(0,lag)
    return pac


def partial_auto_correlation2(data, lag=10, plotit=False, alpha=0.95):
    """
    Calculate sample partial auto correlation for given data
    input:  data    Series or numpy array of data
            lag     positive integer, all lags smaller than the value is calculated
                    default 10
            plotit  indicate whether to make plots or not, default False
            alpha   a number from 0 to 1, confidence level, default 0.95
    return a list of numbers
    """
    logger = logging.getLogger(__name__)
    try:
        data = data.dropna().values
    except:
        pass

    from Msklearn import LinearRegression
    from scipy import stats
    pac = np.zeros(lag+1)
    pac[0] = np.mean(data)
    lm = LinearRegression()
    X = np.array(data[:-1]).reshape([-1,1])
    std = np.zeros(lag)
    for i in range(1,lag+1):
        lm.fit(X, data[i:])
        pac[i] = lm.beta[-1]
        std[i-1] = np.sqrt(lm.var_beta[-1,-1]) * stats.t.ppf(0.5-alpha/2, len(data)-2*i-1)
        X = np.concatenate([X[1:,:], np.array(data[:-i-1]).reshape([-1,1])], axis=1)
    if plotit:
        import matplotlib.pyplot as plt
        plt.stem(pac)
        plt.fill_between(range(1,lag+1), std, -std, color='grey', alpha='0.5')
        plt.xlim(0,lag)
    return pac


if __name__=="__main__":
    import argparse
    import analysis
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Basic time series analysis")
    parser.add_argument('--list', type=bool, nargs='?', const=1, default=0, \
                        help='list all available modules')
    parser.add_argument('--doc', type=str, \
                        help='print documents of given function')
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.list:
        for x in getmembers(analysis, isfunction):
            print(x[0])
        exit()
    if not FLAGS.doc is None:
        print(getattr(analysis, FLAGS.doc).__doc__)
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))
