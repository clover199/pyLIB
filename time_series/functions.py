from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import logging

"""
This file defines functions to manipulate time series data
"""


def prev_month(dt):
    """
    Change the month of given date. If day is out of range,
    make it the last day of that month.
    """
    from datetime import date, timedelta
    temp = date(dt.year, dt.month, 1) - timedelta(days=1)
    try:
        return date(temp.year, temp.month, dt.day)
    except:
        return date(dt.year, dt.month, 1) - timedelta(days=1)


def next_month(dt):
    """
    Change the month of given date. If day is out of range,
    make it the last day of that month.
    """
    from datetime import date, timedelta
    temp = date(dt.year, dt.month, 15) + timedelta(days=30)
    try:
        return date(temp.year, temp.month, dt.day)
    except:
        temp = dt + timedelta(days=45)
        return date(temp.year, temp.month, 1) - timedelta(days=1)


def change_month(dt, shift=0):
    """
    Change the month of given date. If day is out of range,
    make it the last day of that month.
    """
    from datetime import date, timedelta
    if shift==0:
        return dt
    elif shift<0:
        temp = dt
        for _ in range(-shift):
            temp = prev_month(temp)
        try:
            return date(temp.year, temp.month, dt.day)
        except:
            temp = next_month(temp)
            return date(temp.year, temp.month, 1) - timedelta(days=1)
    else: # shift>0
        temp = dt
        for _ in range(shift):
            temp = next_month(temp)
        try:
            return date(temp.year, temp.month, dt.day)
        except:
            temp = next_month(temp)
            return date(temp.year, temp.month, 1) - timedelta(days=1)


def moving_agg(data, window=2, step=1, func=np.mean):
    """
    Calculate moving aggregate for the given data
    input:  data    a Series data or numpy array
            window  the window size, default 2
            step    number of steps to move the window every time, default 1
            func    the function for aggregation, default np.mean
    return a numpy array
    """
    logger = logging.getLogger(__name__)
    try:
        data = data.dropna().values
    except:
        pass

    n = data.shape[0]
    new_data = np.zeros(n-window+1)
    for i in range(0, n-window+1, step):
        new_data[i] = func(data[i:i+window])
    return new_data


def resample(data, column=None, style="week", method='close'):
    """
    Resample a DataFrame or Series by given frequency.
    input:  data    a DataFrame or Series with index as Timestamp
            column  the column to be used when input is a DataFrame
                    and keep is 'min' or 'max', default None
            style   the frequency of resample: 'week'(default), 'month', 'year'
            method  the aggregation method: 'open', 'low', 'high', 'close'(default), 'sum'
    return a slice of input data. If method='sum', function returns a new Series.
    """
    logger = logging.getLogger(__name__)
    df = pd.DataFrame(np.zeros(data.shape[0]), columns=['value'])
    if style=='week':
        df['group'] = (data.index.year-2000)*52 + data.index.week
        jump = df.index[df.group.diff()<0]
        for j in jump:
            df.at[j, 'group'] += 52
            while(j+1<df.shape[0] and df.group[j]>df.group[j+1]):
                j = j+1
                df.at[j, 'group'] += 52
    elif style=='month':
        df['group'] = (data.index.year-2000)*12 + data.index.month
    elif style=='year':
        df['group'] = data.index.year
    else:
        raise ValueError("Available styles are: 'week', 'month', 'year'")

    if method=='close':
        locs =  df.groupby('group').agg(lambda x: x.index[-1]).values.flatten()
        return data.iloc[locs]
    elif method=='open':
        locs = df.groupby('group').agg(lambda x: x.index[0]).values.flatten()
        return data.iloc[locs]
    if data.ndim==2:
        if column is None:
            raise ValueError("resample: Must provide the column name")
        df['value'] = data[column].values
    else:
        df['value'] = data.values
    if method=='low':
        locs = df.groupby('group')['value'].agg(lambda x: x.argmin()).values
        return data.iloc[locs]
    elif method=='high':
        locs = df.groupby('group')['value'].agg(lambda x: x.argmax()).values
        return data.iloc[locs]
    elif method=='sum':
        df['index'] = data.index.values
        val = df.groupby('group')['value'].sum()
        ind = df.groupby('group')['index'].max()
        return pd.Series(val.values, index=ind.values)
    else:
        raise ValueError("Available methods are: 'open', 'min', 'max', 'close', 'sum'")


def get_xy(data, p, intercept=False):
    """
    Get X and Y for autoregression model from time series data
    input:  data    Series or numpy array of data
            p       positive integer, number of previous data points to be used
            intercept  indicate whether to include intercept column, default False
    return X, Y
    """
    logger = logging.getLogger(__name__)
    try:
        data = data.dropna().values
    except:
        pass

    if intercept:
        x = [np.ones(len(data)-p)]
    else:
        x = []
    for i in range(1, 1+p):
        x.append(data[p-i:-i])
    x = np.array(x).T
    return x, data[p:]


def generate_ARMA(size, phi=[], theta=[], init=None):
    """
    generate x_t = phi_p*x_t-p + w_t + theta_q*w_t-q
    input:  size    int, size of data to be generated
            phi     a list of phi values
            theta   a list of theta values
            init    a list of initial values, should be the same size as phi,
                    default all zeros
    """
    logger = logging.getLogger(__name__)
    p = len(phi)
    q = len(theta)
    w = np.random.normal(0, 1, size+q)
    if init is None:
        init = np.zeros(p)
    data = np.concatenate([init, np.zeros(size)])
    for i in range(size):
        data[i+p] += w[i+q] + sum(np.array(theta[::-1])*w[i:i+q])
    for i in range(size):
        data[i+p] += sum(np.array(phi[::-1])*data[i:i+p])
    return data[p:]


def generate_GARCH(size, alpha=[1], beta=[]):
    """
    generate x_t = sig_t * w_t
    s^2_t = a_0 + sum_p b_p * s^2_t-p + sum_q a_q * x^2_t-q
    input:  size    int, size of data to be generated
            alpha   a list of alpha values, default [1]
            beta    a list of beta values, default []
    """
    logger = logging.getLogger(__name__)
    q = len(alpha)-1
    p = len(beta)
    w = np.random.normal(0, 1, size)
    data = np.zeros(size+q)
    sigma = np.zeros(1+p)
    for i in range(size):
        s2 = alpha[0] + sigma[1:].dot(beta) + (data[i:i+q]**2).dot(alpha[1:])
        data[i+q] = np.sqrt(s2) * w[i]
        sigma = np.roll(sigma, -1)
        sigma[-1] = s2
    return data[q:]


if __name__=="__main__":
    import argparse
    import functions
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Functions for time series analysis")
    parser.add_argument('--list', type=bool, nargs='?', const=1, default=0, \
                        help='list all available modules')
    parser.add_argument('--doc', type=str, \
                        help='print documents of given function')
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.list:
        for x in getmembers(functions, isfunction):
            print(x[0])
        exit()
    if not FLAGS.doc is None:
        print(getattr(functions, FLAGS.doc).__doc__)
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))
