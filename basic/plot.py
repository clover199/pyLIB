from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

"""
This file defines functions to make basic plots.
"""


def fit_t(values):
    """
    Fit data with student t-distribution and display coefficients.
    A fit with normal distribution is displayed in a black dashed line for comparison
    """
    logger = logging.getLogger(__name__)
    from scipy import stats
    df = stats.t.fit(values)
    bins = 50
    n = values.shape[0]
    df = stats.t.fit(values)
    lim = np.max(np.abs(values))
    x = np.linspace(-lim, lim, 1000)
    y = stats.t.pdf(x, *df) * 2*lim/bins*n
    sk, p = stats.kstest(values, 't', df)
    plt.title("size={:d}   df={:.2f}   SK-test p={:.4f}".format(n,df[0],p))
    plt.hist(values, bins=bins, range=[-lim,lim])
    plt.plot(x, y, '-')
    df = stats.norm.fit(values)
    y = stats.norm.pdf(x, *df) * 2*lim/bins*n
    plt.plot(x, y, 'k--')


def plot_correlation(df):
    """
    make correlation plots of input pandas DataFrame
    """
    plt.matshow(df.corr(), fignum=0)
    plt.xticks(range(df.shape[1]), df.columns)
    plt.yticks(range(df.shape[1]), df.columns)
    plt.colorbar()


def plot_hbar(names, values, errs=None, sort=False, tol=0):
    """
    make horizontal bar plot
    input:  names   name of each bar
            values  value of each bar
            errs    error of each bar, default none
            sort    indicate whether to sort values before plot, default False
            tol     the smallest value to be plotted, default 0
    """
    n = len(names)
    keep = sum(values>=tol)
    if sort:
        arg = np.argsort(values)[-keep:]
        names = names[arg]
        values = values[arg]
        if not (errs is None):
            errs = errs[arg]
    ys = np.arange(keep)
    plt.barh(ys, values, xerr=errs)
    plt.yticks(ys, names)


def plot_stacked_bar(data, ticks=None, names=None):
    """
    make stacked bar plot
    input:  data    a 2D array with rows for different type
            ticks   the names of xticks, default None
            names   name of the types, default None
    """
    r, c = data.shape
    if names is None:
        names = np.arange(r)
    bottom = np.sum(data, axis=0)
    top = max(bottom)
    x = np.arange(c)
    for i in range(r):
        bottom = bottom - data[i]
        plt.bar(x, data[i], width=1, bottom=bottom, label=names[i])
    if r<20:
        plt.legend(bbox_to_anchor=(1.01,0.99), loc='upper left')
    for i in range(r):
        for j in range(c):
            if data[i,j]>0.1:
                plt.text(s=names[i], x=x[j]-0.45, y=np.sum(data[i:,j])-data[i,j]/2)
    plt.xlim(-0.5, c-0.5)
    plt.ylim(0, top)
    if not (ticks is None):
        if c>6:
            plt.xticks(x[::c//6], ticks[::c//6])
        else:
            plt.xticks(x, ticks)


if __name__=="__main__":
    import argparse
    import plot
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Plot functions")
    parser.add_argument('--list', type=bool, nargs='?', const=1, default=0, \
                        help='list all available modules')
    parser.add_argument('--doc', type=str, \
                        help='print documents of given function')
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.list:
        for x in getmembers(plot, isfunction):
            print(x[0])
        exit()
    if not FLAGS.doc is None:
        print(getattr(plot, FLAGS.doc).__doc__)
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))
