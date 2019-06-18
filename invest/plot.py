from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

"""
This file defines functions to make investment related plots.
"""

# 2019-5-17
def return_vol(ret, vol, tickers):
    """
    Make plot of return vs volatility
    input:  ret     a list of return values (decimal)
            vol     a list of volatility values (decimal)
            tickers a list of tickers
    All inputs should have the same length
    Return figure handler
    """
    logger = logging.getLogger(__name__)
    ret = ret*100
    vol = vol*100
    fig = plt.figure()
    ax = fig.add_subplot(111)
    p = ax.plot(vol, ret, 'o')
    xmax = max(0, max(vol)+1)
    ax.hlines(xmin=0, xmax=xmax, y=0, linestyle='--')
    ax.set_xlim(0, xmax)
    ax.set_ylim(min(0, min(ret)-1), max(0, max(ret)+1))
    ax.set_xlabel("Volatility in %")
    ax.set_ylabel("Return in %")
    ax.grid()
    for i,t in enumerate(tickers):
        t = ax.text(vol[i], ret[i], t)
    return fig


def plot_day_price(data):
    """
    Make plot of daily price change given data
    """
    from datetime import timedelta

    plt.axes([0,0,1,0.3])
    plt.bar(data.index, data.Volume, width=0.5)
    plt.yticks([])
    plt.ylabel("Volume")

    plt.axes([0,0.3,1,0.7])
    plt.bar(data.index, data.High-data.Low, bottom=data.Low, alpha=0.5, width=0.8)
    prices = data.Open.rename('Price')
    prices.index = prices.index - timedelta(hours=6)
    temp = data.Close.rename('Price')
    temp.index = temp.index + timedelta(hours=6)
    prices = prices.append(temp)
    prices.sort_index(inplace=True)
    plt.plot(prices.index, prices, 'ko-')
    plt.xticks([])
    plt.ylabel("Price")


def pie_plot(values, labels):
    pie = plt.figure(figsize=(3,3))
    ax = pie.add_subplot(111)
    ax.axis('equal')
    explode = np.append(0.1, np.zeros(len(values)-1))
    ax.pie(values, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)
    pie.patch.set_alpha(0)
    return pie


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
        exit()
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))
