"""
The file can be executed directly from command line
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

import sys
import os
path = '\\'.join(os.path.abspath(__file__).split('\\')[:-3])
if not path in sys.path:
    sys.path.append(path)


# 2019-6-19
def show_trend(ticker, start=None, end=None):
    """
    Plot price change, return and volatility
    input:  ticker  ticker of the stock/ETF
            start   start date, default '2000-01-01'
            end     end date, default current
    Return correlation between return and volatility
    """
    from invest.useful import convert_time
    from invest.get_data import read_ETF
    from invest.calculation import get_returns
    from time_series.functions import moving_agg, resample
    from basic.mathe import covariance, correlation
    start, end = convert_time(start, end)
    data = read_ETF(ticker)[start:end]

    fig = plt.figure(figsize=(14,6))
    fig.add_axes([0.05,0.68,0.94,0.3])
    for c in ['Close','Adj Close']:
        plt.plot(data.index, data[c], label=c)
    plt.xlim(data.index[0],data.index[-1])
    plt.xticks([])
    plt.ylabel("Price ($)")
    plt.legend(loc='best')

    weekly = resample(data, style='week', method='close')
    df = get_returns(weekly.Close, 'simple')

    fig.add_axes([0.05,0.38,0.94,0.3])
    m = moving_agg(df, window=52, step=1, func=np.sum)
    plt.plot(df.index[51:], m*100)
    plt.hlines(xmin=data.index[0], xmax=data.index[-1], y=0, linestyle='--')
    plt.xlim(data.index[0],data.index[-1])
    plt.xticks([])
    plt.ylabel("Annual Return (%)")
    plt.legend(loc='best')

    fig.add_axes([0.05,0.08,0.94,0.3])
    v = moving_agg(df, window=52, step=1, func=covariance)
    v = np.sqrt(v*52)
    plt.plot(df.index[51:], v*100)
    plt.xlim(data.index[0],data.index[-1])
    plt.ylabel("Volatility (%)")
    plt.gca().set_ylim(bottom=0)
    plt.legend(loc='best')

    corr = correlation(m, v)
    print("Correlation between return and volatility:", corr)
    plt.show()
    return corr


if __name__=="__main__":
    import argparse
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Show price trend, return and volatility")
    parser.add_argument('ticker', type=str,\
                        help='ticker of the stock/ETF')
    parser.add_argument('--start', type=str, \
                        help="start date, default '2000-01-01'")
    parser.add_argument('--end', type=str, \
                        help='end date, default current')
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))
    show_trend(FLAGS.ticker.upper(), FLAGS.start, FLAGS.end)
