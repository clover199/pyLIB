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
def show_yield(ticker, start=None, end=None, weeks=52):
    """
    Calculate annualized return. Simple return is calculated
    input:  ticker  ticker of the stock/ETF
            start   start date, default '2000-01-01'
            end     end date, default current
            weeks   number of weeks for each calculation, default 52
    Return a DataFrame with three rows: Adj Close, Dividend, Close
    """
    from invest.useful import convert_time
    from invest.get_data import read_ETF
    from invest.calculation import add_dividend, get_returns
    from time_series.functions import resample
    start, end = convert_time(start, end)
    data = read_ETF(ticker)[start:end]
    add_dividend(data, price='Close', adj='Adj Close', out='Dividend')
    data['Dividend'] = np.cumsum(data.Dividend)
    weekly = resample(data, style='week', method='close')
    weekly = weekly[(weekly.shape[0]-1)%weeks::weeks]
    df = get_returns(weekly[['Adj Close','Dividend','Close']], 'simple')
    df['Dividend'] = (weekly.Dividend.diff() / weekly.Close.shift(1))
    df = df*100 * 52/weeks
    from datetime import timedelta
    ds = df.index
    xlim = [ds[0]-timedelta(days=3*weeks), ds[-1]+timedelta(days=3*weeks)]
    plt.figure(figsize=(14,3))
    plt.title("Annualized Return")
    plt.hlines(xmin=xlim[0], xmax=xlim[1], y=0)
    plt.hlines(xmin=xlim[0], xmax=xlim[1],
               y=df['Adj Close'].mean(), linestyle='--', color='#1f77b4')
    plt.hlines(xmin=xlim[0], xmax=xlim[1],
               y=df.Dividend.mean(), linestyle='--', color='#ff7f0e')
    plt.bar(ds, df.Close, width=5*weeks, label='Yield')
    plt.bar(ds, df.Dividend, bottom=df.Close, width=5*weeks, label='Div_Yield')
    plt.plot(ds, df['Adj Close'], 'o-', label='Adj_Yield')
    plt.xlabel("Date to sell")
    plt.xlim(xlim)
    plt.ylim([np.min(df.values)-0.2, np.max(df.values)+0.2])
    plt.legend(bbox_to_anchor=(1.01,0.9), loc='upper left')
    plt.grid()
    df.index = df.index.date

    print(np.round(df,2))
    plt.show()
    return np.round(df,2).T


if __name__=="__main__":
    import argparse
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Calculate annualized return. Simple return is calculated")
    parser.add_argument('ticker', type=str,\
                        help='ticker of the stock/ETF')
    parser.add_argument('--start', type=str, \
                        help="start date, default '2000-01-01'")
    parser.add_argument('--end', type=str, \
                        help='end date, default current')
    parser.add_argument('--weeks', type=int, default=52, \
                        help='number of weeks for each calculation, default 52')
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))
    show_yield(FLAGS.ticker.upper(), FLAGS.start, FLAGS.end, FLAGS.weeks)
