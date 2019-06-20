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

def show_performance(ticker, current=None):
    """
    Calculate total return and effective annual return for different starting point.
    Simple return is calculated
    input:  ticker  ticker of the stock/ETF
            current the date to be calculated
    Return a DataFrame with two rows: Total Return, Annual Return
    """
    from datetime import date, timedelta
    from invest.get_data import read_ETF
    from time_series.functions import change_month
    data = read_ETF(ticker)[['Close','Adj Close']]
    if current is None:
        current = data.index[-1]
    else:
        current = data[:current].index[-1]
    mns = np.array([1, 3, 6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120])
    dates = [change_month(current,-x) for x in mns]
    df = pd.DataFrame(columns=['Close','Adj Close'])
    dates = np.array(dates)
    dates = dates[np.where(dates>=data.index[0].date())]
    for d in dates:
        vals = data[:d]
        if vals.empty:
            break
        df.loc[vals.index[-1].date(), :] = 100 * ( data.loc[current,:] / vals.iloc[-1,:] - 1)
    df['label'] = ['1M','3M','6M','1Y','2Y','3Y','4Y','5Y','6Y','7Y','8Y','9Y','10Y'][:df.shape[0]]
    df['years'] = mns[:df.shape[0]] / 12
    result = pd.DataFrame(index=df.index)
    result['Total Return'] = df.Close
    result['Annual Return'] = df.Close / df.years
    result['Adjusted Total Return'] = df['Adj Close']
    result['Adjusted Annual Return'] = df['Adj Close'] / df.years

    plt.figure(figsize=(14,8))
    plt.subplot(2,1,1)
    plt.title("Performance for close price (not adjusted)")
    plt.bar(df.index.map(lambda x:x-timedelta(days=12)), df.Close, width=24, label='Total Return')
    plt.bar(df.index.map(lambda x:x+timedelta(days=12)), df.Close/df.years, width=24, label='Annual Return')
    plt.hlines(xmin=df.index[-1]-timedelta(days=24), xmax=current, y=0)
    plt.xticks(df.index, df.label)
    plt.xlim(df.index[-1]-timedelta(days=24), current)
    plt.ylim(max(-10,np.min(result.values)-1),min(20,np.max(result.values)+1))
    plt.legend(loc='best')
    plt.grid()
    plt.subplot(2,1,2)
    plt.title("Performance for close price (adjusted)")
    plt.bar(df.index.map(lambda x:x-timedelta(days=12)), df['Adj Close'], width=24, label='Total Return')
    plt.bar(df.index.map(lambda x:x+timedelta(days=12)), df['Adj Close']/df.years, width=24, label='Annual Return')
    plt.hlines(xmin=df.index[-1]-timedelta(days=24), xmax=current, y=0)
    plt.xticks(df.index, df.label)
    plt.xlim(df.index[-1]-timedelta(days=24), current)
    plt.ylim(max(-10,np.min(result.values)-1),min(20,np.max(result.values)+1))
    plt.legend(loc='best')
    plt.grid()

    print(np.round(result[::-1].astype(float), 2))
    plt.show()
    return np.round(result[::-1].astype(float), 2).T


if __name__=="__main__":
    import argparse
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Calculate total return and effective "
        "annual return for different starting point.")
    parser.add_argument('ticker', type=str,\
                        help='ticker of the stock/ETF')
    parser.add_argument('--current', type=str, \
                        help="the date to be calculated, default current")
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))
    show_performance(FLAGS.ticker.upper(), FLAGS.current)
