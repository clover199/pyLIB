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


def investment_performance(ticker, price, quantity, start, end=None):
    """
    Calculate performance of given stock/ETF.
    input:  ticker  ticker of the stock/ETF
            start   start date
            end     end date, default current
    """
    from invest.useful import convert_time
    from invest.get_data import read_ETF
    from invest.plot import plot_day_price
    from invest.calculation import add_dividend, get_return_vol
    from time_series.functions import resample
    start, end = convert_time(start, end)
    data = read_ETF(ticker)[start:end]
    plot_day_price(data)
    plt.plot([start], [price], 'ro')
    plt.show()
    add_dividend(data, price='Close', adj='Adj Close', out='Dividend')
    temp = data[['Dividend']][data.Dividend!=0].copy()
    temp.index = temp.index.date
    display(temp.T)
    weekly = resample(data, style='week', method='close')
    rv = get_return_vol(weekly[['Close','Adj Close']], scale=52, ret=False, plotit=False)
    rv['Total Return'] = data[['Close','Adj Close']].iloc[-1,:] / data[['Close','Adj Close']].iloc[0,:] - 1
    rv = rv * 100
    rv['Gain'] = np.round(price * quantity * rv['Total Return'] / 100, 2)
    display(rv)
    print("Actual gain without reinvest: {:.2f}".format( (data.Close[-1]-price) * quantity ))
    print("Dividend gain: {:.2f}".format( data.Dividend.sum() * quantity ))


def Sharpe_filter(tickers, keep=50, start='2015-01-01', end=None, column='Adj Close'):
    """
    Find stocks/ETFs with high Sharpe ratio.
    input:  tickers tickers of the stocks/ETFs
            keep    number of stocks/ETFs to keep
            start   start date, default '2015-01-01'
            end     end date, default current
            column  the column of data to use, default 'Adj Close'
    Return a list of tickers
    """
    from invest.get_data import read_portfolio
    from invest.calculation import get_return_vol
    from time_series.functions import resample

    data = read_portfolio(tickers, column, start, end)
    weekly = resample(data, column, style="week", method='close')
    rv = get_return_vol(weekly, scale=52, ret=False, plotit=False)
    rv['Sharpe'] = rv.Return / rv. Volatility
    rv = rv.sort_values('Sharpe')[::-1]
    return rv[:keep].index.values


def find_allocation(tickers, start='2015-01-01', end=None, column='Adj Close'):
    """
    Find the allocation with the highest Sharpe ratio.
    input:  tickers tickers of the stocks/ETFs
            start   start date, default '2015-01-01'
            end     end date, default current
            column  the column of data to use, default 'Adj Close'
    Return a pandas Serires of allocations
    """
    from invest.get_data import read_portfolio
    from invest.calculation import minimize_risk
    from time_series.functions import resample
    from basic.plot import plot_stacked_bar

    data = read_portfolio(tickers, column, start, end)
    weekly = resample(data, column, style="week", method='close')
    rv = minimize_risk(weekly, returns=None, strict=True, riskfree=None, max_alloc=1, scale=52,
                  ret=False, verbose=True, plotit=True)
    plt.figure(figsize=(14,3))
    plot_stacked_bar(rv[rv.columns[-3::-1]].T.values, names=rv.columns[-3::-1], ticks=np.round(rv.index*100,2))
    plt.xlabel("Return in %")
    plt.show()
    sharpe = rv.Return/rv.Volatility
    arg = sharpe.argmax()
    return rv.loc[arg,:][:-2]


def portfolio_analysis(tickers, alloc=None, start='2010-01-01', end=None):
    """
    Given a series of tickers, return a summery of each ETF in a DataFrame
    input:  tickers tickers of stocks/ETFs
            alloc   allocation for given stocks/TEFs, default None
            start   start date, default "2010-01-01"
            end     end date, default today
    """
    from invest.get_data import read_ETF, read_portfolio
    from invest.calculation import get_return_vol, get_alpha_beta, minimize_risk
    from invest.useful import convert_time
    from time_series.functions import resample
    from basic.useful import progress_bar
    from basic.plot import plot_correlation, plot_stacked_bar
    start, end = convert_time(start, end)
    if not (alloc is None):
        if len(tickers)!=len(alloc):
            raise ValueError("Length of shares and tickers should be the same if shares is given.")
        alloc = np.array(alloc) / np.sum(alloc)

    plt.figure(figsize=(15,3))
    plt.subplot(131)
    plt.title("Calculated by Close Price")
    close = read_portfolio(tickers, 'Close', start, end)
    close = close / close.iloc[0,:]
    weekly = resample(close, style="week", method='close')
    al_clo = minimize_risk(weekly, returns=None, strict=True, riskfree=None, max_alloc=1, scale=52,
                  ret=False, verbose=False, plotit=True)
    if not (alloc is None):
        weekly = weekly.dot(alloc)
        rv = get_return_vol(weekly, scale=52) * 100
        plt.plot(rv.Volatility, rv.Return, 'bo')

    plt.subplot(132)
    plt.title("Calculated by Adjusted Close Price")
    adj = read_portfolio(tickers, 'Adj Close', start, end)
    adj = adj / adj.iloc[0,:]
    weekly = resample(adj, style="week", method='close')
    al_adj = minimize_risk(weekly, returns=None, strict=True, riskfree=None, max_alloc=1, scale=52,
                  ret=False, verbose=False, plotit=True)
    if not (alloc is None):
        weekly = weekly.dot(alloc)
        rv = get_return_vol(weekly, scale=52) * 100
        plt.plot(rv.Volatility, rv.Return, 'bo')

    plt.subplot(133)
    df = pd.DataFrame()
    def func(t):
        data = read_ETF(t)
        if data.index[0]<start:
            data = data[start:end]
            a, b = get_alpha_beta(data.Close, ret_type='simple', dspl=False)
            df.at[t, 'alpha'] = a
            df.at[t, 'beta'] = b
    progress_bar(tickers, func, disable=True)
    plt.plot(df.beta, df.alpha, 'o')
    if not (alloc is None):
        total = close.dot(alloc)
        alpha, beta = get_alpha_beta(total, ret_type='simple', dspl=False)
        plt.plot(beta, alpha, 'bo')
    plt.xlabel('Beta')
    plt.ylabel("Alpha")
    plt.hlines(xmin=-0.1, xmax=1.2, y=0, linestyles='--')
    plt.xlim(-0.1,1.2)
    for t in df.index:
        plt.text(df.beta[t], df.alpha[t], t)

    plt.figure(figsize=(14,3))
    plt.title("Allocation calculated from close price")
    plot_stacked_bar(al_clo[al_clo.columns[-3::-1]].T.values, names=al_clo.columns[-3::-1],
                     ticks=np.round(al_clo.Return*100,2))
    plt.xlabel("Return in %")

    plt.figure(figsize=(14,3))
    plt.title("Allocation calculated from adjusted close price")
    plot_stacked_bar(al_adj[al_adj.columns[-3::-1]].T.values, names=al_adj.columns[-3::-1],
                     ticks=np.round(al_adj.Return*100,2))
    plt.xlabel("Return in %")

    plt.figure(figsize=(14,3))
    plt.title("Price change (not adjusted)")
    for c in close.columns:
        plt.plot(close.index, close[c], '-', label=c)
    plt.hlines(xmin=close.index[0], xmax=close.index[-1], y=1, linestyles='--')
    plt.xlim(close.index[0], close.index[-1])
    plt.legend(bbox_to_anchor=(1.01,0.99), loc='upper left')
    if not (alloc is None):
        plt.plot(close.index, total, 'k-')

    plt.figure(figsize=(6,5))
    plot_correlation(close)
    plt.show()




if __name__=="__main__":
    import argparse
    import analysis
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Functions for investment analysis")
    parser.add_argument('--list', type=bool, nargs='?', const=1, default=0, \
                        help='list all available modules')
    parser.add_argument('--doc', type=str, \
                        help='print documents of given function')
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    parser.add_argument('--eval', type=str, \
                        help='evaluate functions')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.list:
        for x in getmembers(analysis, isfunction):
            print(x[0])
        exit()
    if not FLAGS.doc is None:
        print(getattr(analysis, FLAGS.doc).__doc__)
        exit()
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))
    if not FLAGS.eval is None:
        eval(FLAGS.eval)
