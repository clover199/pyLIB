from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

import sys
import os
path = '\\'.join(os.path.abspath(__file__).split('\\')[:-2])
if not path in sys.path:
    sys.path.append(path)

"""
This file defines functions for basic time series analysis
"""


def show_trend(ticker, start='1990-01-01', end=None):
    """
    Plot price change, return and volatility
    input:  ticker  ticker of the stock/ETF
            start   start date, default '2010-01-01'
            end     end date, default current
    Return correlation between return and volatility
    """
    from invest.useful import convert_time
    from invest.get_data import read_ETF
    from invest.calculation import get_returns
    from time_series.functions import moving_agg, resample
    from basic.math import covariance, correlation
    start, end = convert_time(start, end)
    data = read_ETF(ticker)[start:end]

    fig = plt.figure(figsize=(14,6))
    fig.add_axes([0,0.7,1,0.3])
    for c in ['Close','Adj Close']:
        plt.plot(data.index, data[c], label=c)
    plt.xlim(data.index[0],data.index[-1])
    plt.xticks([])
    plt.ylabel("Price")
    plt.legend(loc='best')

    weekly = resample(data, style='week', method='close')
    df = get_returns(weekly.Close, 'simple')

    fig.add_axes([0,0.4,1,0.3])
    m = moving_agg(df, window=52, step=1, func=np.sum)
    plt.plot(df.index[51:], m*100)
    plt.hlines(xmin=data.index[0], xmax=data.index[-1], y=0, linestyle='--')
    plt.xlim(data.index[0],data.index[-1])
    plt.xticks([])
    plt.ylabel("Annual Return")
    plt.legend(loc='best')

    fig.add_axes([0,0.1,1,0.3])
    v = moving_agg(df, window=52, step=1, func=covariance)
    v = np.sqrt(v*52)
    plt.plot(df.index[51:], v*100)
    plt.xlim(data.index[0],data.index[-1])
    plt.ylabel("Volatility")
    plt.gca().set_ylim(bottom=0)
    plt.legend(loc='best')

    return correlation(m, v)


def show_yield(ticker, start='1990-01-01', end=None, weeks=52):
    """
    Calculate annualized return. Simple return is calculated
    input:  ticker  ticker of the stock/ETF
            start   start date, default '2010-01-01'
            end     end date, default current
            weeks   number of weeks for each calculation
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
    return np.round(df,2).T


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
    return np.round(result[::-1].T.astype(float), 2)


def show_dividend(ticker, start='1990-01-01', end=None):
    """
    Show annual dividend yield
    input:  ticker  ticker of the stock/ETF
            start   start date, default '2010-01-01'
            end     end date, default current
    Return correlation between return and volatility
    """
    from invest.useful import convert_time
    from invest.get_data import read_ETF
    from invest.calculation import add_dividend, get_dividend_yield
    from time_series.functions import moving_agg, resample
    from basic.math import covariance, correlation
    start, end = convert_time(start, end)
    data = read_ETF(ticker)[start:end]
    add_dividend(data, price='Close', adj='Adj Close', out='Div')
    div = get_dividend_yield(data, price='Close', div='Div', style='simple')

    plt.figure(figsize=(14,4))
    plt.bar(div.index, div*100)

    ret = data.Div[data.Div>0].tail(6).to_frame()
    ret.index = ret.index.date
    return ret.T


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


def ETF_describe(ticker):
    """ display a comprehensive analysis of one ETF by ticker """
    from invest.get_data import read_ETF, read_ETF_list
    from invest.calculation import get_returns, get_VaR, get_alpha_beta, add_dividend, get_dividend_yield, get_return_vol
    from time_series.functions import resample

    data = read_ETF(ticker)
    print("Ticker:", ticker)
    print("Full name:", read_ETF_list().at[ticker,'Fund Name'])
    print("Start date:", data.index[0].date())
    print("Last date:", data.index[-1].date())

    print("Value at Risk (VAR), alpha/beta, annual volatility and yield calculated by close price")
    alpha = [0.99,0.95,0.9,0.75]
    weekly = resample(data, style="week", method='close')
    ret = get_returns(weekly.Close, style='simple')
    ret_adj = get_returns(weekly['Adj Close'], style='simple')
    risk = pd.DataFrame(0, index=['Last 1 years'], columns=alpha+['alpha','beta','Volatility','Yield','Adj Yield'])
    years = [1,2,3,5,10]
    for y in [x for x in years if (x-1)*52<=weekly.shape[0]]:
        risk.loc['Last {} years'.format(y),alpha] = get_VaR(ret[-52*y:], alpha=alpha, ret=True, scale=52) * 100
        risk.loc['Last {} years'.format(y),['alpha','beta']] = get_alpha_beta(data.Close[-252*y:])
        risk.loc['Last {} years'.format(y), ['Volatility','Yield']] = \
            get_return_vol(ret[-52*y:], scale=52, ret=True).values.flatten()[::-1]*100
        risk.at['Last {} years'.format(y), 'Adj Yield'] = np.mean(ret_adj[-52*y:]) * 5200
    display(np.round(risk,2))

    add_dividend(data, price='Close', adj='Adj Close', out='Dividend')
    div = get_dividend_yield(data, price='Close', div='Dividend', style='simple').rename("Dividend Yield")
    div = np.round(div[-5:].to_frame()*100, 2)
    for c in div.index:
        div.at[c,'payment No.'] = np.sum(data[data.Dividend!=0].index.year==c)
    display(div)


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
