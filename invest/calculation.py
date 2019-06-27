from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import logging

import sys
import os
path = '\\'.join(os.path.abspath(__file__).split('\\')[:-2])
if not path in sys.path:
    sys.path.append(path)

"""
This file defines functions to calculate properties of financial data.
"""


def summary(data, digits=2, name=None):
    """
    Calculate some simple statistics of the input time sereise data as a Series
    or columns of a DataFrame.
    input:  data    a pandas Series
            digits  number of digits to keep, default 2
            name    name of the series, default None
                    only used when data is a Series
    Return a DataFrame with with rows for different input columns
    """
    logger = logging.getLogger(__name__)
    if data.ndim!=1:
        raise ValueError("invest.calculation.summary only takes pandas Series as input data")

    if name is None:
        name = data.name
        if name is None:
            name = 0

    df = pd.DataFrame()
    df.at[name, 'Start'] = data.index[0]
    df.at[name, 'End'] = data.index[-1]
    n = data.shape[0]
    mean = np.mean(data)
    df.at[name, 'Size'] = n
    df.at[name, 'Mean'] = np.round(mean, digits)
    df.at[name, 'Std'] = np.round(np.sqrt( np.mean((data-mean)**2) * n / (n-1) ), digits)
    df.at[name, 'Skew'] = np.round( np.mean((data-mean)**3) / np.mean((data-mean)**2)**1.5, digits)
    df.at[name, 'Kurtosis'] = np.round( np.mean((data-mean)**4) / np.mean((data-mean)**2)**2 - 3, digits)
    data = np.sort(data.values).flatten()
    df.at[name, 'min'] = data[0]
    for p in [0.25, 0.5, 0.75]:
        i = int(n*p)
        ratio = np.abs(n*p - i - p)
        df.at[name, "{:.0%}".format(p)] = ratio * data[i-1] + (1-ratio) * data[i]
    df.at[name, 'max'] = data[n-1]
    df = df.astype({'Size':int})
    return df

# 2019-5-17
def get_returns(data, style='simple', fillna=True):
    """
    Calculate simple and log returns for the input time sereise data
    input:  data    a pandas Series or DataFrame
            style   return type: 'simple' (default), or 'log'
            fillna  indicate whether to fillna by 0. default True
    Return the same type as input data.
    """
    logger = logging.getLogger(__name__)
    if style=='simple':
        ret = (data-data.shift(1)) / data.shift(1)
    elif style=='log':
        ret = np.log( data / data.shift(1) )
    else:
        raise ValueError("style can be either 'simple' or 'log'")
    if fillna:
        return ret.iloc[1:].fillna(0)
    else:
        return ret.iloc[1:]

# 2019-6-24
def add_dividend(data, price='Close', adj='Adj_Close', out='Dividend'):
    """
    Calculate dividend from stock/bond value and adjusted value.
    input:  data    a DataFrame with column 'Close'
            price   name of the column with original price, default 'Close'
            adj     name of the column with adjusted price, default 'Adj_Close'
            out     name of output, default 'Dividend'
    No return values. The original data is changed with one additional column.
    """
    logger = logging.getLogger(__name__)
    share = data[adj] / data[price]
    share = (share - share.shift(1)) / share.shift(1)
    data.loc[:,out] = np.round(share * data[price], 3).fillna(0)


def get_dividend_yield(data, price='Close', div='Dividend', style='simple'):
    """
    Calculate annual yield of dividend with price and dividend/adjusted price.
    input:  data    a pandas DataFrame with at least two columns
            price   the column name of original prices, default 'Close'
            div     the column name of dividend, default 'Dividend'
            style   calculation method: 'simple' (default), or 'log'
    Either adj or div should be not None.
    Return a Series with dividend yield at year end.
    """
    logger = logging.getLogger(__name__)
    from time_series.functions import resample
    rsp = resample(data[price], style='year', method='close')
    div = resample(data[div], style='year', method='sum')
    ret = ( div / rsp.shift(1) ).iloc[1:].fillna(0)
    ret.index = ret.index.year
    if style=='simple':
        return ret
    if style=='log':
        return np.log( 1 + ret )


def get_alpha_beta(data, ret_type='simple', dspl=False):
    """
    Calculate alpha and beta for input time series data using CAPM
    R - risk_free = alpha + beta * (M - resk_free)
    input:  data    a pandas Series
            ret_type return type: 'simple' (default), or 'log'
            dspl    indicate whether to display fit summary
    return two values alpha, beta
    """
    logger = logging.getLogger(__name__)
    if data.ndim!=1:
        raise ValueError("invest.calculation.get_alpha_beta only takes pandas Series as input data")
    from invest.get_data import read_market, read_interest
    df = data.to_frame()
    df['data'] = get_returns(data, ret_type)
    df['market'] = get_returns(read_market(), ret_type)
    df['interest'] = read_interest()
    df['date'] = df.index
    df['risk_free'] = df.interest * 0.01 * (df.date-df.date.shift(1)).dt.days / 260
    df.dropna(axis=0, how='any', inplace=True)
    df['y'] = df.data - df.risk_free
    df['x'] = df.market - df.risk_free
    from machine_learning.Msklearn import LinearRegression
    lm = LinearRegression(intercept=True)
    lm.fit(df.x.values, df.y.values)
    if dspl:
        lm.summary()
    alpha, beta = lm.beta
    return alpha, beta


def get_VaR(data, alpha=0.99, ret=False, scale=52):
    """
    Calculate Value at Risk for given time series data.
    Simple return is used.
    input:  data    a pandas Series
            alpha   a number or list of numbers from 0 to 1
                    confidence percentage, default 0.99
            ret     indicate whether the input data is return or price (default).
                    default False, and simple return is calculated
            scale   factor if convert to annual return. Default 1.
                    12 for monthly, 52 for weekly, and 252 for daily data
    return  a (list of) number of losing percentage with given probability
    """
    logger = logging.getLogger(__name__)
    if data.ndim!=1:
        raise ValueError("invest.calculation.get_VaR only takes pandas Series as input data")
    if ret:
        ret = data
    else:
        ret = get_returns(data, 'simple')
    from basic.mathe import covariance
    vol = np.sqrt(covariance(ret) * scale)
    ret = np.mean(ret) * scale
    from scipy import stats
    return stats.norm.ppf(1-np.array(alpha), ret, vol)

# 2019-5-17
def get_return_vol(data, scale=1, ret=False, plotit=False):
    """
    Calculate return and volatility of given data.
    input:  data    a DataFrame or Series with ETF prices
            scale   factor if convert to annual return. Default 1
                    12 for monthly, 52 for weekly, and 252 for daily data
            ret     indicate whether the input data is return or price (default).
                    default False, and simple return is calculated
            plotit  indicate whether to make plots
    return a DataFrame with tickers as index and columns: "Return", "Volatility"
    """
    from basic.mathe import covariance
    logger = logging.getLogger(__name__)
    if data.ndim==1:
        data = data.to_frame()
    if ret:
        rts = data
    else:
        rts = get_returns(data, 'simple')
    ret = rts.mean().values * scale
    vol = rts.apply(lambda x: np.sqrt(covariance(x) * scale)).values
    if plotit:
        from invest.plot import return_vol
        return_vol(ret, vol, data.columns)
    return pd.DataFrame({"Return":ret, "Volatility":vol}, index=data.columns)

# 2019-5-??
def minimize_risk(data, returns=None, strict=True, riskfree=None, max_alloc=1,
                  shot_sell=False, scale=1, ret=False, verbose=True, plotit=False):
    """
    Calculate the portfolio with the lowest risk given returns.
    input:  data        a DataFrame with ETF prices or returns
            returns     a list of return values to use
            strict      indicate whether the return value should be strict, default True
            riskfree    the risk-free return rate, default None.
                        If not None, then a risk free stock is added for consideration
            max_alloc   a number from 0 to 1. The maximum percentage of one stock
            short_sell  indicate whether to allow short sell. Default False.
            scale       factor if convert to annual return. Default 1.
                        12 for monthly, 52 for weekly, 252 for daily
            ret         indicate whether the input data is return or price (default).
                        default False, and simple return is calculated
            verbose     indicate whether to print progress bar, default True.
            plotit      indicate whether to make Return-Volatility plots, default False.
    Returns a DataFrame with columns as [tickers,Volatility,Return] and index as targeted returns
    """
    logger = logging.getLogger(__name__)
    if ret:
        weekly = data
    else:
        weekly = get_returns(data, 'simple')
    ret = weekly.mean().values * scale
    cov = weekly.cov().values * scale
    if short_sell:
        return pd.DataFrame()
    n = data.shape[1]
    if riskfree is None:
        aloc = pd.DataFrame(columns=np.append(data.columns, ['Volatility','Return']))
        bounds = [(0,max_alloc)]*n
    else:
        ret = np.append(ret, riskfree)
        cov = np.hstack([ np.vstack([cov,np.zeros([1,n])]), np.zeros([n+1,1]) ])
        aloc = pd.DataFrame(columns=np.append(data.columns, ['risk-free','Volatility','Return']))
        bounds = [(0,max_alloc)]*n + [(0,1)]
        n += 1
    if returns is None:
        returns = np.linspace(min(ret),max(ret), 25, endpoint=True)

    from scipy.optimize import minimize
    from basic.useful import progress_bar
    def func(alpha):
        def loss(x):
            return x.dot(cov).dot(x)
        def jac(x):
            return cov.dot(x) * 2
        cons1 = {'type':'eq',
                 'fun': lambda x: np.ones(n).dot(x) - 1,
                 'jac': lambda x: np.ones(n)}
        types = 'eq'
        if not strict: types = 'ineq'
        cons2 = {'type':types,
                 'fun': lambda x: ret.dot(x) - alpha,
                 'jac': lambda x: ret}
        x = minimize(loss, np.ones(n)/n, jac=jac, constraints=[cons1,cons2], bounds=bounds, method='SLSQP')
        aloc.loc[alpha, :] = np.append(np.round(x['x'],4), [np.sqrt(x['fun']), ret.dot(x['x'])] )
        return ""
    progress_bar(returns, func, disable=not verbose)
    if plotit:
        import matplotlib.pyplot as plt
        from invest.plot import return_vol
        vol = np.sqrt( np.diag(cov) )
        return_vol(ret, vol, data.columns)
        plt.plot(aloc.Volatility*100, aloc.Return*100, '.-')
        sharpe = aloc.Return/aloc.Volatility
        arg = sharpe.argmax()
        plt.plot(aloc.Volatility[arg]*100, aloc.Return[arg]*100, 'rX', markersize=12)
        print("Max Sharpe ratio is {:.2f}".format(sharpe[arg]))
    return aloc.astype(float)


if __name__=="__main__":
    import argparse
    import calculation
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Functions for time series calculations")
    parser.add_argument('--list', type=bool, nargs='?', const=1, default=0, \
                        help='list all available modules')
    parser.add_argument('--doc', type=str, \
                        help='print documents of given function')
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.list:
        for x in getmembers(calculation, isfunction):
            print(x[0])
        exit()
    if not FLAGS.doc is None:
        print(getattr(calculation, FLAGS.doc).__doc__)
        exit()
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))
