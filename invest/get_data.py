"""
This file defines functions to update or get financial data.
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import logging
import os

root = '\\'.join(os.path.abspath(__file__).split('\\')[:-2])
data_dir = root + "\\data_temp"

# 2019-6-19
def create_data_folder():
    if 'data_temp' not in os.listdir(root):
        os.chdir(root)
        os.mkdir('data_temp')

# 2019-6-19
def remove_data_folder():
    logger = logging.getLogger(__name__)
    if 'data_temp' in os.listdir(root):
        logger.warning("Deleting folder data_temp")
        os.chdir(data_dir)
        for name in os.listdir(data_dir):
            os.remove(name)
        os.chdir(root)
        os.rmdir('data_temp')

# 2019-6-19
def load_data_from_yahoo(tickers, start=None, end=None):
    """
    Get historical data from Yahoo Finance
    input:  tickers     one ticker or a list of tickers
            start       start date, default "2000-1-1"
            end         end date, default current
    return whatever the website returns
    """
    from pandas_datareader import data as pdr
    import sys
    if not root in sys.path:
        sys.path.append(root)
    from invest.useful import convert_time
    logger = logging.getLogger(__name__)
    start, end = convert_time(start, end)
    try:
        return pdr.get_data_yahoo(tickers, start=start, end=end)
    except:
        logger.error("Failed to load {} from Yahoo Finance".format(tickers))
        return pd.DataFrame()


def update_basic_data(dt="2000-01-01"):
    """
    Get historical data from FRED and save to local folder 'root/data_temp'
    """
    from pandas_datareader import data as pdr
    pdr.get_data_fred('DGS3MO', start=dt).dropna()\
        .rename(columns={'DGS3MO':'Interest'}).to_csv(data_dir+'\\interest.csv')
    print("Interest data loaded and saved as 'interest.csv' from", dt, "to today")
    pdr.get_data_yahoo('^GSPC', start=dt).dropna()\
        .rename(columns={'Close':'Market'})[['Market']].to_csv(data_dir+'\\market.csv')
    print("Market data loaded and saved as 'market.csv' from", dt, "to today")


def read_interest():
    if 'interest.csv' not in os.listdir(data_dir):
        update_basic_data()
    return pd.read_csv(data_dir+'\\interest.csv', index_col=0, parse_dates=True)["Interest"]


def read_market():
    if 'market.csv' not in os.listdir(data_dir):
        update_basic_data()
    return pd.read_csv(data_dir+'\\market.csv', index_col=0, parse_dates=True)["Market"]

# 2019-6-19
def update_ETF_data(tickers, start="2000-01-01", file_dir=data_dir):
    """
    Get historical prices of given tickers from web and save to local folder
    input:  tickers     one ticker or a list of tickers
            start       start date, default "2000-1-1"
            file_dir    directory to save data, default 'root/data_temp'
    No return values
    """
    logger = logging.getLogger(__name__)
    logger.debug("Start date: {}".format(start))
    if type(tickers)==str:
        data = load_data_from_yahoo(tickers, start)
        data.to_csv(file_dir+'\\{}.csv'.format(tickers))
    else:
        data = load_data_from_yahoo(tickers, start)
        for t in tickers:
            d = data.xs(t, level=1, axis=1).dropna()
            if d.empty:
                logger.error("No data fetched for symbol {} using YahooDailyReader".format(t))
            else:
                d.to_csv(file_dir+'{}.csv'.format(t))
    logger.info("Data for tickers {} saved to folder {}".format(tickers, file_dir))


def update_precleared_ETF(dt="2000-01-01"):
    """
    Get historical prices of Precleared ETFs from Yahoo Finance and save to local folder 'data'
    ETFs with less than a year data or closed is not saved.
    """
    logger = logging.getLogger(__name__)
    from pandas_datareader import data as pdr
    from datetime import datetime, date
    from basic.useful import progress_bar
    etfs = pd.read_csv('Preclearance_ETF.csv', index_col=1, header=0, encoding='ISO-8859-1')
    etfs['Exist'] = 0

    def func(t):
        try:
            data = pdr.get_data_yahoo(t, start=dt)
        except:
            return ""
        else:
            if t=='QUS':
                data = data[date(2015,4,16):]
            if data.shape[0]<252 or (datetime.now()-data.index[-1]).days>7:
                return ""
            if data.index[-1]==data.index[-2]:
                data = data[:-1]
            etfs.at[t,'Exist'] = 1
            data.to_csv('data/{}.csv'.format(t))
            return ""
    progress_bar(etfs.index, func)
    etfs = etfs[etfs.Exist==1]
    etfs[['Fund Name']].to_csv('data/ETF_list.csv')


def read_ETF_list():
    """ Get all ETF names in folder 'data' """
    return pd.read_csv('data/ETF_list.csv', index_col=0, header=0, encoding='ISO-8859-1')

# 2019-6-19
def read_ETF(ticker, dt='2000-1-1', file_dir=data_dir):
    """
    Read historical data by ticker
    input:  ticker      one ticker
            dt          start date, default "2000-1-1"
            file_dir    directory to save data, default 'root/data_temp'
    Return a pandas DataFrame
    """
    logger = logging.getLogger(__name__)
    if not os.path.isdir(data_dir):
        logger.error("Path {} doesn't exist".format(file_dir))
        return pd.DataFrame()
    if '{}.csv'.format(ticker) not in os.listdir(file_dir):
        update_ETF_data(ticker, dt, file_dir)
    return pd.read_csv(file_dir+'\\{}.csv'.format(ticker), index_col=0, header=0, parse_dates=True)

# 2019-7-6
def read_portfolio(tickers, column='Adj Close', start='2010-01-01', end=None, keep_na=False, file_dir=data_dir):
    """
    Get weekly data for the given portfolio.
    input:  tickers     a list of tickers of the stocks to consider
            column      the column of data to use
            start       start date, default "2000-1-1"
            end         end date, default current date
            keep_na     indicate whether to keep the ETFs that begin after 'start'
                        default False.
            file_dir    directory to save data, default 'root/data_temp'
    """
    from invest.useful import convert_time
    from datetime import timedelta
    start, end = convert_time(start, end)
    data = []
    for t in tickers:
        data.append(read_ETF(t, start, file_dir)[column].rename(t))
    data = pd.concat(data, axis=1)[start:end]
    if keep_na:
        return data
    data = data.dropna(axis=1, how='any')
    if len(tickers)>data.shape[1]:
        print("{} out of {} ETFs start after {}".format(len(tickers)-data.shape[1],
                                                        len(tickers), start.date()))
    return data


if __name__=="__main__":
    import argparse
    import get_data
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Functions to get data")
    parser.add_argument('--list', type=bool, nargs='?', const=1, default=0, \
                        help='list all available modules')
    parser.add_argument('--doc', type=str, \
                        help='print documents of given function')
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.list:
        for x in getmembers(get_data, isfunction):
            print(x[0])
        exit()
    if not FLAGS.doc is None:
        print(getattr(get_data, FLAGS.doc).__doc__)
        exit()
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))
