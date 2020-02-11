"""
This file defines functions to update/load/read financial data.
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import logging
import os

root = '\\'.join(os.path.abspath(__file__).split('\\')[:-2])
data_dir = root + "\\data_temp"

# 2019-11-3
def create_data_folder():
    """
    Generate a default temporary folder to store data downloaded from web
    No input or output
    """
    logger = logging.getLogger(__name__)
    if 'data_temp' not in os.listdir(root):
        logger.info("Create data_temp folder at " + root)
        os.chdir(root)
        os.mkdir('data_temp')
    else:
        logger.info("Folder data_temp already exists at " + root)

# 2019-11-3
def remove_data_folder():
    """
    Delete the default temporary folder that stores data downloaded from web
    No input or output
    """
    logger = logging.getLogger(__name__)
    if 'data_temp' in os.listdir(root):
        logger.warning("Delete folder data_temp at " + root)
        os.chdir(data_dir)
        for name in os.listdir(data_dir):
            os.remove(name)
        os.chdir(root)
        os.rmdir('data_temp')

# 2019-11-4
def load_data_from_yahoo(tickers, start=None, end=None):
    """
    Get historical data from Yahoo Finance
    input:  tickers     one ticker or a list of tickers
            start       start date, default "2000-1-1"（defined in invest.useful)
            end         end date, default current date（defined in invest.useful)
    return whatever the website returns
    """
    from pandas_datareader import data as pdr
    import sys
    if not root in sys.path:
        sys.path.append(root)
    from invest.useful import convert_time
    logger = logging.getLogger(__name__)
    start, end = convert_time(start, end)
    logger.info("Download {} from Yahoo Finance for date range {} to {}".format(
        tickers, start.date(), end.date()))
    try:
        return pdr.get_data_yahoo(tickers, start=start, end=end)
    except:
        logger.error("Failed to download {} from Yahoo Finance. Return empty DataFrame"\
            .format(tickers))
        return pd.DataFrame()

# ----- deprecated -----
def update_basic_data(dt="2000-01-01"):
    """
    Get historical data from FRED and save to local folder 'root/data_temp'
    """
    raise "Function update_basic_data is deprecated."
    from pandas_datareader import data as pdr
    pdr.get_data_fred('DGS3MO', start=dt).dropna()\
        .rename(columns={'DGS3MO':'Interest'}).to_csv(data_dir+'\\interest.csv')
    print("Interest data loaded and saved as 'interest.csv' from", dt, "to today")
    pdr.get_data_yahoo('^GSPC', start=dt).dropna()\
        .rename(columns={'Close':'Market'})[['Market']].to_csv(data_dir+'\\market.csv')
    print("Market data loaded and saved as 'market.csv' from", dt, "to today")

# ----- deprecated -----
def read_interest():
    raise "Function read_interest is deprecated."
    if 'interest.csv' not in os.listdir(data_dir):
        update_basic_data()
    return pd.read_csv(data_dir+'\\interest.csv', index_col=0, parse_dates=True)["Interest"]

# ----- deprecated -----
def read_market():
    raise "Function read_market is deprecated."
    if 'market.csv' not in os.listdir(data_dir):
        update_basic_data()
    return pd.read_csv(data_dir+'\\market.csv', index_col=0, parse_dates=True)["Market"]

# 2019-11-4
def update_ETF_data(ticker, start=None, end=None, file_dir=data_dir):
    """
    Get historical prices of given tickers from web and save to local folder
    input:  ticker      one ticker
            start       start date, default "2000-1-1"（defined in invest.useful)
            end         end date, default current date（defined in invest.useful)
            file_dir    directory to save data, default 'root/data_temp'
    Return DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.debug("Start date: {}".format(start))
    data = load_data_from_yahoo(ticker, start=start, end=end)
    if not data.empty:
        data.to_csv(file_dir+'\\{}.csv'.format(ticker))
        logger.info("Save {}.csv to folder {}".format(ticker, file_dir))
    return data

# ----- deprecated -----
def update_precleared_ETF(dt="2000-01-01"):
    """
    Get historical prices of Precleared ETFs from Yahoo Finance and save to local folder 'data'
    ETFs with less than a year data or closed is not saved.
    """
    raise "Function update_precleared_ETF is deprecated."
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

# ----- deprecated -----
def read_ETF_list():
    """ Get all ETF names in folder 'data' """
    raise "Function read_ETF_list is deprecated."
    return pd.read_csv('data/ETF_list.csv', index_col=0, header=0, encoding='ISO-8859-1')

# 2019-11-4 tend to deprecate
def read_ETF(ticker, start=None, end=None, file_dir=data_dir, update=False):
    """
    Read historical data by ticker
    input:  ticker      one ticker
            start       start date, default "2000-1-1"（defined in invest.useful)
            end         end date, default current date（defined in invest.useful)
            file_dir    directory to save data, default 'root/data_temp'
            update      indicate whether to update local file, default False
    Return a pandas DataFrame
    """
    logger = logging.getLogger(__name__)
    if not os.path.isdir(file_dir):
        logger.error("Path {} doesn't exist".format(file_dir))
        return pd.DataFrame()
    if update or '{}.csv'.format(ticker) not in os.listdir(file_dir):
        return update_ETF_data(ticker, start=start, end=end, file_dir=file_dir)
    return pd.read_csv(file_dir+'\\{}.csv'.format(ticker),
                       index_col=0, header=0, parse_dates=True)

# 2019-11-4 tend to deprecate
def read_portfolio(tickers, column='Close', start=None, end=None,
                   file_dir=data_dir, update=False):
    """
    Get weekly data for the given portfolio.
    input:  tickers     a list of tickers of the stocks to consider
            column      the column of data to use, default "Colse"
            start       start date, default "2000-1-1"（defined in invest.useful)
            end         end date, default current date（defined in invest.useful)
            file_dir    directory to load data, default 'root/data_temp'
            update      indicate whether to update local file, default False
    """
    data = []
    for t in tickers:
        data.append(read_ETF(
            t, start=start, end=end, file_dir=file_dir, update=update
        )[column].rename(t))
    data = pd.concat(data, axis=1)
    return data

# 2019-11-4
def get_latest_ETF(ticker, start=None, file_dir=data_dir):
    """
    Get historical data by ticker from local file or from Yahoo Finance.
    If the local file exists and is updated today, return local file, otherwise
    download from Yahoo. If download fails, return the local file.
    input:  ticker      one ticker
            start       start date, default "2000-1-1"（defined in invest.useful)
            file_dir    directory to save data, default 'root/data_temp'
                        if None, download from Yahoo
    Return a pandas DataFrame
    """
    logger = logging.getLogger(__name__)
    if file_dir is None:
        logger.info("No file path given. Just download data from Yahoo Finance")
        return load_data_from_yahoo(ticker, start=start, end=None)
    if not os.path.isdir(data_dir):
        logger.warning("Path {} doesn't exist. Download data from Yahoo Finance"\
            .format(file_dir))
        return load_data_from_yahoo(ticker, start=start, end=None)
    if '{}.csv'.format(ticker) not in os.listdir(file_dir):
        logger.info("{}.csv doesn't exist in {}. Download data from Yahoo Finance"\
            .format(ticker, file_dir))
        return update_ETF_data(ticker, start=start, end=None, file_dir=file_dir)
    import datetime
    update_stamp = os.path.getmtime(file_dir+'\\{}.csv'.format(ticker))
    update_time = datetime.datetime.fromtimestamp(update_stamp)
    if update_time.date() == datetime.date.today():
        logger.info("Read data from {}\\{}.csv".format(file_dir, ticker))
        data = pd.read_csv(file_dir+'\\{}.csv'.format(ticker),
                           index_col=0, header=0, parse_dates=True)
        if start is None:
            return data
        else:
            before = data[:start]
            if before.empty:
                return update_ETF_data(ticker, start=start, end=None, file_dir=file_dir)
            else:
                return data[start:]
    logger.info("Update {}\\{}.csv".format(file_dir, ticker))
    return update_ETF_data(ticker, start=start, end=None, file_dir=file_dir)

# 2019-11-4
def get_latest_ETFs(tickers, start=None, file_dir=data_dir):
    """
    Get historical data by ticker from local file or from Yahoo Finance.
    If the local file exists and is updated today, return local file, otherwise
    download from Yahoo. If download fails, return the local file.
    input:  tickers     a list of tickers
            start       start date, default "2000-1-1"（defined in invest.useful)
            file_dir    directory to save data, default 'root/data_temp'
                        if None, download from Yahoo
    Return a pandas DataFrame with multi-index
    """
    data = []
    for t in tickers:
        d = get_latest_ETF(t, start=start, file_dir=file_dir)
        d.columns = pd.MultiIndex.from_product([d.columns, [t]])
        data.append(d)
    return pd.concat(data, axis=1).sort_index(axis=1)


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
