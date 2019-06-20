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
def show_dividend(ticker, start=None, end=None):
    """
    Show annual dividend yield
    input:  ticker  ticker of the stock/ETF
            start   start date, default '2000-01-01'
            end     end date, default current
    Return a DataFrame
    """
    from invest.useful import convert_time
    from invest.get_data import read_ETF
    from invest.calculation import add_dividend, get_dividend_yield
    start, end = convert_time(start, end)
    data = read_ETF(ticker)[start:end]
    add_dividend(data, price='Close', adj='Adj Close', out='Div')
    div = get_dividend_yield(data, price='Close', div='Div', style='simple')

    plt.figure(figsize=(14,4))
    plt.bar(div.index, div*100)

    ret = data.Div[data.Div>0].tail(6).to_frame()
    ret.index = ret.index.date

    print(ret)
    plt.show()
    return ret.T


if __name__=="__main__":
    import argparse
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Show annual dividend yield")
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
    show_dividend(FLAGS.ticker.upper(), FLAGS.start, FLAGS.end)
