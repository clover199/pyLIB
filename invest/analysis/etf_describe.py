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
def ETF_describe(ticker):
    """ Display a comprehensive analysis of one ETF by ticker """
    from invest.get_data import read_ETF, read_ETF_list
    from invest.calculation import get_returns, get_VaR, get_alpha_beta, add_dividend
    from invest.calculation import get_dividend_yield, get_return_vol
    from time_series.functions import resample

    data = read_ETF(ticker)
    print("Ticker:", ticker)
    # print("Full name:", read_ETF_list().at[ticker,'Fund Name'])
    print("Start date:", data.index[0].date())
    print(data.iloc[-1,:-1].to_frame().T)

    print("\nValue at Risk (VAR), alpha/beta, annual volatility and yield "
        "calculated by close price\n")
    alpha = [0.99,0.95,0.9,0.75]
    weekly = resample(data, style="week", method='close')
    ret = get_returns(weekly.Close, style='simple')
    ret_adj = get_returns(weekly['Adj Close'], style='simple')
    risk = pd.DataFrame(0, index=['Last 1 years'],
                columns=alpha+['alpha','beta','Volatility','Yield','Adj Yield'])
    years = [1,2,3,5,10]
    for y in [x for x in years if (x-1)*52<=weekly.shape[0]]:
        risk.loc['Last {} years'.format(y),alpha] = get_VaR(ret[-52*y:], alpha=alpha,
                                                            ret=True, scale=52) * 100
        risk.loc['Last {} years'.format(y),['alpha','beta']] = get_alpha_beta(data.Close[-252*y:])
        risk.loc['Last {} years'.format(y), ['Volatility','Yield']] = \
            get_return_vol(ret[-52*y:], scale=52, ret=True).values.flatten()[::-1]*100
        risk.at['Last {} years'.format(y), 'Adj Yield'] = np.mean(ret_adj[-52*y:]) * 5200
    print(np.round(risk,2))
    print()
    add_dividend(data, price='Close', adj='Adj Close', out='Dividend')
    div = get_dividend_yield(data,
        price='Close', div='Dividend', style='simple').rename("Dividend Yield")
    div = np.round(div[-5:].to_frame()*100, 2)
    for c in div.index:
        div.at[c,'payment No.'] = np.sum(data[data.Dividend!=0].index.year==c)
    print(div)
    print('\nThe most recent 6 dividend payments')
    print(data[data.Dividend>0].Dividend[-6:].rename("Dividend per share"))


if __name__=="__main__":
    import argparse
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Display a comprehensive analysis "
                                                 "of one ETF by ticker ")
    parser.add_argument('ticker', type=str,\
                        help='ticker of the stock/ETF')
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))
    ETF_describe(FLAGS.ticker.upper())
