"""
The file can be executed directly from command line
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from datetime import date, timedelta

import sys
import os
path = '\\'.join(os.path.abspath(__file__).split('\\')[:-3])
if not path in sys.path:
    sys.path.append(path)


# 2020-2-17
def holdings_performance(file, frame=None):
    """
    Calculate performance of current holdings.
    input:  file    a csv file containing all transactions
                    columns are Date, Ticker, Shares, Price
            frame   a tkinter frame to show results. Default None
    """
    logger = logging.getLogger(__name__)
    try:
        holdings = pd.read_csv(file, index_col=None, parse_dates=[0]).dropna()
    except Exception as err:
        logger.error("Cannot open holdings file {}".format(file))
        logger.error(err)
        return
    logger.debug("input transactions:")
    logger.debug("{} \t{}\t{}\t{}".format(*holdings.columns))
    for i in range(holdings.shape[0]):
        logger.debug("{} \t{} \t{} \t{}".format(holdings.iloc[i,0].date(),
                                                    *holdings.iloc[i,1:]))
    tickers = holdings.Ticker.unique()
    if len(tickers)==0:
        logger.info("No data provided")
        return

    from invest.get_data import get_latest_ETFs
    first_date = holdings.Date.min() - timedelta(days=7)
    data = get_latest_ETFs(tickers, start=first_date)

    from invest.calculation import add_dividend
    for t in tickers:
        add_dividend(data, price=('Close',t), adj=('Adj Close',t), out=('Dividend',t))

    columns = ["Ticker","Buy Date","Buy Price","Buy Shares","Buy Value","Current Price"]
    output = pd.DataFrame(columns=columns, index=holdings.index)
    output['Ticker'] = holdings.Ticker.values
    output['Buy Date'] = holdings.Date.dt.date.values
    output['Buy Price'] = holdings.Price.values
    output['Buy Shares'] = holdings.Shares.values
    output['Buy Value'] = holdings.Price.values * holdings.Shares.values
    for t in tickers:
        index = output.index[output.Ticker==t]
        output.loc[index, 'Current Price'] = data['Close'][t][-1]
        # calculate reinvested value
        # settlement date is two business days after buy date
        dates = holdings.Date[index]
        loc = np.array([data.index.get_loc(x) for x in dates]) + 2
        loc = np.clip(loc, 0, data.shape[0]-1)
        output.loc[index, 'Close'] = data['Close'][t].iloc[loc].values
        output.loc[index, 'Adj Close'] = data['Adj Close'][t].iloc[loc].values
        output.loc[index, 'Dividend'] = [data.Dividend[t].iloc[d:].sum() for d in loc]
    output['Current Price'] = output['Current Price'].astype(float)
    output['Current Shares'] = output['Close'] / output['Adj Close'] * output['Buy Shares']
    output['Reinvested Shares'] = np.round(output['Current Shares'] - output['Buy Shares'], 5)
    output['Total Value'] = output['Current Shares'] * output['Current Price']

    output['Capital Gain'] = (output['Current Price'] - output['Buy Price']) * output['Buy Shares']
    output['Dividend Gain'] = output['Dividend'] * output['Buy Shares']
    output['Total Gain'] = output['Current Price'] * output['Current Shares'] \
                         - output['Buy Price'] * output['Buy Shares']

    buy = output.groupby('Ticker')["Buy Shares",'Buy Value'].sum()
    buy['Buy Value'] = np.round(buy['Buy Value'], 2)
    buy['Ave Price'] = np.round(buy['Buy Value'] / buy['Buy Shares'], 2)
    buy['Current Price'] = np.round(data['Close'][buy.index].iloc[-1,:].values, 2)
    buy['Changes %'] = np.round((buy['Current Price'] / buy['Ave Price'] - 1)*100, 2)

    gain = output.groupby('Ticker')['Current Shares', 'Total Value', 'Capital Gain',
                                    'Dividend Gain','Total Gain'].sum()
    gain = np.round(gain, 2)

    from gui.tkinter_widget import display_dataframe
    import tkinter as tk
    if frame is None:
        root = tk.Tk()
    else:
        root = frame
    tk.Label(root, font='bold', text="Purchase summary").grid(row=0,column=0, padx=10, pady=10)
    display_dataframe(root, buy).grid(row=1,column=0, padx=10, pady=10)
    tk.Label(root, font='bold', text="Gain summary").grid(row=0,column=1, padx=10, pady=10)
    display_dataframe(root, gain).grid(row=1,column=1, padx=10, pady=10)
    capital_gain = gain['Capital Gain'].sum()
    total_value = gain['Total Value'].sum()
    total_invest = buy['Buy Value'].sum()
    total_gain = total_value - total_invest
    tk.Label(root, text="Capital {:.2f} / {:.2f} = {:.2%}"\
        .format(capital_gain, total_invest, capital_gain/total_invest))\
        .grid(row=2, column=1, padx=10, pady=10)
    tk.Label(root, text="Total {:.2f} / {:.2f} = {:.2%}"\
        .format(total_gain, total_invest, total_gain/total_invest))\
        .grid(row=3, column=1, padx=10, pady=10)
    if frame is None:
        root.mainloop()


if __name__=="__main__":
    import argparse
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Show annual dividend yield")
    parser.add_argument('holdings', type=str,\
                        help='the csv file with transaction details.\n Columns are'
                            "Date, Ticker, Shares, Price")
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))
    holdings_performance(FLAGS.holdings)
