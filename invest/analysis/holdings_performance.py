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


# 2019-7-28
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
        logger.error("details:" + err)
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
    from invest.get_data import load_data_from_yahoo, get_ETFs
    first_date = holdings.Date.min() - timedelta(days=7)
    data = get_ETFs(tickers, start=first_date)

    from invest.calculation import add_dividend
    for t in tickers:
        add_dividend(data, price=('Close',t), adj=('Adj Close',t), out=('Dividend',t))

    columns = ["Ticker","Buy Date","Buy Price","Current Price","Buy Shares",
        "Reinvested Shares","Capital Gain","Dividend Gain","Total Gain"]
    output = pd.DataFrame(columns=columns, index=holdings.index)
    output['Ticker'] = holdings.Ticker.values
    output['Buy Date'] = holdings.Date.dt.date.values
    output['Buy Price'] = holdings.Price.values
    for t in tickers:
        index = output.index[output.Ticker==t]
        output.loc[index, 'Current Price'] = data['Close'][t][-1]
        dates = holdings.Date[index]
        output.loc[index, 'Close'] = data['Close'][t][dates].values
        output.loc[index, 'Adj Close'] = data['Adj Close'][t][dates].values
        output.loc[index, 'Dividend'] = [data.Dividend[t][d:].sum() for d in dates.values]
    output['Current Price'] = output['Current Price'].astype(float)
    output['Buy Shares'] = holdings.Shares.values
    current_share = output['Close'] / output['Adj Close'] * output['Buy Shares']
    output['Reinvested Shares'] = np.round(current_share - output['Buy Shares'], 5)
    output['Value'] = current_share * output['Current Price']

    days = (date.today() - output['Buy Date']).dt.days / 365
    output['Capital Gain'] = (output['Current Price'] - output['Buy Price']) * output['Buy Shares']
    output['Capital Gain %'] = (output['Current Price'] / output['Buy Price'] - 1) / days * 100
    output['Dividend Gain'] = output['Dividend'] * output['Buy Shares']
    output['Dividend Gain %'] = output['Dividend'] / output['Buy Price'] / days * 100
    output['Total Gain'] = output['Current Price'] * current_share - output['Buy Price'] * output['Buy Shares']
    out = file.replace('/','\\').split('\\')[:-1]
    out.append("output.csv")
    out = '\\'.join(out)
    logger.info("Detailed investment summary saved in file {}".format(out))
    output.to_csv(out, index=False)
    summary = output.groupby('Ticker')['Value','Capital Gain',
                                       'Dividend Gain','Total Gain'].sum()
    print(summary)
    summary['Dividend Gain'] = np.round(summary['Total Gain']-summary['Capital Gain'], 2)
    summary['Value'] = np.round(summary['Value'], 2)
    summary['Capital Gain'] = np.round(summary['Capital Gain'], 2)
    summary['Total Gain'] = np.round(summary['Total Gain'], 2)

    from time_series.functions import resample
    from invest.calculation import get_returns
    weekly = resample(data, style="week", method='close')
    dates = list(np.sort(holdings.Date.unique()))
    dates.append(date.today())  # dates with transactions
    rets = pd.Series()    # weekly returns of total investments
    vals = pd.Series()    # total value of investments
    perf = pd.Series()    # performance, i.e. current value / invested value
    max_value = 0
    for d in range(len(dates)-1):
        tickers = holdings.Ticker[holdings.Date<=dates[d]]
        from_date = dates[d] - np.timedelta64(7, 'D')
        to_date = dates[d+1]
        shares = holdings.Shares[holdings.Date<=dates[d]]
        price = weekly['Close'][tickers][from_date:to_date]
        value = pd.Series(price.values.dot(shares.values), index=price.index)
        rets = rets.append(get_returns(value))
        price = data['Close'][tickers][dates[d]:dates[d+1]]
        value = pd.Series(price.values.dot(shares.values), index=price.index)
        vals = vals.append(value)
        buys = holdings.Price[holdings.Date<=dates[d]]
        perf = perf.append(value / np.dot(buys.values, shares.values))

    xmin, xmax = vals.index[0], vals.index[-1]
    plt.figure(figsize=(15,5))
    plt.plot(rets.index, rets*100, label='Weekly Return')
    plt.plot(perf.index, (perf-1)*100, label='Performance')
    plt.hlines(y=0, xmin=xmin, xmax=xmax, linestyles='--', color='k')
    plt.ylabel("Return (%)", fontsize=20)
    plt.yticks(fontsize=14)
    plt.legend()
    # plt.twinx()
    # plt.plot(vals.index, vals, label='Total value')
    plt.xlim(xmin, xmax)

    from gui.tkinter_widget import display_dataframe, plot_embed_toolbar
    import tkinter as tk
    if frame is None:
        root = tk.Tk()
    else:
        root = frame
    plot_embed_toolbar(root, fig=plt.gcf()).grid(row=0, column=0, columnspan=2)
    tk.Label(root, text="""Notes: \n
        'Annual Return' is just the rescaled weekly return.
        'Performance' is the total return percentage of the day
        (will be affected by buy/sell activities).\n
        """).grid(row=1, column=0, padx=10, pady=10)
    display_dataframe(root, summary).grid(row=1,column=1, padx=10, pady=10)
    total_value = output['Value'].sum()
    total_invest = (output['Buy Shares'] * output['Buy Price']).sum()
    tk.Label(root, font='bold', text="total {:.2f} / {:.2f} = {:.2%}"\
        .format(total_value, total_invest, total_value/total_invest))\
        .grid(row=2, column=1, padx=10, pady=10)
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
