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


# 2019-11-5
def ETF_describe_print(ticker):
    """ Display a comprehensive analysis of one ETF by ticker """
    from invest.get_data import get_latest_ETF
    from invest.calculation import get_returns, get_VaR, get_alpha_beta
    from invest.calculation import add_dividend, get_dividend_yield
    from time_series.functions import resample

    data = get_latest_ETF(ticker)
    print("\nTicker:", ticker)
    print("\nData start from date:", data.index[0].date())
    print("\n----- Current positions -----")
    print( np.round( data.iloc[-1,:-1].to_frame().T, 2) )

    market = get_latest_ETF('SPY')
    wm = resample(market, style="week", method='close')
    weekly = resample(data, style="week", method='close')
    a, b = get_alpha_beta(weekly.Close, wm.Close, risk_free=0, scale=1, dspl=False)
    print("\nGet Alpha and Beta by using SPY as market and zero risk-free rate:")
    print("Alpha: ", np.round(a, 4) )
    print("Beta:  ", np.round(b, 4) )

    alpha = [0.99, 0.95, 0.9, 0.75, 0.5]
    print("\n------ Value at Risk (VAR) calculated by weekly close price -----")
    print(100 * np.round(
        get_VaR(weekly.Close, alpha=alpha, scale=52), 4) )

    print("\nDividend annual yield and payment frequency per year")
    add_dividend(data, price='Close', adj='Adj Close', out='Dividend')
    div = get_dividend_yield(data,
        price='Close', div='Dividend', style='simple').rename("Dividend Yield (%)")
    div = np.round(div[-5:].to_frame()*100, 2)
    for c in div.index:
        div.at[c,'payment No.'] = np.sum(data[data.Dividend!=0].index.year==c)
    print(div)
    print('\nThe most recent 6 dividend payments')
    print(data[data.Dividend>0].Dividend[-6:].rename("Dividend per share").to_frame())


# 2019-11-7
def ETF_describe_window(frame, ticker='SPY', start=None, end=None):
    """ Display a comprehensive analysis in windows window """
    import tkinter as tk
    from invest.get_data import get_latest_ETF
    from invest.calculation import get_returns, get_VaR, get_alpha_beta
    from invest.calculation import add_dividend, get_dividend_yield
    from time_series.functions import resample
    from gui.tkinter_widget import display_dataframe

    data = get_latest_ETF(ticker, start=start)
    if end is not None:
        data = data[:end]
    tk.Label(frame, text=ticker, font=("Helvetica", 16))\
        .grid(row=1, column=1, padx=10, pady=10)
    tk.Label(frame, text="Data start from date: {}".format(data.index[0].date()))\
        .grid(row=1, column=2, padx=10, pady=10)
    df = np.round( data.iloc[-1,:-1].to_frame().T, 2)
    df.index = [x.date() for x in df.index]
    df.index.name = "Today:"
    display_dataframe(frame, df).grid(row=3, column=1, columnspan=2, padx=10, pady=10)

    market = get_latest_ETF('SPY')
    wm = resample(market, style="week", method='close')
    weekly = resample(data, style="week", method='close')
    a, b = get_alpha_beta(weekly.Close, wm.Close, risk_free=0, scale=1, dspl=False)
    print("\nGet Alpha and Beta by using SPY as market and zero risk-free rate:")
    print("Alpha: ", np.round(a, 4) )
    print("Beta:  ", np.round(b, 4) )

    alpha = [0.99, 0.95, 0.9, 0.75, 0.5]
    print("\n------ Value at Risk (VAR) calculated by weekly close price -----")
    print(100 * np.round(
        get_VaR(weekly.Close, alpha=alpha, scale=52), 4) )

    print("\nDividend annual yield and payment frequency per year")
    add_dividend(data, price='Close', adj='Adj Close', out='Dividend')
    div = get_dividend_yield(data,
        price='Close', div='Dividend', style='simple').rename("Dividend Yield (%)")
    div = np.round(div[-5:].to_frame()*100, 2)
    for c in div.index:
        div.at[c,'payment No.'] = np.sum(data[data.Dividend!=0].index.year==c)
    print(div)
    print('\nThe most recent 6 dividend payments')
    print(data[data.Dividend>0].Dividend[-6:].rename("Dividend per share").to_frame())

def ETF_describe():
    import tkinter as tk
    from datetime import date
    root = tk.Tk()
    ticker = "SPY"
    tk.Entry(root, textvariable=ticker).grid(row=1, column=1)
    start_date = "2000-1-1"
    tk.Entry(root, textvariable=start_date).grid(row=1, column=2)
    end_date = "{}".format(date.today())
    tk.Entry(root, textvariable=end_date).grid(row=1, column=3)
    frame = tk.Frame(root)
    ETF_describe_window(frame, ticker)
    frame.grid(row=2, column=1, columnspan=3)
    root.mainloop()


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Display a comprehensive analysis "
                                                 "of one ETF by ticker ")
    parser.add_argument('--ticker', type=str, \
                        help='ticker of the stock/ETF')
    parser.add_argument('--window', default=False, action='store_true', \
                        help='indicate whether to display in window')
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))

    if not FLAGS.ticker is None:
        if FLAGS.window:
            import tkinter as tk
            root = tk.Tk()
            ETF_describe_window(root, FLAGS.ticker)
            root.mainloop()
        else:
            ETF_describe_print(FLAGS.ticker)
    else:
        ETF_describe()
