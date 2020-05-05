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


# 2020-2-14
def ETF_describe_window(frame, ticker='SPY', start=None, end=None):
    """ Display a comprehensive analysis in windows window """
    import tkinter as tk
    from invest.get_data import get_latest_ETF
    from invest.calculation import get_alpha_beta, get_return_vol
    from invest.calculation import add_dividend, get_dividend_yield
    from time_series.functions import resample
    from gui.tkinter_widget import display_dataframe

    for widget in frame.winfo_children():
        widget.destroy()

    data = get_latest_ETF(ticker, start=start)
    if end is not None:
        data = data[:end]
    tk.Label(frame, text=ticker, font=("Helvetica", 14))\
        .grid(row=1, column=1, padx=10, pady=5)
    tk.Label(frame,
        text="from \n{} \nto \n{}".format(data.index[0].date(),
                                          data.index[-1].date())
    ).grid(row=1, column=2, padx=10, pady=5)
    df = np.round( data.iloc[-1,:-1].to_frame().T, 2)
    df.index = [x.date() for x in df.index]
    df.index.name = "Today:"
    display_dataframe(frame, df).grid(row=3, column=1, columnspan=2, padx=2, pady=2)

    # Alpha and Beta are calculated by using SPY as market and zero risk-free rate
    # Weekly close price is used
    market = get_latest_ETF('SPY')
    wm = resample(market, style="week", method='close')
    weekly = resample(data, style="week", method='close')
    a, b = get_alpha_beta(weekly.Close, wm.Close, risk_free=0, scale=52, dspl=False)
    tk.Label(frame, text="Alpha: {}\nBeta: {}".format(np.round(a, 4), np.round(b, 4)))\
        .grid(row=4, column=1, padx=2, pady=2)

    # Return and volatility are calculated by using weekly close price
    rv = get_return_vol(weekly.Close, scale=52)
    tk.Label(frame, text="Return: {:.2%}\nVolatility: {:.2%}".format(
        rv.Return[0], rv.Volatility[0])).grid(row=4, column=2, padx=2, pady=2)

    add_dividend(data, price='Close', adj='Adj Close', out='Dividend')
    div = get_dividend_yield(data,
        price='Close', div='Dividend', style='simple').rename("Dividend Yield (%)")
    div = np.round(div[-5:].to_frame()*100, 2)
    for c in div.index:
        div.at[c,'Payment No.'] = np.sum(data[data.Dividend!=0].index.year==c)
    div.index.name = "Year"
    display_dataframe(frame, div).grid(row=5, column=1, padx=2, pady=2)
    divs = data[data.Dividend>0].Dividend[-5:].rename("Dividend per share").to_frame()
    divs.index = [d.date() for d in divs.index]
    divs.index.name = "Date"
    display_dataframe(frame, divs).grid(row=5, column=2, padx=2, pady=2)
    return frame

def ETF_describe():
    import tkinter as tk
    from datetime import date
    root = tk.Tk()
    ticker = tk.StringVar()
    ticker.set("SPY")
    tk.Entry(root, textvariable=ticker).grid(row=1, column=1, padx=2, pady=2)
    start_date = tk.StringVar()
    start_date.set("2000-1-1")
    tk.Entry(root, textvariable=start_date).grid(row=1, column=2, padx=2, pady=2)
    end_date = tk.StringVar()
    end_date.set( "{}".format(date.today()) )
    tk.Entry(root, textvariable=end_date).grid(row=1, column=3, padx=2, pady=2)
    frame = tk.Frame(root)
    tk.Button(root, text='Show',
        command=lambda: ETF_describe_window(frame, ticker.get(),
                                            start_date.get(), end_date.get())).\
        grid(row=1, column=4, padx=2, pady=2)

    tk.Button(root, text='Display in new window',
        command=lambda: ETF_describe_window(tk.Tk(), ticker.get(),
                                start_date.get(), end_date.get()).mainloop()
    ).grid(row=1, column=5, padx=2, pady=2)
    frame.grid(row=2, column=1, columnspan=5, padx=2, pady=2)
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
