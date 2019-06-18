import sys
import os
path = '\\'.join(os.path.abspath(__file__).split('\\')[:-2])
if not path in sys.path:
    sys.path.append(path)
file_dir = '\\'.join(os.path.abspath(__file__).split('\\')[:-1])

import tkinter as tk
import numpy as np
import pandas as pd
import logging
from datetime import date

from gui.tkinter_widget import entry_table, left_right_list, my_entry, my_scale, plot_embed
from invest.calculation import get_returns, add_dividend, get_return_vol, minimize_risk
from invest.get_data import update_ETF_data, read_ETF, read_portfolio
from invest.plot import return_vol, pie_plot
from time_series.functions import resample

def load_all_from_web():
    try:
        etfs = pd.read_csv("https://www.nasdaq.com/investing/etfs/etf-finder-results.aspx?download=Yes")
    except:
        alert = tk.Tk()
        tk.Label(alert, text='Failed to load ETFs from https://www.nasdaq.com').pack()
        alert.mainloop()
    else:
        etfs.to_csv("data/All_ETFs.csv", index=False)
    try:
        cpn = pd.read_csv("https://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download")
    except:
        alert = tk.Tk()
        tk.Label(alert, text='Failed to load companies traded at NASDAQ from https://www.nasdaq.com').pack()
        alert.mainloop()
    else:
        cpn.to_csv("data/All_nasdaq.csv", index=False)
    try:
        cpn = pd.read_csv("https://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nyse&render=download")
    except:
        alert = tk.Tk()
        tk.Label(alert, text='Failed to load companies traded at NYSE from https://www.nasdaq.com').pack()
        alert.mainloop()
    else:
        cpn.to_csv("data/All_nyse.csv", index=False)
    try:
        cpn = pd.read_csv("https://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=amex&render=download")
    except:
        alert = tk.Tk()
        tk.Label(alert, text='Failed to load companies traded at AMEX from https://www.nasdaq.com').pack()
        alert.mainloop()
    else:
        cpn.to_csv("data/All_amex.csv", index=False)

# 2019-5-17
class main_window(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self._logger_ = logging.getLogger(__name__)
        self._logger_.info("Initialize main window")

        self.option_add("*Font", '100')

        self._logger_.info("Root directory is: {}".format(file_dir))

        if 'data' not in os.listdir(file_dir):
            self._logger_.info("Create new directory under current folder: "+file_dir)
            os.mkdir('data')

        self.frames = {}
        for window in [holdings_window]: #[filter_window, portfolio_window, single_window, holdings_window]:
            frame = window(self)
            self.frames[window] = frame
            frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        # self.show_frame(holdings_window)

        menu = tk.Menu(self, tearoff=0)
        menu.add_cascade(label="Filter", command=lambda:self.show_frame(filter_window))
        menu.add_cascade(label="Portfolio", command=lambda:self.show_frame(portfolio_window))
        menu.add_cascade(label="Single", command=lambda:self.show_frame(single_window))
        menu.add_cascade(label="Holdings", command=lambda:self.show_frame(holdings_window))
        master['menu'] = menu
        self.pack()

    def show_frame(self, name):
        self._logger_.info("Bring up frame {}".format(name))
        self.frames[name].tkraise()


class filter_window(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self._logger_ = logging.getLogger(__name__)
        self._logger_.info("Initialize filter window")


class portfolio_window(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self._logger_ = logging.getLogger(__name__)
        self._logger_.info("Initialize portfolio window")

        tickers=['SPY','FDLO','USMV','MUB']

        self.select = left_right_list(self, [], tickers, command=self.update_plot,
                                      left_name='Available', right_name='Selected')
        self.select.left.listbox.config(width=10)
        self.select.right.listbox.config(width=10)
        self.select.grid(row=0, column=0, padx=10, pady=10)

        self.benchmark = my_entry(self, name='Benchmark', button_name='set', command=self.set_benchmark)
        self.benchmark.entry.config(width=10)
        self.benchmark.grid(row=1, column=0, padx=10, pady=10)
        tk.Label(self, text="Dow \nS&P 500: SPY\n").grid(row=2, column=0, padx=10, pady=10)

        fig, fig_pie = self.initial_plot()
        self.plot = plot_embed(self, fig)
        self.plot.grid(row=0, rowspan=3, column=1, columnspan=2, padx=10, pady=10)

        self.plot_pie = plot_embed(self, fig_pie)
        self.plot_pie.grid(row=3, rowspan=3, column=0)

        self.length = my_scale(self, name='Duration:', command=self.update_plot,
                               labels=['5Y','3Y','2Y','1.5Y','1Y','3Q','2Q','1Q','1M'],
                               values=[ 260, 156, 104,    78,  52,  39,  26,  13,   5],
                               default=104)
        self.length.scale.config(length=200)
        self.length.grid(row=3, column=1, pady=10, sticky=tk.W)

        self.end = my_scale(self, name='End Date:', command=self.update_plot,
                            labels=["{}".format(x.date()) for x in self._data_.index],
                            values=self._data_.index, default=self._data_.index[-1])
        self.end.scale.config(length=200)
        self.end.grid(row=4, column=1, pady=10, sticky=tk.W)


        from_, _to, step = 0, 0.2, 0.01
        returns = np.arange(from_,_to+0.0001,step)
        self.ret = my_scale(self, name='Targeted Return:', command=self.update_plot_pie,
                            labels=["{:.2%}".format(x) for x in returns],
                            values=returns, default=0.05)
        self.ret.scale.config(length=200)
        self.ret.grid(row=5, column=1, pady=10, sticky=tk.W)
        self.update_plot()

    def initial_plot(self):
        self._logger_.debug("Initialize plots for portfolio window")
        tickers = self.select.get_right()
        start, end = '2000-01-01', None
        self._data_ = get_returns(
            resample(
                read_portfolio(tickers, column='Close', start=start, end=end, keep_na=True),
                column=None, style="week", method='close'),
            style='simple', fillna=False)
        self._data_['benchmark'] = 0
        data = self._data_.iloc[-52:,:-1].dropna(axis=1, how='any')
        rv = get_return_vol(pd.concat([data*3,-data], axis=1),
                            scale=52, ret=True, plotit=False)
        fig = return_vol(rv.Return, rv.Volatility, rv.index)
        fig.axes[0].plot([0],[0], 'r*')
        return fig, pie_plot([10,6], labels=['a','b'])

    def update_plot_pie(self, x=None):
        self._logger_.debug("Update pie plot for portfolio window")
        index = self.ret.scale.get()
        if self._data_sharpe_[index] is None:
            ret = self.ret.get()
            sharpe = minimize_risk(self._data_current_, returns=[ret], strict=False,
                                   riskfree=None, max_alloc=1, scale=52, ret=True,
                                   verbose=False, plotit=False)
        else:
            sharpe = self._data_sharpe_[index]
        self.plot.axes.lines[1].set_data(sharpe.Volatility*100, sharpe.Return*100)
        self.plot.show()
        self.plot_pie.axes.clear()
        self.plot_pie.axes.pie(sharpe.iloc[0,:-2],
                               explode=np.append(0.1,np.zeros(sharpe.shape[1]-3)),
                               labels=sharpe.columns[:-2], autopct='%1.1f%%', shadow=True)
        self.plot_pie.show()

    def update_plot(self, x=None):
        tickers = list(self.select.get_right())
        length = self.length.get()
        end = self.end.get()
        ret = self.ret.get()
        self._logger_.debug("""Update plot with {} weeks ending on {}.
            Targeted return value is {}""".format(length, end, ret))
        data = self._data_.loc[:end,:][-length:]
        data = data[tickers].apply(lambda x: x-data.benchmark).dropna(axis=1, how='any')
        self._data_current_ = data
        rv = get_return_vol(data, scale=52, ret=True, plotit=False) * 100
        sharpe = minimize_risk(data, returns=None, strict=False, riskfree=None, max_alloc=1, scale=52,
                               ret=True, verbose=False, plotit=False)
        self.plot.axes.lines[0].set_data(rv.Volatility, rv.Return)
        self.plot.axes.texts.clear()
        for ticker in data.columns:
            self.plot.axes.text(rv.Volatility[ticker], rv.Return[ticker], ticker)
        self.plot.show()
        self._data_sharpe_ = [None for i in self.ret._val_]
        self.update_plot_pie()

    def set_benchmark(self):
        ticker = self.benchmark.get().upper()
        if ticker=="":
            self._logger_.info("Remove benchmark")
            self._data_['benchmark'] = 0
        else:
            self._logger_.info("Set benchmark as {}".format(ticker))
            try:
                data = read_ETF(ticker)
            except:
                self._logger_.error("Cannot load benchmark {}".format(ticker))
            else:
                self._data_['benchmark'] = get_returns(
                    resample(data.Close, column=None, style="week", method='close'),
                    style='simple', fillna=False)
        self.update_plot()


class single_window(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self._logger_ = logging.getLogger(__name__)
        self._logger_.info("Initialize single window")
        tk.Label(self, text="single").pack()

# 2019-5-??
class holdings_window(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self._logger_ = logging.getLogger(__name__)
        self._logger_.info("Initialize holdings window")

        if 'holdings.csv' in os.listdir(file_dir+'\\data'):
            self.main_frame()
        else:
            self._logger_.info("Create new file holdings.csv")
            frame = tk.Frame(self)
            tk.Label(frame, text="Start by entering your transaction history")\
                .grid(row=0, column=0, pady=10)
            self.add_transactions(frame)
            frame.pack()

    def main_frame(self):
        holdings = pd.read_csv(file_dir+'\\data\\holdings.csv', index_col=None, parse_dates=[0]).dropna()
        data = {}
        for t in holdings.Ticker.unique():
            data[t] = read_ETF(t)
            if (data[t].index[-1].date() != date.today()) and (date.today().weekday()<5):
                self._logger_.debug("Latest date is {}. Update local data for ticker {}"\
                    .format(data[t].index[-1].date(), t))
                update_ETF_data([t], dt="2000-01-01", file_dir=file_dir+'\\data\\')
            add_dividend(data[t], price='Close', adj='Adj Close', out='Dividend')

        def pop_up():
            self._logger_.info("Popup window")
            frame = tk.Tk()
            self.transactions(frame)
            frame.mainloop()
        tk.Button(self, text='Add Transactions', command=pop_up).pack(anchor=tk.E)

        fig = plot_embed(self)
        fig.pack()
        table = tk.Frame(self)
        table.pack()
        for i, text in enumerate(["Ticker","Buy Date","Buy Price","Current Price","Buy Shares",
                                  "Reinvested Shares","Capital Gain","Dividend Gain","Total Gain"]):
            tk.Label(table, text=text).grid(row=0, column=i, padx=5, pady=5)
        total_gain = 0
        for loc, i in enumerate(holdings.index):
            t = holdings.Ticker[i]
            d = holdings.Date[i]
            p = holdings.Price[i]
            s = holdings.Shares[i]
            r = (date.today()-d.date()).days / 365
            print(r)
            vals = [t, "{}".format(d.date()), "{:.2f}".format(p), "{:.2f}".format(data[t].iloc[-1,:].Close), "{:.0f}".format(s)]
            current_share = data[t].loc[d,'Close'] / data[t].loc[d,'Adj Close'] * s
            vals.append("{:.4f}".format(current_share - s))
            vals.append("{:.2f} ({:.2%})".format((data[t].iloc[-1,:].Close-p) * s, (data[t].iloc[-1,:].Close/p-1)/r))
            vals.append("{:.2f} ({:.2%})".format(data[t].loc[d:,'Dividend'].sum() * s, data[t].loc[d:,'Dividend'].sum()/p/r))
            gain = data[t].iloc[-1,:].Close * current_share - p*s
            total_gain += gain
            vals.append("{:.2f} ({:.2%})".format(gain, gain/p/s/r))
            for j, text in enumerate(vals):
                tk.Label(table, text=text).grid(row=loc+1, column=j, padx=5, pady=5)
        tk.Label(table, text="Total gain is: {:.2f}".format(total_gain))\
            .grid(row=holdings.shape[0]+1, column=0, columnspan=3)
        # print(holdings)
        # print(holdings.dtypes)


    def add_transactions(self, frame):
        self.trans_frame = frame
        bg = frame.cget('bg')
        self.trans = pd.DataFrame([[tk.StringVar() for _ in range(4)]],
            columns=['Date\nyyyy-mm-dd','Ticker','Shares','Price'],
            index=[0])
        entry_table(frame, self.trans, width=10, show_index=False, bg=bg).grid(row=1, column=0)
        self.add = tk.Button(frame, text='Add Row', command=self.add_row)
        self.add.grid(row=2, column=0)
        tk.Button(frame, text="Save", command=self.save_trans).grid(row=1, column=1)

    def add_row(self):
        bg = self.trans_frame.cget('bg')
        row = self.trans.shape[0]
        new_row = pd.DataFrame([[tk.StringVar() for _ in range(4)]],
                               index=[row], columns=self.trans.columns)
        self.trans = pd.concat([self.trans, new_row], axis=0)
        entry_table(self.trans_frame, self.trans.iloc[-1:,:], width=10,
                    show_index=False, show_column=False, bg=bg).grid(row=row+1, column=0)
        self.add.grid(row=row+2, column=0)

    def save_trans(self):
        data = pd.DataFrame("", index=self.trans.index, columns=self.trans.columns)
        for i in self.trans.index:
            for j in self.trans.columns:
                data.at[i,j] = self.trans.at[i,j].get()
        data.rename(columns={'Date\nyyyy-mm-dd':'Date'}, inplace=True)
        if 'holdings.csv' in os.listdir(file_dir+'\\data'):
            current = pd.read_csv(file_dir+'\\data\\holdings.csv', index_col=None, dtype=str)
            data = pd.concat([current, data], axis=0)
        data.dropna().to_csv(file_dir+"\\data\\holdings.csv", index=False)
        self.trans_frame.destroy()
        self.main_frame()

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="APP for investment")
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))

    main_window(tk.Tk()).mainloop()
