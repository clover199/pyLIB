
import sys
import os
path = '\\'.join(os.path.abspath(__file__).split('\\')[:-3])
if not path in sys.path:
    sys.path.append(path)

import tkinter as tk
import numpy as np
import pandas as pd
import logging
from datetime import date

from gui.tkinter_widget import left_right_list, plot_embed, my_scale, my_entry
from invest.calculation import get_returns, get_return_vol, minimize_risk
from invest.get_data import update_ETF_data, read_ETF, read_portfolio
from invest.plot import return_vol, pie_plot
from time_series.functions import resample

# 2019-7-10
class portfolio_window(tk.Frame):
    def __init__(self, master, file):
        """
        input:  file    a csv file containing all tickers
                        should include a column: Ticker
        """
        super().__init__(master)
        self._logger_ = logging.getLogger(__name__)
        self._logger_.debug("Initialize portfolio window")

        try:
            data = pd.read_csv(file, index_col=None).dropna()
        except Exception as err:
            self._logger_.error("Cannot open tickers file {}".format(file))
            self._logger_.error(err)
            return
        self._logger_.debug("input data columns: {}".format(data.columns))
        tickers = np.unique(data.Ticker.values)

        self.select = left_right_list(self, [], tickers, command=self.update_plot,
                                      left_name='Available', right_name='Selected')
        self.select.left.listbox.config(width=10)
        self.select.right.listbox.config(width=10)
        self.select.grid(row=0, column=0, padx=10, pady=10)

        self.benchmark = my_entry(self, name='Benchmark', button_name='set', command=self.set_benchmark)
        self.benchmark.entry.config(width=10)
        self.benchmark.grid(row=1, column=0, padx=10, pady=10)

        fig, fig_pie = self.initial_plot()
        self.plot = plot_embed(self, fig)
        self.plot.grid(row=0, rowspan=3, column=1, columnspan=2, padx=10, pady=10)

        self.plot_pie = plot_embed(self, fig_pie)
        self.plot_pie.grid(row=3, rowspan=3, column=0)

        self.length = my_scale(self, name='Duration:', command=self.update_plot,
                               labels=['5Y','3Y','2Y','1.5Y','1Y','3Q','2Q','1Q','12W','11W','10W','9W','8W','7W','6W','1M'],
                               values=[ 260, 156, 104,    78,  52,  39,  26,  13,   12,   11,   10,   9,   8,   7,   6,   5],
                               default=52)
        self.length.scale.config(length=200)
        self.length.grid(row=3, column=1, pady=10, sticky=tk.W)

        self.end = my_scale(self, name='End Date:', command=self.update_plot,
                            labels=["{}".format(x.date()) for x in self._data_.index],
                            values=self._data_.index, default=self._data_.index[-1])
        self.end.scale.config(length=200)
        self.end.grid(row=4, column=1, pady=10, sticky=tk.W)

        from_, _to, step = 0, 0.4, 0.01
        returns = np.arange(from_, _to+1e-6, step)
        self.ret = my_scale(self, name='Targeted Return:', command=self.update_plot_pie,
                            labels=["{:.2%}".format(x) for x in returns],
                            values=returns, default=0.05)
        self.ret.scale.config(length=200)
        self.ret.grid(row=5, column=1, pady=10, sticky=tk.W)
        self.update_plot()

        tk.Button(self, text='update data', command=self.update_data).grid(row=0,column=3)

    def initial_plot(self, column='Adj Close', style='week', start='2015-1-1'):
        self._logger_.debug("Initialize plots for portfolio window")
        tickers = self.select.get_right()
        self._data_ = get_returns(
            resample(
                read_portfolio(tickers, column=column, start=start),
                column=None, style=style, method='close'),
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
        self.plot.fig.axes[0].lines[1].set_data(sharpe.Volatility*100, sharpe.Return*100)
        self.plot.show()
        self.plot_pie.fig.axes[0].clear()
        self.plot_pie.fig.axes[0].pie(sharpe.iloc[0,:-2],
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
        self.plot.fig.axes[0].lines[0].set_data(rv.Volatility, rv.Return)
        self.plot.fig.axes[0].texts.clear()
        for ticker in data.columns:
            self.plot.fig.axes[0].text(rv.Volatility[ticker], rv.Return[ticker], ticker)
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
                data = read_ETF(ticker, file_dir=path+"\\data_temp")
            except:
                self._logger_.error("Cannot load benchmark {}".format(ticker))
            else:
                self._data_['benchmark'] = get_returns(
                    resample(data.Close, column=None, style="week", method='close'),
                    style='simple', fillna=False)
        self.update_plot()

    def update_data(self):
        for t in self._data_.columns:
            update_ETF_data(t)
        self.initial_plot()
        self.update_plot_pie()


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="APP for investment")
    parser.add_argument('tickers', type=str,\
                        help='the csv file with tickers.\n Should have a column: Ticker')
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))

    root = tk.Tk()
    window = portfolio_window(root, FLAGS.tickers)
    window.pack()
    root.mainloop()
