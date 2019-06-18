from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import logging

"""
This file defines different investment strategies. Every strategy has methods
init:   Inputs are a pandas Series or 1D array and other parameters (optional).
        The training process is completed in this method
step:   Inputs are current price and strategy parameters.
        Output an indicator buy (1), sell (-1) or no action (0)
"""


class single_linear_trend:
    """
    Buy when price start to increase and cell when price start to decrease.
    Linear regression is used to calculate the trend.
    paras:  sigs    the shreshold number of sigmas for buying/selling, default 5
                    lower means more likely to triger buy/sell
    """
    def __init__(self, data):
        self.n = len(data) + 1
        x = np.arange(self.n)
        self.x_mean = np.mean(x)
        self.x_SS = np.sum((x-self.x_mean)**2)
        self.y_sum = np.sum(data)
        self.y2_sum = np.sum(data**2)
        self.xy = np.dot(x[:-1], data)
        self.data = data.values * 1
        self.loc = 0

    def step(self, price, sigs=5):
        prev_price = self.data[self.loc]
        self.data[self.loc] = price
        self.loc =  (self.loc + 1) % len(self.data)
        self.y_sum += price
        self.y2_sum += price**2
        self.xy += (self.n-1) * price
        slope = (self.xy - self.x_mean*self.y_sum) / self.x_SS
        std = np.sqrt( (self.y2_sum - self.y_sum**2/self.n - slope * (self.xy-self.x_mean*self.y_sum)) \
                      / self.x_SS / (self.n-2) )
        out = 0
        if slope/std>sigs:
            out = 1
        elif slope/std<-sigs:
            out = -1
        self.y_sum -= prev_price
        self.y2_sum -= prev_price**2
        self.xy -= self.y_sum
        return out


class single_linear_range:
    """
    Buy/sell when price is above/below calclated range box.
    The range is calculated by mean and standard deviation
    paras:  sigs    the shreshold number of sigmas for buying/selling, default 3
                    lower means more likely to triger buy/sell
    """
    def __init__(self, data):
        self.data = data.values * 1
        self.loc = 0
        self.n = len(data)
        self.mean = np.mean(data)
        self.std = np.mean( (data-self.mean)**2 ) * self.n / (self.n-1)
        self.std = np.sqrt(self.std)

    def step(self, price, sigs=3):
        prev_price = self.data[self.loc]
        self.data[self.loc] = price
        self.loc =  (self.loc + 1) % self.n
        out = 0
        if price > self.mean + sigs * self.std:
            out = 1
        elif price < self.mean - sigs * self.std:
            out = -1
        if out!=0:
            self.mean = np.mean(self.data)
            self.std = np.mean( (self.data-self.mean)**2 ) * self.n / (self.n-1)
            self.std = np.sqrt(self.std)
        return out


def visualize(data, strategy, train_size,  **kwarg):
    """
    A function that visualize the strategy by applying to given data.
    input:  data        a pandas Series or 1D array
            strategy    the name of the strategy. Must have methods step
            train_size  the number of data needed for initial training
            other keywards passed to strategy step method
    """
    n = len(data)
    stg = strategy(data[:train_size])
    hold = False
    value = 100
    share = 0
    buy = []
    sell = []
    for i in range(train_size, n):
        act = stg.step(data[i], **kwarg)
        if hold and act==-1:
            sell.append(i)
            hold = False
            value = share * data[i]
        elif (not hold) and act==1:
            buy.append(i)
            hold = True
            share = value / data[i]
    if hold:
        sell.append(n-1)
        value = share * data[-1]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(14,3))
    plt.title("Grey: training period,   Green: holding period")
    plt.plot(range(n), data, '.-')
    plt.axvspan(-0.5, train_size-0.5, color='grey', alpha=0.5)
    for i in range(len(sell)):
        plt.axvspan(buy[i], sell[i], color='green', alpha=0.5)
    plt.xlim(0, n-1)
    try:
        plt.xticks(range(n)[::n//7], data.index[::n//7].date)
    except:
        pass
    plt.show()
    print('Total return with reinvest {:.2f}'.format(value-100))
    buyat = np.array([data[i] for i in buy])
    sellat = np.array([data[i] for i in sell])
    ret = np.sum( (sellat - buyat) / buyat )
    print('Total gain with fixed initial value {:.2f}'.format(ret*100))


def visualize2(data, strategy, buy_size, sell_size, buy_kwarg, sell_kwarg):
    """
    A function that visualize the strategy by applying to given data.
    The buying and selling training size and parameters are set seperately
    input:  data        a pandas Series or 1D array
            strategy    the name of the strategy. Must have methods step
            buy_size    the number of data needed for initial training for buying
            sell_size   the number of data needed for initial training for selling
            buy_kwarg   other parameters for buying
            sell_kwarg  other parameters for selling
    """
    n = len(data)
    stg = strategy(data[:buy_size])
    hold = False
    value = 100
    share = 0
    buy = []
    sell = []
    for i in range(buy_size, n):
        if hold:
            act = stg.step(data[i], **sell_kwarg)
            if act==-1:
                sell.append(i)
                hold = False
                value = share * data[i]
                if sell_size!=buy_size:
                    stg = strategy(data[i-sell_size+1:i+1])
        else:   # not hold
            act = stg.step(data[i], **buy_kwarg)
            if act==1:
                buy.append(i)
                hold = True
                share = value / data[i]
                if sell_size!=buy_size:
                    stg = strategy(data[i-buy_size+1:i+1])
    if hold:
        sell.append(n-1)
        value = share * data[-1]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(14,3))
    plt.title("Grey: training period,   Green: holding period")
    plt.plot(range(n), data, '.-')
    plt.axvspan(-0.5, buy_size-0.5, color='grey', alpha=0.5)
    for i in range(len(sell)):
        plt.axvspan(buy[i], sell[i], color='green', alpha=0.5)
    plt.xlim(0, n-1)
    try:
        plt.xticks(range(n)[::n//7], data.index[::n//7].date)
    except:
        pass
    plt.show()
    print('Total return with reinvest {:.2f}'.format(value-100))
    buyat = np.array([data[i] for i in buy])
    sellat = np.array([data[i] for i in sell])
    ret = np.sum( (sellat - buyat) / buyat )
    print('Total gain with fixed initial value {:.2f}'.format(ret*100))


if __name__=="__main__":
    import argparse
    import functions
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Investment strategies")
    parser.add_argument('--list', type=bool, nargs='?', const=1, default=0, \
                        help='list all available modules')
    parser.add_argument('--doc', type=str, \
                        help='print documents of given function')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.list:
        for x in getmembers(functions, isfunction):
            print(x[0])
        exit()
    if not FLAGS.doc is None:
        print(getattr(functions, FLAGS.doc).__doc__)
