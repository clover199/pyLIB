from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import logging

"""
This file define functions to generate features for statistics
analysis and machine learning.
"""

def generate(tickers, features):
    """
    input:  tickers     a list of tickers of stocks/ETFs
            features    a list of feature names to be calculated
    Return a pandas DataFrame
    """
    from invest.get_data import read_ETF
    from time_series.functions import resample
    from invest.calculation import get_returns
    data = pd.DataFrame()
    for t in tickers:
        etf = read_ETF(t)
        weekly = resample(etf, style='week', method='close')
        ret = get_returns(weekly, style='simple')
        rv = get_return_vol(ret, scale=52, ret=True, plotit=False)


if __name__=="__main__":
    import argparse
    import functions
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Functions for generating features")
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
