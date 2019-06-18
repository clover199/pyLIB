from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import logging

"""
This file defines useful helper functions for invest calculations.
"""


def convert_time(start, end):
    """
    Convert start and end time from string to datetime.
    If end is None, return current time
    """
    from datetime import datetime
    try:
        start = datetime.strptime(start, '%Y-%m-%d')
    except:
        pass
    if end is None:
        end = datetime.now()
    else:
        try:
            end = datetime.strptime(end, '%Y-%m-%d')
        except:
            pass
    return start, end


if __name__=="__main__":
    import argparse
    import functions
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Useful functions")
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
