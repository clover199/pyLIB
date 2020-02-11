from __future__ import absolute_import, division, print_function
import numpy as np
from pandas._libs.tslib import Timestamp as pdtime
import logging

"""
This file defines useful helper functions for invest calculations.
"""

# 2019-6-17
def convert_time(start=None, end=None):
    """
    Convert start and end time from string to datetime.
    input:  start   start date in string format YYYY-MM-DD
                    If None (default), return 2000-01-01
            end     end date in string format YYYY-MM-DD
                    If None (default), return current datetime
    return converted start, end
    """
    logger = logging.getLogger(__name__)
    from datetime import datetime, date
    if start is None:
        start = datetime(2000,1,1)
    elif type(start) not in [date, datetime, pdtime]:
        try:
            start = datetime.strptime(start, '%Y-%m-%d')
        except:
            logger.error("Cannot convert start: {} as type {}".format(
                start, type(start)))
    if end is None:
        end = datetime.now()
    elif type(end) not in [date, datetime, pdtime]:
        try:
            end = datetime.strptime(end, '%Y-%m-%d')
        except:
            logger.error("Cannot convert end: {} as type {}".format(
                end, type(end)))
    return start, end


if __name__=="__main__":
    import argparse
    import useful
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Useful functions")
    parser.add_argument('--list', type=bool, nargs='?', const=1, default=0, \
                        help='list all available modules')
    parser.add_argument('--doc', type=str, \
                        help='print documents of given function')
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.list:
        for x in getmembers(useful, isfunction):
            print(x[0])
        exit()
    if not FLAGS.doc is None:
        print(getattr(useful, FLAGS.doc).__doc__)
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))
