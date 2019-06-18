from __future__ import absolute_import, division, print_function
import numpy as np
import logging

"""
This file defines some useful functions for easier printing or debuging.
"""

def best_time(t):
    """
    Returns a string of time in a proper format
    input:  t   time in seconds
    output:     string
    """
    if t<60:
        return "{:.2f} s".format(t)
    elif t<3600:
        return "{:.2f} min".format(t/60)
    else:
        return "{:.2f} h".format(t/3600)


def progress_bar(vals, func, disable=False, **kwarg):
    """
    A useful tool to generate progress bar
    input:  vals    the values to iterate
            func    the function that processes the value.
                    Must return a string or nothing
            disable indicate whether to disable progress_bar, default False
            kwarg   other arguments for function
    """
    logger = logging.getLogger(__name__)
    from time import time
    begin = time()
    n = len(vals)
    output = ""
    if not disable:
        print('-'*100, end='\r', flush=True)
    for i, v in enumerate(vals):
        result = func(v, **kwarg)
        if not (result is None):
            output = output + result
        l = int(100*(i+1)/n)
        if not disable:
            print(l*'#'+(100-l)*'-' + " {:.2f} min".format((time()-begin)/60), end='\r', flush=True)
    if not disable:
        print("\nTotal time used:", best_time(time()-begin))
        print(output)


if __name__=="__main__":
    import argparse
    import useful
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Some general helpful functions")
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
