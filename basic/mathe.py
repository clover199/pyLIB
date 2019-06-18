from __future__ import absolute_import, division, print_function
import numpy as np
import logging

"""
This file defines some basic functions for math and statistics problems.
"""


def commute(A, B, tol=1e-10):
    """
    Tell whether matrices A and B commute with each other
    input:  A   matrix
            B   matrix
            tol tolerance, default 1e-10
    output:     bool indicating AB=?=BA
    """
    A = np.array(A)
    B = np.array(B)
    return np.mean(np.abs(A.dot(B)-B.dot(A)))<tol


def inverse(M, tol=1e-6, return_df=False):
    """
    Calculate inverse of hermitian matrix M with or without full rank
    input:  M   a hermitian matrix
            tol tolorance for zero
            return_p: indicates whether to return degree of freedom
    output      inverse of M
    """
    logger = logging.getLogger(__name__)
    logger.info("------ inverse(M, tol=1e-6, return_df=False) ------")
    logger.debug("Input matrix M type: {}".format(type(M)))
    logger.debug("Input matrix M size: {}".format(np.array(M).shape))
    logger.debug("Input tol: {}".format(tol))
    logger.debug("Input return_df: {}".format(return_df))
    D, U = np.linalg.eigh(M)
    D = np.where(np.abs(D)>tol, 1/D, 0)
    if return_df:
        return (U*D).dot(U.conjugate().T), sum(np.abs(D)>0)
    else:
        return (U*D).dot(U.conjugate().T)


def block_inverse(iA, B, C, D):
    """
    Calculate inverse of matrix [[A,B],[C,D]] given inverse of A
    input:  iA  a n-by-n numpy array, inverse of A
            B   a 1D numpy array with length n or n-by-1 numpy array
            C   a 1D numpy array with length n or 1-by-n numpy array
            D   a number
    """
    logger = logging.getLogger(__name__)
    logger.info("------- block_inverse(iA, B, C, D) -------")
    logger.debug("Input matrix types are: {} {} {} {}".format(type(iA), type(B),
                                                              type(C), type(D)))
    logger.debug("Input matrix shapes are: {} {} {} {}".format(
        np.array(iA).shape,
        np.array(B).shape,
        np.array(C).shape,
        np.array(D).shape))
    n = iA.shape[0]
    result = np.zeros([n+1, n+1])
    CA = np.dot(C, iA).reshape([1,n])
    AB = np.dot(iA, B).reshape([n,1])
    det = D-CA.dot(B)
    result[:n,:n] = iA + AB.dot(CA) / det
    result[n:n+1,:n] = -CA / det
    result[:n,n:n+1] = -AB / det
    result[n,n] = 1 / det
    return result

# 2019-5-17
def covariance(x, y=None):
    """
    calculate covariance of x and y.
    input:  x   1D array
            y   1D array, default to be the same as x
    x and y should be of the same length
    Return a number
    """
    n = len(x)
    if n==0:
        return 0
    elif n==1:
        n = 2
    if y is None:
        return np.sum((np.array(x)-np.mean(x))**2) / (n-1)
    return np.dot(np.array(x)-np.mean(x), np.array(y)-np.mean(y)) / (n-1)


def correlation(x, y):
    """
    calculate correlation of x and y.
    input:  x   1D array
            y   1D array
    x and y should be of the same length
    """
    mx = np.mean(x)
    my = np.mean(y)
    return np.dot(x-mx, y-my) / np.sqrt(np.sum((x-mx)**2)) / np.sqrt(np.sum((y-my)**2))


def check_normality(data, bins=100):
    """
    Calculate simple and log returns for the input data
    Return a DataFrame with two columns: SimpleReturn, LogReturn
    input:  data    a Series data or numpy array
            bins    number of bins used in histogram plot
    """
    try:
        data = data.dropna().values
    except:
        pass

    from scipy import stats
    n = len(data)
    mean, std = stats.norm.fit(data)
    df = stats.t.fit(data)
    lim = np.max(np.abs(data))
    x = np.linspace(-lim, lim, 1000)
    y1 = stats.norm.pdf(x, mean, std) * 2*lim/bins*n
    y2 = stats.t.pdf(x, *df) * 2*lim/bins*n

    import matplotlib.pyplot as plt
    plt.figure(figsize=(lim/std, 3))
    plt.hist(data, bins=bins, range=[-lim,lim])
    plt.plot(x, y1, '-', label='Normal')
    plt.plot(x, y2, '-', label='Student t df={:.2f}'.format(df[0]))
    plt.legend(loc='best')
    plt.ylabel('Density')

    (theory, sample), (slope, intercept, r) = stats.probplot(data, dist="norm")
    x = np.linspace(min(theory), max(theory), 1000)
    y = slope * x + intercept

    plt.figure(figsize=(4,4))
    plt.title("Normal Q−Q Plot ")
    plt.plot(theory, sample, '.')
    plt.plot(x, y, '-')
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.show()


def pwl_value(xs, ys, px):
    """
    Return value of a piece-wise linear function represented by points (x,y)
    input:  xs  1D array, x-axis value of points
            ys  1D array, y-axis value of points
            px  the x-axis value of the point
    xs should be non-decreasing
    xs and ys should be of the same length
    """
    logger = logging.getLogger(__name__)
    logger.info("------- pwl_value(xs, ys, px) -------")
    logger.debug("Input xs, ys shapes are {} {}".format(np.array(xs).shape,
                                                        np.array(ys).shape))
    logger.debug("Input point x-axis px is {}".format(px))
    num = np.sum(np.array(xs)<px)
    if num==0:
        return ys[0] + (px-xs[0]) * (ys[1]-ys[0]) / (xs[1]-xs[0])
    if num==len(xs):
        if xs[-2]==xs[-1]:
            return ys[-1]
        return ys[-1] + (px-xs[-1]) * (ys[-2]-ys[-1]) / (xs[-2]-xs[-1])
    if xs[num]==px:
        return ys[num]
    return ys[num] + (px-xs[num]) * (ys[num-1]-ys[num]) / (xs[num-1]-xs[num])


def pwl_derivative(xs, ys):
    """
    Calculate first derivative of a piece-wise linear function represented by points (x,y)
    input:  xs  1D array, x-axis value of points
            ys  1D array, y-axis value of points
    x and y should be of the same length
    Return xs', ys' of a new pwl function
    """
    logger = logging.getLogger(__name__)
    logger.info("------- pwl_derivative(xs, ys) -------")
    logger.debug("Input xs, ys shapes are {} {}".format(np.array(xs).shape,
                                                        np.array(ys).shape))
    x = np.array(xs)
    y = np.array(ys)
    slope = (y[1:]-y[:-1]) / (x[1:]-x[:-1])
    return np.vstack([x[:-1],x[1:]]).T.flatten(), np.repeat(slope, 2)


if __name__=="__main__":
    import argparse
    import mathe
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Basic math and statistics functions")
    parser.add_argument('--list', type=bool, nargs='?', const=1, default=0, \
                        help='list all available modules')
    parser.add_argument('--doc', type=str, \
                        help='print documents of given function')
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.list:
        for x in getmembers(mathe, isfunction):
            print(x[0])
        exit()
    if not FLAGS.doc is None:
        print(getattr(mathe, FLAGS.doc).__doc__)
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))
