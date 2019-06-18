from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import logging

"""
This file defines functions for basic time series models
"""


def ARMA_helper(cors):
    """ Calculate thetas from autocorrelations """
    logger = logging.getLogger(__name__)
    from scipy.linalg import solve_triangular
    n = len(cors)
    if n==0:
        return []
    thetas = np.random.random(n)
    for _ in range(100):
        matrix = np.zeros([n,n])
        for i in range(1,n):
            matrix[i-1, i:] = thetas[:n-i]
        thetas = solve_triangular(matrix, cors, lower=False, unit_diagonal=True)
    return thetas


class ARMA:
    """
    ARMA model
    x_t = phi_0 + sum_p phi_p*x_t-p + theta_0*w_t + sum_q theta_q*w_t-q
    ARMA(p=0, q=0)
    functions:      fit(data):      data must be a 1D array
                    summary():      prints a summary of the model
    attributes:     var     variance ratio
                    arg     arguments of kept features
                    tol_    tolerance
    """
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q

    def fit(self, data):
        self.n = len(data) - self.p
        x, y = get_xy(data, self.p, intercept=True)

        xtx = x.T.dot(x)
        xy = x.T.dot(y)
        inv = np.linalg.inv(xtx)
        self.phi = inv.dot(xy)

        self.residual = x.dot(self.phi) - y
        mean = np.mean(self.residual)
        assert np.abs(mean)<1e-6, "Mean of y and y_hat are not the same"
        ac = auto_correlation(self.residual, lag=self.q)
        df = self.n-self.p-1
        self.var = np.sum( self.residual**2 ) / df
        self.theta = np.append(1, ARMA_helper(ac[1:]))
        self.theta = self.theta * np.sqrt( self.var / np.sum(self.theta**2) )

        ac = ac * self.var    # auto-covariance
        self.sigma = ac[0] * x.T.dot(x)
        for i in range(1, 1+self.q):
            self.sigma += 2 * ac[i] * x[i:,:].T.dot(x[:-i,:])
        self.sigma = inv.dot(self.sigma).dot(inv)

        if self.p>0:
            TSS = np.sum( (y-np.mean(y))**2 )
            self.rsquare = 1 - self.var * df / TSS
            self.adjrsquare = 1 - self.var / TSS * (self.n-1)
            self.F = ( TSS - self.var*df) / self.p / self.var

    def summary(self):
        import pandas as pd
        from scipy import stats
        index = []
        for i in range(1+self.p):
            index.append('AR{}'.format(i))
        for i in range(1+self.q):
            index.append('MA{}'.format(i))
        coefs = pd.DataFrame(1e-6, columns=['coef','std err','t','p>|t|', ''], index=index)
        coefs.iloc[:1+self.p,0] = self.phi
        coefs.iloc[1+self.p:,0] = self.theta
        coefs.iloc[:1+self.p,1] = np.sqrt(np.diag(self.sigma))
        coefs.iloc[:,2] = coefs.iloc[:,0]/coefs.iloc[:,1]
        coefs.iloc[:,3] = 2*stats.t.cdf(-np.abs(coefs.iloc[:,2]), self.n-self.p-1)
        coefs = coefs.astype({'':str})
        coefs[''] = ' '
        for i, v in enumerate(coefs.iloc[:1+self.p,3]):
            if v>0.1: coefs.iat[i,4] = ''
            elif v>0.05: coefs.iat[i,4] = '.'
            elif v>0.01: coefs.iat[i,4] = '*'
            elif v>0.001: coefs.iat[i,4] = '**'
            else: coefs.iat[i,4] = '***'
        display(np.round(coefs,4))
        print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")
        print("Residual standard error: {:.4f} on {:d} degrees of freedom".format(
            np.sqrt(self.var), self.n-self.p-1))
        if self.p>0:
            print('R-squared: {:.5f},'.format(self.rsquare),
                  'Adjusted R-squared: {:.5f}'.format(self.adjrsquare))
            p = 1 - stats.f.cdf(self.F, self.p, self.n-self.p-1)
            print('F-statistic: {:.4g} on {:d} and {:d} DF, p-value: {:.4g}'.format(
                self.F, self.p, self.n-self.p-1, p))

    def ARCH(p=1):
        x, y = get_xy(p, self.residual)
        from Msklearn import LinearRegression
        lm = LinearRegression(intercept=True)
        lm.fit(x, y)
        display(lm.summary())


class GARCH:
    """
    Generalized Autoregressive Conditional Heteroskedasticity
    x_t = sig_t * w_t
    s^2_t = a_0 + sum_p b_p * s^2_t-p + sum_q a_q * x^2_t-q
    ARCH(q) is just GARCH(p=0, q)
    GARCH(p=0, q=1)
    functions:      fit(data):      data must be a 1D array
                    summary():      prints a summary of the model
                    ARCH(p=1):      calculate time dependent variance
                    plot(X, y, residual=True):
                                    X must be a 1D or 2D array
                                    y is a 1D array
                                    residual indicates whether to make residual plot
    attributes:     var     variance ratio
                    arg     arguments of kept features
                    tol_    tolerance
    """
    def __init__(self, p=0, q=1):
        self.p = p
        self.q = q

    def fit(self, data):
        self.n = len(data) - self.q
        x, y = get_xy(data**2, self.q, intercept=True)

        xtx = x.T.dot(x)
        xy = x.T.dot(y)
        inv = np.linalg.inv(xtx)
        self.alpha = inv.dot(xy)

        self.residual = x.dot(self.phi) - y
        mean = np.mean(self.residual)
        assert np.abs(mean)<1e-6, "Mean of y and y_hat are not the same"
        df = self.n-self.q-1
        self.var = np.sum( self.residual**2 ) / df
        self.sigma = self.var * x.T.dot(x)

        if self.q>0:
            TSS = np.sum( (y-np.mean(y))**2 )
            self.rsquare = 1 - self.var * df / TSS
            self.adjrsquare = 1 - self.var / TSS * (self.n-1)
            self.F = ( TSS - self.var*df) / self.p / self.var

    def summary(self):
        import pandas as pd
        from scipy import stats
        index = []
        for i in range(1+self.q):
            index.append('alpha{}'.format(i))
        for i in range(1+self.p):
            index.append('beta{}'.format(i))
        coefs = pd.DataFrame(1e-6, columns=['coef','std err','t','p>|t|', ''], index=index)
        coefs.iloc[:1+self.q,0] = self.alpha
        coefs.iloc[1+self.q:,0] = self.beta
        coefs.iloc[:1+self.q,1] = np.sqrt(np.diag(self.sigma))
        coefs.iloc[:,2] = coefs.iloc[:,0]/coefs.iloc[:,1]
        coefs.iloc[:,3] = 2*stats.t.cdf(-np.abs(coefs.iloc[:,2]), self.n-self.q-1)
        coefs = coefs.astype({'':str})
        coefs[''] = ' '
        for i, v in enumerate(coefs.iloc[:1+self.q,3]):
            if v>0.1: coefs.iat[i,4] = ''
            elif v>0.05: coefs.iat[i,4] = '.'
            elif v>0.01: coefs.iat[i,4] = '*'
            elif v>0.001: coefs.iat[i,4] = '**'
            else: coefs.iat[i,4] = '***'
        display(np.round(coefs,4))
        print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")
        print("Residual standard error: {:.4f} on {:d} degrees of freedom".format(
            np.sqrt(self.var), self.n-self.q-1))
        if self.q>0:
            print('R-squared: {:.5f},'.format(self.rsquare),
                  'Adjusted R-squared: {:.5f}'.format(self.adjrsquare))
            p = 1 - stats.f.cdf(self.F, self.q, self.n-self.q-1)
            print('F-statistic: {:.4g} on {:d} and {:d} DF, p-value: {:.4g}'.format(
                self.F, self.q, self.n-self.q-1, p))


if __name__=="__main__":
    import argparse
    import models
    from inspect import getmembers, isfunction, isclass

    parser = argparse.ArgumentParser(description="Basic time series models")
    parser.add_argument('--list', type=bool, nargs='?', const=1, default=0, \
                        help='list all available modules')
    parser.add_argument('--doc', type=str, \
                        help='print documents of given function')
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.list:
        print("Functions:")
        for x in getmembers(models, isfunction):
            print("\t", x[0])
        print("Classes:")
        for x in getmembers(models, isclass):
            print("\t", x[0])
        exit()
    if not FLAGS.doc is None:
        print(getattr(models, FLAGS.doc).__doc__)
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))
