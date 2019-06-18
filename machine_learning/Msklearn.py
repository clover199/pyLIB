from __future__ import absolute_import, print_function, division
import numpy as np
import logging

"""
This file defines functions that minic sklearn functions
"""


def LeastSquare(x, y):
    """
    A simple linear regression with 1D data
    input:  x   1D-array of variant
            y   1D-array of response
    output:     slope, intercept
    """
    n = x.shape[0]
    mx = np.mean(x)
    my = np.mean(y)
    slope = (np.dot(x,y) - n*mx*my) / (np.dot(x,x) - n*mx**2)
    inter = my - slope*mx
    return slope, inter


class LinearRegression:
    """
    Linear regression
    LinearRegression(intercept=True, tol=1e-6)
        intercept   indicate whether to fit intercept
        tol:        tolerance of zero values
    functions:      fit(X, y, sig=None):    X must be a 1D or 2D array
                                            y is a 1D array
                                            sig is error covariance matrix or weight array
                    transform(X):   X must be a 1D or 2D array
                    summary():      prints a summary of the model
                    plot(X, y, residual=True):
                                    X must be a 1D or 2D array
                                    y is a 1D array
                                    residual indicates whether to make residual plot
    attributes:     var     variance ratio
                    arg     arguments of kept features
                    tol_    tolerance
    """
    def __init__(self, intercept=True, tol=1e-6):
        import sys
        import os
        path = '\\'.join(os.path.abspath(__file__).split('\\')[:-2])
        if not path in sys.path:
            sys.path.insert(0, path)
        self.tol_ = tol
        self.intercept = intercept
        self.logger = logging.getLogger(__name__)
        self.logger.info("------ LinearRegression(self, intercept=True, tol=1e-6) initialized ------")

    def fit(self, x, y, sig=None):
        from basic.mathe import inverse
        if x.ndim==1:
            x = x.reshape([-1,1])
        if self.intercept:
            x = np.concatenate([np.ones([x.shape[0],1]), x], axis=1)
        n, p = x.shape

        if sig is None:
            xtx = x.T.dot(x)
            xy = x.T.dot(y)
            self.var_beta, p = inverse(xtx, self.tol_, return_df=True)
            self.beta = self.var_beta.dot(xy)
            y_hat = x.dot(self.beta)
            self.sig = np.sum((y_hat-y)**2)/(n-p)
            self.var_beta *= self.sig
        else:
            if sig.ndim==1:
                sig_inv = sig
            else:
                sig_inv = inverse(sig, self.tol_)
            xtx = x.T.dot(sig_inv).dot(x)
            xy = x.T.dot(sig_inv).dot(y)
            self.var_beta, p = inverse(xtx, self.tol_, return_df=True)
            self.beta = self.var_beta.dot(xy)
            y_hat = x.dot(self.beta)

        if self.intercept:
            difference = np.abs(np.mean(y)-np.mean(y_hat))
            assert difference<self.tol_, \
            "Mean of y and y_hat are not the same. Difference {:g}".format(difference)
            yc = np.mean(y)
            self.rsquare = np.mean((y_hat-yc)**2) / np.mean((y-yc)**2)
            self.adjrsquare = 1-(1-self.rsquare)*(n-1)/(n-p)
            self.F = (np.sum((y_hat-yc)**2)) / (p-1) / (np.sum((y_hat-y)**2)) * (n-p)
        else:
            self.rsquare = np.mean(y_hat**2) / np.mean(y**2)
            self.adjrsquare = 1-(1-self.rsquare)*n/(n-p)
            self.F = (np.sum((y_hat)**2)) / p / (np.sum((y_hat-y)**2)) * (n-p)

        self.n = n
        self.p = p

    def transform(self, x):
        if x.ndim==1:
            x = x.reshape([-1,1])
        if self.intercept:
            x = np.concatenate([np.ones([x.shape[0],1]), x], axis=1)
        return x.dot(self.beta)

    def summary(self):
        import pandas as pd
        from scipy import stats
        if self.intercept:
            index = ['const']+['x{:d}'.format(x) for x in range(1,self.beta.shape[0])]
        else:
            index = ['x{:d}'.format(x+1) for x in range(self.beta.shape[0])]
        coefs = pd.DataFrame(columns=['coef','std err','t','p>|t|', ''], index=index)
        coefs.iloc[:,0] = self.beta
        coefs.iloc[:,1] = np.sqrt(np.diag(self.var_beta))
        coefs.iloc[:,2] = coefs.iloc[:,0]/coefs.iloc[:,1]
        coefs.iloc[:,3] = 2*stats.t.cdf(-np.abs(coefs.iloc[:,2]), self.n-self.p)
        for i, v in enumerate(coefs.iloc[:,3]):
            if v>0.1: coefs.iat[i,4] = ''
            elif v>0.05: coefs.iat[i,4] = '.'
            elif v>0.01: coefs.iat[i,4] = '*'
            elif v>0.001: coefs.iat[i,4] = '**'
            else: coefs.iat[i,4] = '***'
        try:
            display(np.round(coefs,4))
        except:
            print(np.round(coefs,4))
        print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")
        print('R-squared: {:.3f},'.format(self.rsquare),
              'Adjusted R-squared: {:.3f}'.format(self.adjrsquare))
        if self.intercept:
            p = 1 - stats.f.cdf(self.F, self.p-1, self.n-self.p)
            print('F-statistic: {:.4g} on {:d} and {:d} DF, p-value: {:.4g}'.format(self.F, self.p-1, self.n-self.p, p))
        else:
            p = 1 - stats.f.cdf(self.F, self.p, self.n-self.p)
            print('F-statistic: {:.4g} on {:d} and {:d} DF, p-value: {:.4g}'.format(self.F, self.p, self.n-self.p, p))

    def plot(self, x, y, residual=True):
        import matplotlib.pyplot as plt
        from scipy import stats
        if x.ndim==1:
            x = x.reshape([-1,1])
        xmean = np.mean(x, axis=0)
        D, U = np.linalg.eigh( (x-xmean).T.dot(x-xmean) )
        pc = (x-xmean).dot(U[:,-1])
        pcs = np.linspace(min(pc), max(pc), 1000, endpoint=True)
        new_x = np.zeros([1000, x.shape[1]])
        new_x[:,-1] = pcs
        new_x = new_x.dot(U.T) + xmean
        if self.intercept:
            x = np.concatenate([np.ones([x.shape[0],1]), x], axis=1)
            new_x = np.concatenate([np.ones([new_x.shape[0],1]), new_x], axis=1)
        ys = new_x.dot(self.beta)

        plt.plot(pcs, ys, 'k-')
        plt.plot(pc, y, 'ko')
        var = np.sum(new_x.T * self.var_beta.dot(new_x.T), axis=0)
        t5 = stats.t.ppf(0.025, self.n-self.p)
        ci = t5 * np.sqrt(var)
        pi = t5 * np.sqrt(self.sig+var)
        plt.fill_between(pcs, ys-ci, ys+ci, color='grey', alpha='0.5')
        plt.text(pcs[0], ys[0]+ci[0], "CI")
        plt.fill_between(pcs, ys-pi, ys+pi, color='grey', alpha='0.2')
        plt.text(pcs[0], ys[0]+pi[0], "PI")
        plt.xlim([min(pcs),max(pcs)])
        plt.xlabel("1st PC")
        plt.ylabel("y")
        plt.show()

        if residual:
            y_hat = x.dot(self.beta)
            h = np.sum(x.T * self.var_beta.dot(x.T), axis=0)
            r_std = (y-y_hat) / np.sqrt( self.sig - h )
            r_stu = r_std * np.sqrt( (self.n-self.p-1) / (self.n-self.p-r_std**2) )
            plt.plot([min(y_hat),max(y_hat)],[0,0],'k-')
            pv = stats.t.ppf(0.025, self.n-self.p-1)
            plt.fill_between([min(y_hat),max(y_hat)], [pv,pv],[-pv,-pv], color='grey', alpha='0.5')
            plt.text(min(y_hat), pv, "95%")
            pv = stats.t.ppf(0.005, self.n-self.p-1)
            plt.fill_between([min(y_hat),max(y_hat)], [pv,pv],[-pv,-pv], color='grey', alpha='0.2')
            plt.text(min(y_hat), pv, "99%")
            plt.plot(y_hat, r_stu, 'ko')
            plt.xlim([min(y_hat),max(y_hat)])
            plt.xlabel("Fitted values")
            plt.ylabel("Studentized residuals")
            plt.text(min(y_hat), pv, "95%")
            plt.show()


class select_cov:
    """
    Calculate the variance of features and select the features with non-zero variance
    select_cov(tol=1e-6)
        tol:    tolerance of zero values
    functions:  fit(X):     X must be a 2D array, return variance ratio
                transform(X):   X must be a 2D array
                inverse_transfor(X):    X can be 1D or 2D array
    attributes: var     variance ratio
                arg     arguments of kept features
                tol_    tolerance
    """
    def __init__(self, tol=1e-6):
        self.tol_ = tol

    def fit(self, data):
        self.var = np.mean((data-np.mean(data,axis=0))**2, axis=0)
        self.var = self.var/sum(self.var)
        self.arg = np.argwhere(self.var>self.tol_).ravel()
        print("kept variance ratio {:.6f} from {:d} features out of {:d} features".format(sum(self.var[self.arg]),
                                                                                   len(self.arg),
                                                                                   data.shape[1]))
        return self.var

    def transform(self, data):
        return data[:,self.arg]

    def inverse_transform(self, data):
        if data.ndim==1:
            result = np.zeros(len(self.var))
            result[self.arg] = data
            return result
        if data.ndim==2:
            result = np.zeros([data.shape[0], len(self.var)])
            result[:,self.arg] = data
            return result


class PrincipalComponentsAnalysis:
    """
        Principal Components Analysis
    """

    def fit(self, X):
        n = X.shape[0]
        X = X.reshape([n, -1])
        self.mean_ = np.mean(X, axis=0)

        # SVD of centered data matrix
        U, S, Vh = np.linalg.svd(X-self.mean_, full_matrices=False)

        self.components_ = Vh.T.conjugate()
        self.explained_variance_ = S**2/(n-1)

    def transform(self, X):
        X = X.reshape([X.shape[0], -1])
        return (X-self.mean_).dot(self.components_)


class LinearDiscriminantAnalysis:
    """
        Linear Discriminant Analysis
    """

    def fit(self, X, y):
        n = X.shape[0]
        X = X.reshape([n, -1])
        self.classes_ = np.sort(np.unique(y))
        nk = self.classes_.shape[0]
        self.priors_ = np.array([np.sum(y==i)/n for i in self.classes_])
        self.means_ = np.array([np.mean(X[y==i], axis=0) for i in self.classes_])
        self.xbar_ = np.mean(self.means_, axis=0)

        # SVD of a new data matrix with each group centered at the origin
        temp = np.concatenate([X[y==l]-self.means_[i] for i,l in enumerate(self.classes_)], axis=0)
        U, S, Vh = np.linalg.svd(temp, full_matrices=False)

        # calculate coefficients and intercept in decision function
        V = Vh.T.conjugate()
        self.coef_ = (V/S**2).dot(Vh).dot( (self.means_-self.xbar_).T ) * (n-nk)
        self.intercept_ = -np.sum((self.means_+self.xbar_).conjugate() * self.coef_.T, axis=1)/2 \
                        + np.log(self.priors_)

        # calculate transformation coefficients
        U, S0, Vh0 = np.linalg.svd( (self.means_-self.xbar_).dot(V/S) )
        self.scalings_ = (V/S).dot( Vh0.T.conjugate()[:,:S0.shape[0]-1] ) * np.sqrt(n-nk)

    def decision_function(self, X):
        n = X.shape[0]
        X = X.reshape([n, -1])
        return self.intercept_ + X.conjugate().dot(self.coef_)

    def transform(self, X):
        X = X.reshape([X.shape[0], -1])
        return (X-self.xbar_).dot(self.scalings_)


class QuadraticDiscriminantAnalysis:
    """
        Quadratic Discriminant Analysis
    """

    def fit(self, X, y):
        n = X.shape[0]
        X = X.reshape([n, -1])
        self.classes_ = np.sort(np.unique(y))
        self.priors_ = np.array([np.sum(y==i)/n for i in self.classes_])
        self.means_ = np.array([np.mean(X[y==i], axis=0) for i in self.classes_])
        self.rotations_ = []
        self.scalings_ = []
        for i, l in enumerate(self.classes_):
            temp = X[y==l]-self.means_[i]
            U, S, Vh = np.linalg.svd(temp)
            self.rotations_.append(Vh.T.conjugate())
            self.scalings_.append(S**2/(np.sum(y==l)-1))
        self.coef_ = self.priors_ / np.sqrt([np.prod(sig) for sig in self.scalings_])

    def predict_proba(self, X):
        n = X.shape[0]
        X = X.reshape([n, -1])
        quadratic = []
        for i in range(self.classes_.shape[0]):
            rotated = (X-self.means_[i]).dot(self.rotations_[i])
            quadratic.append( np.exp(-np.sum(np.abs(rotated)**2/self.scalings_[i], axis=1)/2) )
        proba = self.coef_ * np.vstack(quadratic).T
        return ( proba.T / np.sum(proba, axis=1) ).T


class KernelPCA:
    """
        Kernel Principal Components Analysis
    """
    def __init__(self, kernel_function, tol=1e-10, **kwargs):
        self.kernel = kernel_function
        self.kernel_params = kwargs
        self.tol_ = tol

    def fit(self, X):
        n = X.shape[0]
        X = X.reshape([n, -1])
        kernel = self.kernel(X, **self.kernel_params)
        H = np.eye(n) - np.ones([n,n])/n
        self.mean_ = np.mean(kernel, axis=1)
        D, U = np.linalg.eigh(H.dot(kernel).dot(H))
        self.alphas_ = U[:,::-1]
        self.lambdas_ = D[::-1]
        self.X_fit_ = X

    def transform(self, X):
        temp = self.kernel(X, Y=self.X_fit_, **self.kernel_params)
        inverse_lambdas = np.where(self.lambdas_>self.tol_, 1/self.lambdas_, 0)
        return (temp-self.mean_).dot(self.alphas_ * np.sqrt(inverse_lambdas) )


def LinearKernel(X, Y=None):
    """
    input:  X,Y     data matrices of the same shape with rows as data points.
                    If Y=None, set Y=X, default None.
    returns X * Y^T
    """
    if Y is None:
        Y = X
    return X.dot(Y.T)

def PolynomialKernel(X, Y=None, degree=3, gamma=None, coef0=1):
    """
    input:  X,Y     data matrices of the same shape with rows as data points.
                    If Y=None, set Y=X, default None.
            degree  polynomial degree
            gamma   rescale factor, default 1/n, n is number of points
            coef0   bias value, default 1
    returns ( gamma * X * Y^T + coef0 ) ^ degree
    """
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0/X.shape[1]
    return np.power( gamma*X.dot(Y.T) + coef0, degree )

def RBFKernel(X, Y=None, gamma=None):
    """
    input:  X,Y     data matrices of the same shape with rows as data points.
                    If Y=None, set Y=X, default None.
            degree  polynomial degree
            gamma   rescale factor, default 1/n, n is number of points
    returns exp( - gamma * ||x-y||^2 )
    """
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0/X.shape[1]
    diff = np.tile( np.sum(X**2,axis=1), [Y.shape[0], 1]).T \
         + np.tile( np.sum(Y**2,axis=1), [X.shape[0], 1]) \
         - 2 * X.dot(Y.T)
    return np.exp( - gamma * diff )

def SigmoidKernel(X, Y=None, gamma=None, coef0=1):
    """
    input:  X,Y     data matrices of the same shape with rows as data points.
                    If Y=None, set Y=X, default None.
            gamma   rescale factor, default 1/n, n is number of points
            coef0   bias value, default 1
    returns x * y^T / ||x|| / ||y||
    """
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0/X.shape[1]
    return np.tanh( gamma*X.dot(Y.T) + coef0)

def EuclideanDistance(X, Y=None):
    """
    input:  X,Y     data matrices of the same shape with rows as data points.
                    If Y=None, set Y=X, default None.
    returns ||x-y||
    """
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0/X.shape[1]
    diff = np.tile( np.sum(X**2,axis=1), [Y.shape[0], 1]).T \
         + np.tile( np.sum(Y**2,axis=1), [X.shape[0], 1]) \
         - 2 * X.dot(Y.T)
    return np.sqrt(diff)

def CosineSimilarity(X, Y=None):
    """
    input:  X,Y     data matrices of the same shape with rows as data points.
                    If Y=None, set Y=X, default None.
    returns ( gamma * X * Y^T + coef0 ) ^ degree = cos theta
    """
    X_normalized = X.T / np.sqrt(np.sum(X**2,axis=1))
    if Y is None:
        Y_normalized = X_normalized
    else:
        Y_normalized = Y.T / np.sqrt(np.sum(Y**2,axis=1))
    return X_normalized.T.dot(Y_normalized)


if __name__=="__main__":
    import argparse
    import Msklearn
    from inspect import getmembers, isfunction, isclass

    parser = argparse.ArgumentParser(description="My Functions as sklearn")
    parser.add_argument('--list', type=bool, nargs='?', const=1, default=0, \
                        help='list all available modules')
    parser.add_argument('--doc', type=str, \
                        help='print documents of given function')
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.list:
        print("Functions:")
        for x in getmembers(Msklearn, isfunction):
            print("\t", x[0])
        print("Classes:")
        for x in getmembers(Msklearn, isclass):
            print("\t", x[0])
        exit()
    if not FLAGS.doc is None:
        print(getattr(Msklearn, FLAGS.doc).__doc__)
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))
