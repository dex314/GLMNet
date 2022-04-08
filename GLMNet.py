
import pandas as pd
import sidetable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sma
import os
from scipy.special import gammaln, logit, psi
from collections import namedtuple
import numpy.matlib as npm

from sklearn.base import RegressorMixin, BaseEstimator

## ========================================================================================
# defined outside of the class for instantiation
import numba
from numba import jit, njit
'''
Numba use case for descent portion "fast" of coordinate descent.
Current error due to cache (3/29/2022):
https://github.com/numba/numba/issues/7798

Setting cache = False
'''
@njit(cache=False, fastmath=True)
def fast(X, y, b0_init, b_init, lam, p, w, z, alpha=1, tol=1e-6, npass_max=1000):
    m,n = np.shape(X)
    b = np.zeros((n,))
    wsum = np.sum(w)
    # xsize = X.size ## is this even necessary? anymore?
    npass, tol_chk = 0, 1
    while tol_chk>tol and npass<npass_max:
        npass += 1
        b0 = np.dot( w.T, np.subtract(z, np.dot(X,b))) / wsum
        # if xsize != 0: ## is this even necessary? anymore?
        for ii in range(0,n):
            xi = X[:,ii]
            b[ii] = np.dot(xi.T, ( w*(np.subtract(z, np.dot(X,b)) - b0 + xi*b[ii]) ) )/m
            f = np.abs(b[ii]) - alpha*lam
            st = np.sign(b[ii]) * (np.abs(f) + f)/2.0
            b[ii] = np.divide(st , np.add( np.dot(xi.T, (w*xi))/m , (1.0-alpha)*lam ))
        tol_chk = np.linalg.norm(np.subtract(b0+b, b0_init+b_init))
    return b0, b, npass
## ========================================================================================

class GLMNet(BaseEstimator, RegressorMixin):
    ''' Elastic Net Base Parent Class '''
    def __init__(self):
        self.in_check = 'In Parent Class'

    def lam_seq_generator(self, X, y, offset=1, alpha=1, nlen=100, manual_lamseq=None):
        '''generates lambda sequence, updated with sklearn style'''
        if manual_lamseq is None:
            m,n=np.shape(X)
            if m>n: lratio = 1e-4
            else:   lratio = 1e-2
            if alpha <= 0: alpha = 1e-2
            ## sklearn versioning
            Xy = np.dot(X.T,y)
            lmax = np.sqrt(np.sum(Xy ** 2, axis=1)).max() / (m * alpha)
            return np.logspace(np.log10(lmax * lratio), np.log10(lmax), num=nlen)[::-1]
        else:
            manual_lamseq = np.array(manual_lamseq)
            if type(manual_lamseq) != np.ndarray and type(manual_lamseq) != list:
                raise Exception('** Manual lambdas must be a list or an numpy array and must be of length >= 2! **')
            assert len(manual_lamseq) >= 2, "** Length of Manual Lam Seq Must Be >= 2. **"
            return manual_lamseq.astype(float)

    def score(self, X, y, offset=1, sample_weight=None, score_method='r2'):
        '''Added for easy sklearn inheritance.
           Can choose r2, mean_deviance, or loglikelihood.
        '''
        y_pred = self.predict(X)
        if score_method == 'r2':
            from sklearn.metrics import r2_score
            return r2_score(y, y_pred, sample_weight=sample_weight)
        elif score_method == 'mean_dev':
            from sklearn.metrics import mean_tweedie_deviance
            return mean_tweedie_deviance(y, y_pred, sample_weight=sample_weight)
        else:
            print("Not yet implemented...")
            return 0

    ## moving this didnt help
    def glink(self, p, offset=1, link='log'):
        '''link function'''
        if link == 'identity':
            return p
        elif link == 'log':
            return np.exp(p + np.log(offset))
        elif link == 'logit':
            return self.sigmoid(p)
        else:
            raise Exception('** This link not implmented! **')
            return 0

    def sigmoid(self,z):
        ''' sigmoid function 1/(1+exp(-z)) for logit '''
        return 1.0/(1.0+np.exp(-z))

    def offset_check(self, offset, y):
        '''confirms offset size matches y'''
        if np.size(offset) == 1:
            if offset == 0:
                return np.ones(y.shape)
            else:
                return offset * np.ones(y.shape)
        else:
            return np.ones(y.shape)

    def array_safety(self, X, y, offset, weights):
        '''confirms that y has 2 dimensions'''
        if y.ndim <= 1:
            y = y.reshape(-1,1)
        mx,nx=np.shape(X)
        my,ny=np.shape(y)
        offset = self.offset_check(offset, y)
        offset = np.reshape(offset, (my*ny,1), order='F')
        weights = self.offset_check(weights, y)
        weights = np.reshape(weights, (my*ny,1), order='F')

        if ny > 1: self.multivariate_model = True
        X = npm.repmat(X, ny, 1)
        y = np.reshape(y, (my*ny,1), order='F')
        assert len(offset) == len(y), "Length of Offset != Length of y"
        assert len(weights) == len(y), "Length of Weights != Length of y"
        return X, y, offset, weights

    def corddesc(self, X, y, b0_init, b_init, lam, offset=1,
                 weights=1, k=1, alpha=1, nullDev=1, tol=1e-6):
        '''coordinate descent initiation to call to fast function'''
        ## coordinate descent initiation portion
        p = self.glink(b0_init + X.dot(b_init), offset, self.link)
        w, z = self.cd_weights(X, y, p, k, b0_init, b_init, self.pwr, weights)
        ## decaalred outside for instantiation
        w = w.flatten()
        z = z.flatten()
        p = p.flatten()
        # out = self.fast(X,y,b0_init,b_init,lam,p,w,z,alpha,tol,1000)
        out = fast(X,y,b0_init,b_init,lam,p,w,z,alpha,tol,1000)
        return out[0], out[1], out[2]

    # def fast(self, X, y, b0_init, b_init, lam, p, w, z, alpha=1, tol=1e-6, npass_max=1000):
    #     '''descent portion "fast" of coordinate descent'''
    #     m,n = np.shape(X)
    #     b = np.zeros((n,))
    #     wsum = np.sum(w)
    #     # xsize = X.size
    #     npass, tol_chk = 0, 1
    #     while tol_chk>tol and npass<npass_max:
    #         npass+=1
    #         b0 = np.dot( w.T, np.subtract(z, np.dot(X,b))) / wsum
    #         # if xsize != 0:
    #         for ii in range(0,n):
    #             xi = X[:,ii]
    #             b[ii] = np.dot(xi.T, ( w*(np.subtract(z, np.dot(X,b)) - b0 + xi*b[ii]) ) )/m
    #             f = np.abs(b[ii]) - alpha*lam
    #             st = np.sign(b[ii]) * (np.abs(f) + f)/2.0 ## SoftThreshHolding
    #             b[ii] = np.divide(st , np.add( np.dot(xi.T, (w*xi))/m , (1.0-alpha)*lam ))
    #         tol_chk = np.linalg.norm(np.subtract(b0+b, b0_init+b_init))
    #     return b0, b, npass

    def fit(self, X=None, y=None, offset=1, weights=None):
        ''' Fit call for consistency with sklearn
            offset is not used for Gaussian or Logistic.
        '''
        if type(X) == pd.core.frame.DataFrame:
            self.param_nm = X.columns
        else:
            self.param_nm = list(str('X'+str(xn)) for xn in range(X.shape[1]))
        X = np.array(X)
        if weights is None:
            weights = 1
        X, y, offset, weights = self.array_safety(X, np.array(y), offset, weights)
        mx,nx=np.shape(X)
        my,ny=np.shape(y)
        self.offset = offset
        self.weights = weights
        self.disp_method = 'x2'
        self.X = X
        self.y = y

        b_init = np.zeros((nx,1))
        b0_init = self.calc_b0_init(y, offset, self.link)
        k_init, dummy = self.disp_est(X, y, b0_init, b_init, self.pwr, offset,
                                        1, self.link, self.disp_method, self.weights)

        dev = self.deviance(X, y, b0_init, b_init, offset, k_init, self.link, self.pwr)
        ylam = y - self.glink(b0_init, offset, self.link)
        lambdas = self.lam_seq_generator(X, ylam, offset, self.alpha, self.lslen, self.manual_lamseq)

        self.lambdas = lambdas
        if self.manual_lamseq is not None:
            self.depth = len(self.manual_lamseq) - 1 ## to reflect appropiate sequence

        ##Storage Methods for Variables----------------------------------------
        minL = min(self.depth, self.lslen)
        betas = np.empty((nx, minL))
        beta0s = np.empty((1, minL))
        ks = np.empty((1, minL))
        yhats = np.empty((minL, my))
        disp_iters = np.empty((minL,1))
        mod_err = np.empty((minL,1))
        npasses = np.empty((minL,1))

        for j in range(minL):
            lambda1 = lambdas[j+1]
            lambda0 = lambdas[j]
            k, disp_iter = self.disp_est(X, y, b0_init, b_init, self.pwr, offset,
                                         k_init, self.link, self.disp_method, self.weights)

            nzb, jdum = np.nonzero( np.abs(X.T.dot(y)/mx) > self.alpha*(2.0*lambda1 - lambda0) )
            x_nzb = np.array(X[:,nzb])
            b_nzb = np.array(b_init[nzb])
            b0, b, npass = self.corddesc(x_nzb, y, b0_init, b_nzb, lambda1, offset,
                                         weights, k, self.alpha, dev/mx, self.tol)

            b0_init = np.copy(b0)
            k_init = np.copy(k)
            b_init[nzb] = b.reshape(-1,1)[:]
            model_dev = self.deviance(X,y,b0_init,b_init,offset,k_init,self.link,self.pwr)
            if (dev-model_dev)/dev > 0.9:  break ##sclars no need to np it
            yhat = b0_init + X.dot(b_init)

            betas[:,j] = np.copy(b_init.ravel())
            beta0s[:,j] = np.copy(b0_init)
            ks[:,j] = np.copy(k_init)
            yhats[j,:] = yhat.ravel()
            disp_iters[j] = disp_iter
            mod_err[j] = model_dev
            npasses[j] = npass

        ## MIN OUT OF SAMPLE ERROR PREDICTION - PICKING LOWEST LAMBDA
        min_errlm_idx = np.where(mod_err == np.nanmin(mod_err))[0][0]
        ## no longer need lowest beta check
        self.B = betas
        self.B0 = beta0s
        self.offset = offset
        self.min_lam_idx = min_errlm_idx
        self.K = ks
        self.disp_iter = disp_iters
        self.yhat = yhats
        self.model_errors = mod_err
        self.npasses = npasses

    def predict(self, X=None, offset=None):
        ''' Predict call for consistency with SKLEARN '''
        b0 = self.B0[-1,-1]
        b = self.B[:,-1].reshape(-1,1)
        if X is None:
            X = self.X
        if offset is None:
            offset = self.offset
        return self.glink(b0 + X.dot(b),offset,self.link)

## ===========================================================================================
class PoissonNet(GLMNet):
    def __init__(self, alpha=1.0, depth=20, lamseq_len=100, link='identity', pwr=None,
                 tol=1e-6, manual_lamseq=None):
        '''Poisson Distribution based Elastic Net. inherits from GLMNet.
            alpha = 1.0 regularization parameter
            depth = 20 early stopping for regularization
            lamseq_len = 100 lambda spacing from min to max
            link = 'identity'
            pwr = None (only fur use with Tweedie)
            tol = 1e-6
            manual_lamseq = None abillity to insert own lambda sequence for testing
        '''
        self.alpha = alpha
        self.depth = depth
        self.lslen = lamseq_len
        assert depth < lamseq_len, "** Depth must be less than length of Lambda sequence. **"
        self.tol = tol
        self.manual_lamseq = manual_lamseq
        self.link = link
        self.family_name = 'Poisson'
        self.pwr = pwr
        GLMNet.__init__(self)
        # super().__init__()

    ## this is slightly better because its a child calling to the parent but there is still
    ## multiple inheritance because the fit call above references child definitions
    ## thats why its still slower and isnt as fast

    def calc_b0_init(self, y, offset, link):
        b0_init = np.log(np.mean(y/offset, axis=0))
        return b0_init

    def deviance(self, X, y, b0, b, offset, k, link, pwr):
        m,n = np.shape(X)
        mu = self.glink(b0 + X.dot(b), offset, link)
        LL = y*mu - mu - gammaln(y+1)
        L = -2.0*LL.T.dot(LL)
        return L

    def cd_weights(self, X, y, p, k, b0, b, pwr, weights):
        q0 =  np.divide( (y-p) , p )
        w = np.ones((len(y),1))*p
        z =  b0 + np.add(X.dot(b), q0)
        return w, z

    def disp_est(self, X, y, b0, b, pwr, offset=1, k=1, link='log', disp_method=None, weights=None):
        k, iters = 1e-5, 0
        return k, iters

    def model_score(self, X, y, offset=1, sample_weight=None):
        return self.score(self, X, y, offset=1, sample_weight=None, score_method='r2')

    ## adding in the fit call  or moving down here didnt change the 0.5 time it took
    ## so despite the code being nearly identical to elastic net, something is still hanging it up

## ===========================================================================================
class GaussNet(GLMNet):
    def __init__(self, alpha=1.0, depth=20, lamseq_len=100, link='identity', pwr=None,
                 tol=1e-6, manual_lamseq=None):
        '''Gaussian Distribution based Elastic Net. inherits from GLMNet.
            alpha = 1.0 regularization parameter
            depth = 20 early stopping for regularization
            lamseq_len = 100 lambda spacing from min to max
            link = 'identity'
            pwr = None (only fur use with Tweedie)
            tol = 1e-6
            manual_lamseq = None abillity to insert own lambda sequence for testing
        '''
        self.alpha = alpha
        self.depth = depth
        self.lslen = lamseq_len
        assert depth < lamseq_len, "** Depth must be less than length of Lambda sequence. **"
        self.tol = tol
        self.manual_lamseq = manual_lamseq
        self.link = link
        self.family_name = 'Gaussian'
        self.pwr = pwr
        GLMNet.__init__(self)

    def calc_b0_init(self, y, offset, link):
        b0_init = np.mean(y, axis=0)
        return b0_init

    def deviance(self, X, y, b0, b, offset, k, link, pwr):
        m,n = np.shape(X)
        mu = self.glink(b0 + X.dot(b), offset, link)
        LL = np.subtract(y, mu)
        L = 0.5/len(y) * LL.T.dot(LL)
        return L

    def cd_weights(self, X, y, p, k, b0, b, pwr, weights):
        w = np.ones((len(y),1))
        z = y.copy()
        return w, z

    def disp_est(self, X, y, b0, b, pwr, offset=1, k=1, link='identity', disp_method=None, weights=None):
        k, iters = 1e-5, 0
        return k, iters

    def model_score(self, X, y, offset=1, sample_weight=None):
        return self.score(self, X, y, offset=1, sample_weight=None, score_method='r2')

## ===========================================================================================
class NegBinNet(GLMNet):
    def __init__(self, alpha=1.0, depth=20, lamseq_len=100, link='identity', pwr=None,
                 tol=1e-6, manual_lamseq=None):
        '''Negative Binomial Distribution based Elastic Net. inherits from GLMNet.
            alpha = 1.0 regularization parameter
            depth = 20 early stopping for regularization
            lamseq_len = 100 lambda spacing from min to max
            link = 'identity'
            pwr = None (only fur use with Tweedie)
            tol = 1e-6
            manual_lamseq = None abillity to insert own lambda sequence for testing
        '''
        self.alpha = alpha
        self.depth = depth
        self.lslen = lamseq_len
        assert depth < lamseq_len, "** Depth must be less than length of Lambda sequence. **"
        self.tol = tol
        self.manual_lamseq = manual_lamseq
        self.link = link
        self.family_name = 'NegBin'
        self.pwr = pwr
        GLMNet.__init__(self)

    def calc_b0_init(self, y, offset, link):
        b0_init = np.log(np.mean(y/offset, axis=0))
        return b0_init

    def deviance(self, X, y, b0, b, offset, k, link, pwr):
        m,n = np.shape(X)
        mu = self.glink(b0 + X.dot(b), offset, link)
        LL = y*np.log(k*mu) - (y+1/k)*np.log(1+k*mu) + gammaln(y+1/k) - gammaln(1/k) - gammaln(y+1)
        L = -2.0*LL.T.dot(LL)
        return L

    def cd_weights(self, X, y, p, k, b0, b, pwr, weights):
        s = np.divide( (k*y+1.0)*p , (k*p + 1.0)**2 )
        q0 = np.divide( k*p+1.0 , (k*y+1.0)*p )
        w = np.ones((len(y),1))*s
        z = b0 + np.add(X.dot(b), np.subtract(y,p)*q0)
        return w, z

    def disp_est(self, X, y, b0, b, pwr, offset=1, k=1, link='log', disp_method=None, weights=None):
        iters, k0 = 0, 0
        ## part of the gradient calc
        mu = self.glink(b0 + X.dot(b), offset, link)
        while np.abs(k-k0) > 1e-3:
            k0 = np.copy(k)

            ## NegBin gradient calc moved inside for this dispersion estiamte
            g1 = psi(y+1/k)*(-1/k**2) + psi(1/k)*(1/k**2) + (1/k**2)*np.log(k) - (1/k**2)
            g2 = (1/k**2)*np.log(1/k + mu) + (1/k**3)/(1/k + mu) + (y/(1/k + mu))*(1/k**2)
            local_grad = -np.sum(g1+g2)

            k = k - 0.01/np.sqrt(len(X)+iters) * local_grad #self.local_grad(X,y,b0,b,offset,k,link)
            iters += 1 ## this is mainly for error checking
            if k<0:
                k = 1e-6
        return k, iters

    def model_score(self, X, y, offset=1, sample_weight=None):
        return self.score(self, X, y, offset=1, sample_weight=None, score_method='r2')

## ===========================================================================================
class BinomialNet(GLMNet):
    def __init__(self, alpha=1.0, depth=20, lamseq_len=100, link='identity', pwr=None,
                 tol=1e-6, manual_lamseq=None):
        '''Binomial Distribution based Elastic Net. inherits from GLMNet.
            alpha = 1.0 regularization parameter
            depth = 20 early stopping for regularization
            lamseq_len = 100 lambda spacing from min to max
            link = 'identity'
            pwr = None (only fur use with Tweedie)
            tol = 1e-6
            manual_lamseq = None abillity to insert own lambda sequence for testing
        '''
        self.alpha = alpha
        self.depth = depth
        self.lslen = lamseq_len
        assert depth < lamseq_len, "** Depth must be less than length of Lambda sequence. **"
        self.tol = tol
        self.manual_lamseq = manual_lamseq
        self.link = link
        self.family_name = 'Binomial'
        self.pwr = pwr
        GLMNet.__init__(self)

    def calc_b0_init(self, y, offset, link):
        if link != 'logit':
            b0_init = np.log(np.mean(y/offset, axis=0))
        else:
            b0_init = np.log(np.mean(y,axis=0)/(1-np.mean(y,axis=0)))
        return b0_init

    def deviance(self, X, y, b0, b, offset, k, link, pwr):
        m,n = np.shape(X)
        mu = self.glink(b0 + X.dot(b), offset, link)
#       LL = np.add(np.where(y>0, y*np.log(mu), 0), np.where(y<1, (1.0-y)*np.log(1.0-mu), 0))
        ## from statsmodels
        LL = gammaln(n+1) - gammaln(y+1) - gammaln(n-y+1) + y*np.log(mu/(1-mu)) + n*np.log(1-mu)
        L = -2.0*LL.T.dot(LL)
        return L

    def cd_weights(self, X, y, p, k, b0, b, pwr, weights):
        s = np.multiply( p, (1.0-p) )
        q0 =  np.divide( (y-p) , s )
        w = np.ones((len(y),1))*s
        z =  b0 + np.add(X.dot(b), q0)
        return w, z

    def disp_est(self, X, y, b0, b, pwr, offset=1, k=1, link='logit', disp_method=None, weights=None):
        k, iters = 1e-5, 0
        return k, iters

    def model_score(self, X, y, offset=1, sample_weight=None):
        return self.score(self, X, y, offset=1, sample_weight=None, score_method='r2')


## ===========================================================================================
class TweedieNet(GLMNet):
    def __init__(self, alpha=1.0, depth=20, lamseq_len=100, link='identity', pwr=1.5,
                 tol=1e-6, manual_lamseq=None):
        '''Tweedie Distribution based Elastic Net. inherits from GLMNet.
            TweedieNet is based on the Compound Tweedie Poisson work done in:
            https://www.math.mcgill.ca/yyang/resources/papers/JCGS_HDtweedie.pdf
            There are stability issues with powers < 1 and >= 2 so it is recommended to
            stay within those boundaries. It is also suggested to keep the weighting at 1
            unless for specific reasons.
            alpha = 1.0 regularization parameter
            depth = 20 early stopping for regularization
            lamseq_len = 100 lambda spacing from min to max
            link = 'identity'
            pwr = 1.5 (only fur use with Tweedie)
            tol = 1e-6
            manual_lamseq = None abillity to insert own lambda sequence for testing
        '''
        self.alpha = alpha
        self.depth = depth
        self.lslen = lamseq_len
        assert depth < lamseq_len, "** Depth must be less than length of Lambda sequence. **"
        self.tol = tol
        self.manual_lamseq = manual_lamseq
        self.link = link
        self.family_name = 'Tweedie'
        self.pwr = pwr
        if (pwr > 0) & (pwr < 1):
            warnings.warn(" ** Warning : Instabilities for 0 > power < 1.0 and p >= 2.0 ! ** ")
        GLMNet.__init__(self)

    def calc_b0_init(self, y, offset, link):
        b0_init = np.log(np.mean(y/offset, axis=0))
        return b0_init

    def deviance(self, X, y, b0, b, offset, k, link, pwr):
        m,n = np.shape(X)
        mu = self.glink(b0 + X.dot(b), offset, link)
        if pwr==0:    ## normal dist
            dev = np.power(y-mu,2)/2
        elif pwr==1:  ## poisson dist
            dev = np.where(y==0, mu, y * np.log(y/mu) + (mu-y))
        elif pwr==2:  ## gamma dist
            dev = np.log(mu/y) + y/mu - 1
        else:
            dev1 = np.power(y,2-pwr) / ((1-pwr)*(2-pwr))
            dev2 = y*np.power(mu,1-pwr)/(1-pwr) + np.power(mu,2-pwr)/(2-pwr)
            dev = dev1 - dev2
        L = -2.0*dev.T.dot(dev)
        return L

    def cd_weights(self, X, y, p, k, b0, b, pwr, weights):
#         mu = np.log(p) ## may have to define these better
        mu = b0 + X.dot(b) #, np.log(offset))
        r1 = weights*y*np.exp(-(pwr-1)*mu)
        r2 = weights*np.exp((2-pwr)*mu)
        w = (pwr-1)*r1 + (2-pwr)*r2 ## vtt
        z = mu + (r1-r2)/w   ## yt
        return w, z

    def disp_est(self, X, y, b0, b, pwr, offset=1, k=1, link='log', disp_method='x2', weights=None):
        iters = 0
        mu = self.glink(b0 + X.dot(b),offset,link)
        if disp_method == 'x2': ##chi sqquare method
            resid = np.power(y - mu, 2) * weights
            mu_var = np.power(mu, pwr)
            k = np.sum(resid / mu_var) / len(resid)
        else:
            k = self.deviance(X, y, b0_init, b_init, pwr, offset, k, link) / len(resid)
        return k, iters

    def model_score(self, X, y, offset=1, sample_weight=None):
        return self.score(self, X, y, offset=1, sample_weight=None, score_method='mean_dev')
