from __future__ import division
import numpy as np
import scipy
import matplotlib.pyplot as plt

from util import AR_striding, project_rowspace, \
        thin_svd, thin_svd_randomized, solve_psd

####################
#  Alg 3 in Ch. 3  #
####################

def compute_Ys(y,i):
    p = y.shape[1]
    Y = AR_striding(y,2*i-1).T
    Yp, Yf = Y[:i*p].copy(), Y[i*p:].copy()
    Ypp, Yfm = Y[:(i+1)*p].copy(), Y[(i+1)*p:].copy()
    return (Yp, Yf), (Ypp, Yfm)

def compute_Os((Yp,Yf),(Ypp,Yfm)):
    Oi = project_rowspace(Yf,Yp)
    Oim1 = project_rowspace(Yfm,Ypp)
    return Oi, Oim1


def viz_rank(Oi,k=None):
    if k is None:
        U, s, VT = np.linalg.svd(Oi)
    else:
        U, s, VT = thin_svd_randomized(Oi,k)

    plt.figure()
    plt.stem(np.arange(s.shape[0]),s)

def viz_rank2(y,i,k=None):
    Y = AR_striding(y,i-1).T
    l = Y.shape[0]//2

    if k is None:
        _, s, _ = np.linalg.svd(np.cov(Y)[:l,l:])
    else:
        _, s, _ = thin_svd_randomized(np.cov(Y)[:l,l:],k)

    plt.figure()
    plt.stem(np.arange(s.shape[0]),s)


def compute_gammas(Oi,nhat,p):
    U,s,VT = thin_svd_randomized(Oi,nhat)
    Gamma_i = U.dot(np.diag(np.sqrt(s)))
    Gamma_im1 = Gamma_i[:-p]
    return Gamma_i, Gamma_im1

def compute_Xs(Gamma_i,Gamma_im1,Oi,Oim1):
    Xihat = solve_psd(Gamma_i.T.dot(Gamma_i), Gamma_i.T.dot(Oi))
    Xip1hat = solve_psd(Gamma_im1.T.dot(Gamma_im1), Gamma_im1.T.dot(Oim1))
    return Xihat, Xip1hat

def system_parameters_from_states(Xihat, Xip1hat, Yii, nhat):
    ACT = np.linalg.lstsq(Xihat.T, np.vstack((Xip1hat, Yii)).T)[0]
    Ahat, Chat = ACT.T[:nhat], ACT.T[nhat:]
    rho = np.vstack((Xip1hat,Yii)) - ACT.T.dot(Xihat)

    QSR = rho.dot(rho.T) / rho.shape[1]
    Q, S, R = QSR[:nhat,:nhat], QSR[:nhat,nhat:], QSR[nhat:, nhat:]
    Bhat = scipy.linalg.sqrtm(Q)
    Dhat = scipy.linalg.sqrtm(R)

    return Ahat, Bhat, Chat, Dhat


def estimate_parameters_4sid(y,i,nhat=None,return_xs=False):
    p = y.shape[1]

    (Yp, Yf), (Ypp, Yfm) = compute_Ys(y,i)
    Oi, Oim1 = compute_Os((Yp,Yf),(Ypp,Yfm))

    if nhat is None:
        viz_rank2(y,i,k=10)
        plt.show()
        nhat = int(raw_input('n = '))

    Gamma_i, Gamma_im1 = compute_gammas(Oi,nhat,p)
    Xihat, Xip1hat = compute_Xs(Gamma_i,Gamma_im1,Oi,Oim1)

    Yii = Yf[:p]
    Ahat, Bhat, Chat, Dhat = system_parameters_from_states(Xihat, Xip1hat, Yii, nhat)

    if return_xs:
        return (Ahat, Bhat, Chat, Dhat), Xihat.T
    else:
        return Ahat, Bhat, Chat, Dhat

####################
#  Alg 2 in Ch. 3  #
####################

def estimate_parameters_moments(y,i,nhat):
    p = y.shape[1]

    ## my version, fast but does it work as well?
    Y = AR_striding(y,2*i-1).T
    l = Y.shape[0]//2
    U, s, VT = thin_svd(np.cov(Y)[:l,l:],nhat)

    ## slow version in book
    # (Yp, Yf), _ = compute_Ys(y,i)
    # Oi = project_rowspace(Yf,Yp)
    # U,s,VT = thin_svd(Oi,nhat)

    Gamma_i = U.dot(np.diag(np.sqrt(s)))

    ## estimate C
    C = Gamma_i[:p]

    ## estimate stable A (other methods on p. 54)
    A = np.linalg.lstsq(Gamma_i,np.vstack((Gamma_i[p:],np.zeros((p,Gamma_i.shape[1])))))[0]

    # TODO compute B, D
    # TODO why is C's first singular value so big? maybe need bigger i here

    return A, C


# python abuse!
from util import attach_print_enter_exit
attach_print_enter_exit()

