from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from synthetic_data import rand_lds_and_data
from estimation import estimate_parameters_4sid
from util import plot_eigvals

# TODO multiple sequences

if __name__ == '__main__':
    # data parameters
    n, p = 10, 5
    T = 20000

    # algorithm parameters
    i = 10

    # generate a system and simulate from it
    (A,B,C,D), (x,y) = rand_lds_and_data(T,n,p,eig_min=0.5,eig_max=1.0)

    # try to recover parameters (up to similarity transform)
    Ahat, Bhat, Chat, Dhat = estimate_parameters_4sid(y,i,nhat=n)

    # inspect the results a bit
    plt.figure()
    print sorted(np.linalg.eigvals(A),key=np.real)
    print sorted(np.linalg.eigvals(Ahat),key=np.real)
    plot_eigvals(A,'bo')
    plot_eigvals(Ahat,'rx')

    plt.savefig('results.pdf')

    plt.show()

