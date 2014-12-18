from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from synthetic_data import rand_lds_and_data
from estimation import estimate_parameters_4sid, estimate_parameters_moments
from util import plot_eigvals, normalize, plot_singularvals

# TODO multiple sequences
# TODO is algorithm 2 more statistically efficient?

if __name__ == '__main__':
    ## data parameters
    n, p = 16, 8
    T = 30000

    ## generate a system and simulate from it
    (A,B,C,D), (x,y) = rand_lds_and_data(T,n,p,eig_min=0.5,eig_max=1.0)


    ## try to recover parameters (up to similarity transform)
    i = 10
    # Ahat, Bhat, Chat, Dhat = estimate_parameters_4sid(y,i,nhat=n)
    Ahat, Chat = estimate_parameters_moments(y,i,n)

    ## inspect the results a bit
    plt.figure(figsize=(5,10))

    plt.subplot(2,1,1)
    plot_eigvals(A,'bo')
    plot_eigvals(Ahat,'rx')
    plt.axis('equal')

    plt.subplot(2,1,2)
    plot_singularvals(C,'b')
    plot_singularvals(Chat,'r')

    plt.savefig('results.pdf')

    plt.show()

