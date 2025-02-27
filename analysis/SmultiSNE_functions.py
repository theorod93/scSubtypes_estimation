# -*- coding: utf-8 -*-
"""
@author: Theodoulos Rodosthenous
"""

'''
In this script, the following functions are included
- t-SNE
- Multi-SNE
- S-multi-SNE (missing data)

NOTE: t-SNE (function 'tsne') is taken directly by the author's package
found in his website: https://lvdmaaten.github.io/tsne/

'''

import numpy as np
import pylab
from scipy.linalg import eigh, svd, qr, solve
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import eigsh
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.neighbors import NearestNeighbors
import math
import matplotlib.pyplot as plt
import pandas as pd

'''
Multi-SNE:
Input: X = (X_1,..., X_M), where M is the number of views
    X_m \in R^{NxD_m}, for each m = {1,...,M}
Output: Y \in R^{Nxd}, d:usually equals to 2 for good 2D visualisation
Two separate methods:
    (A) multi-SNE -> Our approach
    (B) m-SNE -> Xie et al. (2011)
'''

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """
    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    if sumP == 0:
        sumP = 1
    if math.isinf(sumP):
        sumP = 1
    if math.isnan(sumP):
        sumP = 1
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P
# Get Probability distribution (P) from data (X)
def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """
    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)
    # Loop over all datapoints
    for i in range(n):
        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))
        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])
        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.
            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1
        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P
# PCA
def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y
# t-SNE
def tSNE(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """
    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1
    # Initialize variables
    if initial_dims != None:
        X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))
    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    (d1,d2) = P.shape
    k = np.argwhere(np.isnan(P))
    (kDim1, kDim2) = k.shape
    for i in range(kDim1):
        d1 = k[i][0]
        d2 = k[i][1]
        P[d1][d2] = 0
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.                                                              # early exaggeration
    P = np.maximum(P, 1e-12)
    # Run iterations
    for iter in range(max_iter):
        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))
        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))
        # Stop lying about P-values
        if iter == 100:
            P = P / 4.
    # Return solution
    return Y
## Multi-view ##
# Multi-view PCA
# Applied separately on each data-view
def multiPCA(X=np.array([[]]), no_dims=50):
    """
        Runs PCA on M datasets, each being a NxD_i array in order to
        reduce their dimensionality to no_dims dimensions. PCA is applied
        on each dataset separately
    """
    dim = X.shape
    M = dim[0]
    Yout = X
    for view in range(M):
        XsetTemp = X[view]
        Xset = np.array(XsetTemp)
        (n,d) = Xset.shape
        Ytemp = Xset - np.tile(np.mean(Xset, 0), (n,1))
        (l,m) = np.linalg.eig(np.dot(Ytemp.T, Ytemp))
        Y = np.dot(Ytemp, m[:, 0:no_dims])
        Yout[view] = Y
    # Return solution
    return Yout
# Multi-SNE
def multiSNE(X = np.array([[]]), no_dims = 2, initial_dims = 50, perplexity = 30.0, max_iter = 1000):
    """
        Runs multi-SNE on the array(list) X, which includes M datasets
        in the MxNxD_i for each dataset X_i to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = multiSNE(X, no_dims, initial_dims,perplexity,max_iter),
        where X is an MxNxD_i, i from 1 to M NumPy array.
    """
    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1
    # Initialize variables
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    # Now, initialization for each data-set
    dim = X.shape
    M = dim[0]
    # Make sure each data-set has the same axis=0 dimension
    Xtemp = X[1]
    XsTemp = np.array(Xtemp)
    baselineDim = XsTemp.shape[0]
    for set in range(M):
        Xtemp = X[set]
        XsTemp = np.array(Xtemp)
        dimToCheck = XsTemp.shape[0]
        if dimToCheck != baselineDim:
            print("Error: Number of rows (samples) must be the same in all data-sets of list X.")
            return -1
    if initial_dims==None:
        Xpca = np.copy(X)
    else:
        Xpca = multiPCA(X, initial_dims).real
    # Initialise a NumPy array of lists -- size=Num.of.data_views
    a = [(0,0),(0,0), (0,0)]
    b = [(0,0),(0,0)]
    if M==2:
        Xi = np.array([a,b])
    elif M ==3:
        Xi = np.array([a,b,b])
    elif M ==4:
        Xi = np.array([a,b,b,b])
    elif M ==5:
        Xi = np.array([a,b,b,b,b])
    elif M ==6:
        Xi = np.array([a,b,b,b,b,b])
    elif M ==7:
        Xi = np.array([a,b,b,b,b,b,b])
    elif M ==8:
        Xi = np.array([a,b,b,b,b,b,b,b])
    elif M ==9:
        Xi = np.array([a,b,b,b,b,b,b,b,b])
    elif M ==10:
        Xi = np.array([a,b,b,b,b,b,b,b,b,b])
    # Initialise other factors
    Y = np.random.randn(baselineDim, no_dims)
    dY =  np.zeros((baselineDim, no_dims))
    iY = np.zeros((baselineDim, no_dims))
    gains = np.zeros((baselineDim, no_dims))
    P = Xi
    # Compute p-values
    for set in range(M):
        XsetTemp = Xpca[set]
        Xset = np.array(XsetTemp)
        (nI, dI) = Xset.shape
        # Compute p-values for each data-set
        Ptemp = x2p(Xset, 1e-5, perplexity)
        (d1,d2) = Ptemp.shape
        k = np.argwhere(np.isnan(Ptemp))
        (kDim1, kDim2) = k.shape
        for i in range(kDim1):
            d1 = k[i][0]
            d2 = k[i][1]
            Ptemp[d1][d2] = 0
        Ptemp = Ptemp + np.transpose(Ptemp)
        Ptemp = Ptemp / np.sum(Ptemp)
        Ptemp = Ptemp * 4.          # early exaggeration
        Ptemp = np.maximum(Ptemp, 1e-12)
        P[set] = Ptemp
    ## Run iterations
    for iter in range(max_iter):
        sum_Y = np.sum(np.square(Y),1) # Sum of columns squared element-wise
        num = -2. * np.dot(Y, Y.T) # matrix multiplication
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(baselineDim), range(baselineDim)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        # For each data-set
        for set in range(M):
            # Compute pairwise affinities
            # Compute gradient
            Ptemp = P[set]
            Pset = np.array(Ptemp)
            PQ = Pset - Q
            # Updates
            if set == 0:
                for i in range(baselineDim):
                    dY[i, :] = np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0)
            else:
                for i in range(baselineDim):
                    dY[i, :] = dY[i,:] + np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0)
        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
        (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y,0), (baselineDim,1))
        # Stop lying about P-Values
        for set in range(M):
            PsetTemp = P[set]
            Pset = np.array(PsetTemp)
            if (iter + 1) % 10 ==0:
                C = np.sum(Pset * np.log(Pset / Q))
                print("Iteration %d: error is %f" % (iter + 1, C))
            if iter == 100:
                Pset = Pset / 4.
                P[set] = Pset
    # Return solution
    return Y
# M-SNE
def mSNE(X = np.array([[]]), no_dims = 2, initial_dims = 50, perplexity = 30.0, max_iter = 1000):
    """
        Runs m-SNE on the array(list) X, which includes M datasets
        in the MxNxD_i for each dataset X_i to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = multi_SNE(X, no_dims, initial_dims,perplexity,max_iter),
        where X is an MxNxD_i, i from 1 to M NumPy array.
    """
    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1
    # Initialize variables
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    # Now, initialization for each data-set
    dim = X.shape
    M = dim[0]
    # Make sure each data-set has the same axis=0 dimension
    Xtemp = X[1]
    XsTemp = np.array(Xtemp)
    baselineDim = XsTemp.shape[0]
    for view in range(M):
        Xtemp = X[view]
        XsTemp = np.array(Xtemp)
        dimToCheck = XsTemp.shape[0]
        if dimToCheck != baselineDim:
            print("Error: Number of rows (samples) must be the same in all data-sets of list X.")
            return -1
    Xpca = multiPCA(X, initial_dims).real
    # Initialise a NumPy array of lists -- size=Num.of.data_views
    a = [(0,0),(0,0), (0,0)]
    b = [(0,0),(0,0)]
    if M==2:
        Xi = np.array([a,b])
    elif M ==3:
        Xi = np.array([a,b,b])
    elif M ==4:
        Xi = np.array([a,b,b,b])
    elif M ==5:
        Xi = np.array([a,b,b,b,b])
    elif M ==6:
        Xi = np.array([a,b,b,b,b,b])
    elif M ==7:
        Xi = np.array([a,b,b,b,b,b,b])
    elif M ==8:
        Xi = np.array([a,b,b,b,b,b,b,b])
    elif M ==9:
        Xi = np.array([a,b,b,b,b,b,b,b,b])
    elif M ==10:
        Xi = np.array([a,b,b,b,b,b,b,b,b,b])
    # Initialise other factors
    Y = np.random.randn(baselineDim, no_dims)
    dY =  np.zeros((baselineDim, no_dims))
    iY = np.zeros((baselineDim, no_dims))
    gains = np.zeros((baselineDim, no_dims))
    Ptemp = np.zeros((baselineDim, baselineDim))
    P = Xi
    # Compute p-values
    for view in range(M):
        XsetTemp = Xpca[view]
        Xset = np.array(XsetTemp)
        (nI, dI) = Xset.shape
        # Compute p-values for each data-set
        PtempOut = x2p(Xset, 1e-5, perplexity)
        (d1,d2) = PtempOut.shape
        k = np.argwhere(np.isnan(PtempOut))
        (kDim1, kDim2) = k.shape
        for i in range(kDim1):
            d1 = k[i][0]
            d2 = k[i][1]
            PtempOut[d1][d2] = 0
        PtempOut = PtempOut + np.transpose(PtempOut)
        PtempOut = PtempOut / np.sum(PtempOut)
        PtempOut = PtempOut * 4.          # early exaggeration
        PtempOut = np.maximum(PtempOut, 1e-12)
        Ptemp = PtempOut + Ptemp
        n = nI
    P = Ptemp
   # Run iterations
    for iter in range(max_iter):
        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))
        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))
        # Stop lying about P-values
        if iter == 100:
            P = P / 4.
    # Return solution
    return Y
## Semi-supervised ##
# Missing Data multi-PCA
def SmultiPCA(X=np.array([[]]), no_dims=50):
    """
        Runs PCA on M datasets, each being a NxD_i array in order to
        reduce their dimensionality to no_dims dimensions. PCA is applied
        on each dataset separately.
        Ignore NaN values in producing PCA on matrices with missing samples/labels
    """
    dim = X.shape
    M = dim[0]
    Yout = X
    for set in range(M):
        Yout[set] = Yout[set][:,0:no_dims]
        XsetTemp = X[set]
        Xset = np.array(XsetTemp)
        (n,d) = Xset.shape
        # Check for missing data
        Xset_df = pd.DataFrame(Xset)
        Xset_df_drop = Xset_df.dropna()
        index_NaN = Xset_df.index ^ Xset_df_drop.index
        index_noNaN = Xset_df.notna().any(axis = 1)
        Xset_df_NaN = pd.DataFrame(Xset[index_NaN,:])
        Xset_df_noNaN = Xset_df.drop(index_NaN)
        (n_noNaN, d_noNaN) = Xset_df_noNaN.shape
        # PCA
        Ytemp = Xset_df_noNaN - np.tile(np.mean(Xset_df_noNaN, 0), (n_noNaN,1))
        (l,m) = np.linalg.eig(np.dot(Ytemp.T, Ytemp))
        Y = np.dot(Ytemp, m[:, 0:no_dims])
        Yout[set][index_noNaN] = Y
        Yout[set][index_NaN] = np.NaN
    return Yout
# S-multi-SNE
# Missing data multi-SNE
def SmultiSNE(X = np.array([[]]), no_dims = 2, initial_dims = 50, perplexity = 30.0, max_iter = 1000):
    """
        Executes multi-SNE with missing data

        INPUT:  X : Numpy array [[]]
                no_dims : Dimension of the Embeddings
                initial_dims : Initial reduction via pca
                perplexity : tuning parameter
                max_iter : Number of max iterations
        OUTPUT: Y : Numpy array []
    """
    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1
    # Initialize variables
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    # Now, initialization for each data-set
    dim = X.shape
    M = dim[0]
    # Make sure each data-set has the same axis=0 dimension
    Xtemp = X[1]
    XsTemp = np.array(Xtemp)
    baselineDim = XsTemp.shape[0]
    for set in range(M):
        Xtemp = X[set]
        XsTemp = np.array(Xtemp)
        dimToCheck = XsTemp.shape[0]
        if dimToCheck != baselineDim:
            print("Error: Number of rows (samples) must be the same in all data-sets of list X.")
            return -1
    Xpca = SmultiPCA(X, initial_dims).real
    # Initialise a NumPy array of lists -- size=Num.of.data_views
    a = [(0,0),(0,0), (0,0)]
    b = [(0,0),(0,0)]
    if M==2:
        Xi = np.array([a,b])
    elif M ==3:
        Xi = np.array([a,b,b])
    elif M ==4:
        Xi = np.array([a,b,b,b])
    elif M ==5:
        Xi = np.array([a,b,b,b,b])
    elif M ==6:
        Xi = np.array([a,b,b,b,b,b])
    elif M ==7:
        Xi = np.array([a,b,b,b,b,b,b])
    elif M ==8:
        Xi = np.array([a,b,b,b,b,b,b,b])
    elif M ==9:
        Xi = np.array([a,b,b,b,b,b,b,b,b])
    elif M ==10:
        Xi = np.array([a,b,b,b,b,b,b,b,b,b])
    # Initialise other factors
    Y = np.random.randn(baselineDim, no_dims)
    dY =  np.zeros((baselineDim, no_dims))
    iY = np.zeros((baselineDim, no_dims))
    gains = np.zeros((baselineDim, no_dims))
    P = Xi
    # Compute p-values
    for set in range(M):
        XsetTemp = Xpca[set]
        Xset = np.array(XsetTemp)
        (nI, dI) = Xset.shape
        # Compute p-values for each data-set
        Ptemp = x2p(Xset, 1e-5, perplexity)
        (d1,d2) = Ptemp.shape
        k = np.argwhere(np.isnan(Ptemp))
        (kDim1, kDim2) = k.shape
        for i in range(kDim1):
            d1 = k[i][0]
            d2 = k[i][1]
            Ptemp[d1][d2] = 0
        Ptemp = Ptemp + np.transpose(Ptemp)
        Ptemp = Ptemp / np.sum(Ptemp)
        Ptemp = Ptemp * 4.          # early exaggeration
        Ptemp = np.maximum(Ptemp, 1e-12)
        P[set] = Ptemp
    ## Run iterations
    for iter in range(max_iter):
        sum_Y = np.sum(np.square(Y),1) # Sum of columns squared element-wise
        num = -2. * np.dot(Y, Y.T) # matrix multiplication
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(baselineDim), range(baselineDim)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        # For each data-set
        for set in range(M):
            # Compute pairwise affinities
            # Compute gradient
            Ptemp = P[set]
            Pset = np.array(Ptemp)
            PQ = Pset - Q
            if set == 0:
                for i in range(baselineDim):
                    # Check is NOT NaN
                    if (np.invert(np.isnan(X[set][i,:])).any()):
                        dY[i, :] = np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0)
            else:
                for i in range(baselineDim):
                    # Check is NOT NaN
                    if (np.invert(np.isnan(X[set][i,:])).any()):
                        dY[i, :] = dY[i,:] + np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0)
        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
        (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y,0), (baselineDim,1))
        # Stop lying about P-Values
        for set in range(M):
            PsetTemp = P[set]
            Pset = np.array(PsetTemp)
            if (iter + 1) % 10 ==0:
                C = np.sum(Pset * np.log(Pset / Q))
                print("Iteration %d: error is %f" % (iter + 1, C))
            if iter == 100:
                Pset = Pset / 4.
                P[set] = Pset
    # Return solution
    return Y
## Weighted ##
# Multi-SNE
def multiSNE_weights(X = np.array([[]]), no_dims = 2, initial_dims = 50, perplexity = 30.0,
                       max_iter = 1000,  weights = np.array([]), weightUpdating = True, lambdaParameter = 1):
    """
        Runs multi-SNE on the array(list) X, which includes M datasets
        in the MxNxD_i for each dataset X_i to reduce its
        dimensionality to no_dims dimensions.
        Allows the option of updating the weights automatically depending on the KL-divergence on each data-view
        The syntaxis of the function is
        `Y = multiSNE_weights(X, no_dims, initial_dims,perplexity,max_iter),
        where X is an MxNxD_i, i from 1 to M NumPy array.
    """
    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1
    # Initialize variables
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    # Now, initialization for each data-set
    dim = X.shape
    M = dim[0]
    if not weights: # Given empty weights vector
        w = np.ones(M) # Weights on each data-set
    else:
        w=weights # Weights on each data-set
    w = w / sum(w) # Sum to 1
    Ctemp = np.zeros(M)
    z = np.zeros(M) #  Used in the automatic update of weights
    Weights = np.zeros((max_iter, M))
    Errors = np.zeros((max_iter, M))
    # Make sure each data-set has the same axis=0 dimension
    Xtemp = X[1]
    XsTemp = np.array(Xtemp)
    baselineDim = XsTemp.shape[0]
    for set in range(M):
        Xtemp = X[set]
        XsTemp = np.array(Xtemp)
        dimToCheck = XsTemp.shape[0]
        if dimToCheck != baselineDim:
            print("Error: Number of rows (samples) must be the same in all data-sets of list X.")
            return -1
    Xpca = multiPCA(X, initial_dims).real
    # Initialise a NumPy array of lists -- size=Num.of.data_views
    a = [(0,0),(0,0), (0,0)]
    b = [(0,0),(0,0)]
    if M==2:
        Xi = np.array([a,b])
    elif M ==3:
        Xi = np.array([a,b,b])
    elif M ==4:
        Xi = np.array([a,b,b,b])
    elif M ==5:
        Xi = np.array([a,b,b,b,b])
    elif M ==6:
        Xi = np.array([a,b,b,b,b,b])
    elif M ==7:
        Xi = np.array([a,b,b,b,b,b,b])
    elif M ==8:
        Xi = np.array([a,b,b,b,b,b,b,b])
    elif M ==9:
        Xi = np.array([a,b,b,b,b,b,b,b,b])
    elif M ==10:
        Xi = np.array([a,b,b,b,b,b,b,b,b,b])
    # Initialise other factors
    Y = np.random.randn(baselineDim, no_dims)
    dY =  np.zeros((baselineDim, no_dims))
    iY = np.zeros((baselineDim, no_dims))
    gains = np.zeros((baselineDim, no_dims))
    P = Xi
    # Compute p-values
    for set in range(M):
        XsetTemp = Xpca[set]
        Xset = np.array(XsetTemp)
        (nI, dI) = Xset.shape
        # Compute p-values for each data-set
        Ptemp = x2p(Xset, 1e-5, perplexity)
        (d1,d2) = Ptemp.shape
        k = np.argwhere(np.isnan(Ptemp))
        (kDim1, kDim2) = k.shape
        for i in range(kDim1):
            d1 = k[i][0]
            d2 = k[i][1]
            Ptemp[d1][d2] = 0
        Ptemp = Ptemp + np.transpose(Ptemp)
        Ptemp = Ptemp / np.sum(Ptemp)
        Ptemp = Ptemp * 4.          # early exaggeration
        Ptemp = np.maximum(Ptemp, 1e-12)
        P[set] = Ptemp
    ## Run iterations
    for iter in range(max_iter):
        sum_Y = np.sum(np.square(Y),1) # Sum of columns squared element-wise
        num = -2. * np.dot(Y, Y.T) # matrix multiplication
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(baselineDim), range(baselineDim)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        # For each data-set
        for set in range(M):
            # Compute pairwise affinities
            # Compute gradient
            Ptemp = P[set]
            Pset = np.array(Ptemp)
            PQ = Pset - Q
            if set == 0:
                for i in range(baselineDim):
                    dY[i, :] = float(w[set])*np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0)
            else:
                for i in range(baselineDim):
                    dY[i, :] = dY[i,:] + float(w[set])*np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0)
        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
        (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y,0), (baselineDim,1))
        # Compute curent values of cost functions
        for set in range(M):
            PsetTemp = P[set]
            Pset = np.array(PsetTemp)
            if (iter + 1) % 10 ==0:
                C = np.sum(Pset * np.log(Pset / Q))
                print("Iteration %d: error is %f" % (iter + 1, C))
                #Ctemp[set] = C
                if np.isnan(C):
                    Ctemp[set] = 0
                else:
                    Ctemp[set] = C
                #z[set] = - C/lambdaParameter - 1
                #w[set] = np.sqrt( abs( (z[set]**2)*sum(np.delete(w**2, set)) ) / abs( 1-z[set]**2 ) )
            # Stop lying about P-Values
            if iter == 100:
                Pset = Pset / 4.
                P[set] = Pset
        if weightUpdating:
            wc = Ctemp.copy()
            if sum(wc)==0:
                wc = np.ones(M)
            wc = wc / sum(wc)
            w = 1-wc
            w = w / sum(w)
        else:
            for set in range(M):
                w[set] = 1/M
        Weights[iter,:] = w
        Errors[iter,:] = Ctemp
        if (iter + 1) % 10 ==0:
            print("Printing weight vector for iteration %d:" % (iter + 1))
            print(*w, sep = ", ")
    # Return solution
    return Y,Weights,Errors
# S-mult-SNE
def SmultiSNE_weights(X = np.array([[]]), no_dims = 2, initial_dims = 50, perplexity = 30.0,
                       max_iter = 1000,  weights = np.array([]), weightUpdating = True, lambdaParameter = 1):
    """
        Executes multi-SNE with missing data

        INPUT:  X : Numpy array [[]]
                weights: Numpy array [] -- weights per data-view
                weightUpdate: Either update weights per iteration, or keep the same throughout
                lambdaParameter: Parameter for weight updates
                no_dims : Dimension of the Embeddings
                initial_dims : Initial reduction via pca
                perplexity : tuning parameter
                max_iter : Number of max iterations
        OUTPUT: Y : Numpy array []
    """
    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1
    # Initialize variables
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    # Now, initialization for each data-set
    dim = X.shape
    M = dim[0]
    if not weights: # Given empty weights vector
        w = np.ones(M) # Weights on each data-set
    else:
        w=weights # Weights on each data-set
    w = w / sum(w) # Sum to 1
    Ctemp = np.zeros(M)
    z = np.zeros(M) #  Used in the automatic update of weights
    Weights = np.zeros((max_iter, M))
    Errors = np.zeros((max_iter, M))
    # Make sure each data-set has the same axis=0 dimension
    Xtemp = X[1]
    XsTemp = np.array(Xtemp)
    baselineDim = XsTemp.shape[0]
    for set in range(M):
        Xtemp = X[set]
        XsTemp = np.array(Xtemp)
        dimToCheck = XsTemp.shape[0]
        if dimToCheck != baselineDim:
            print("Error: Number of rows (samples) must be the same in all data-sets of list X.")
            return -1
    Xpca = SmultiPCA(X, initial_dims).real
    # Initialise a NumPy array of lists -- size=Num.of.data_views
    a = [(0,0),(0,0), (0,0)]
    b = [(0,0),(0,0)]
    if M==2:
        Xi = np.array([a,b])
    elif M ==3:
        Xi = np.array([a,b,b])
    elif M ==4:
        Xi = np.array([a,b,b,b])
    elif M ==5:
        Xi = np.array([a,b,b,b,b])
    elif M ==6:
        Xi = np.array([a,b,b,b,b,b])
    elif M ==7:
        Xi = np.array([a,b,b,b,b,b,b])
    elif M ==8:
        Xi = np.array([a,b,b,b,b,b,b,b])
    elif M ==9:
        Xi = np.array([a,b,b,b,b,b,b,b,b])
    elif M ==10:
        Xi = np.array([a,b,b,b,b,b,b,b,b,b])
    # Initialise other factors
    Y = np.random.randn(baselineDim, no_dims)
    dY =  np.zeros((baselineDim, no_dims))
    iY = np.zeros((baselineDim, no_dims))
    gains = np.zeros((baselineDim, no_dims))
    P = Xi
    # Compute p-values
    for set in range(M):
        XsetTemp = Xpca[set]
        Xset = np.array(XsetTemp)
        (nI, dI) = Xset.shape
        # Compute p-values for each data-set
        Ptemp = x2p(Xset, 1e-5, perplexity)
        (d1,d2) = Ptemp.shape
        k = np.argwhere(np.isnan(Ptemp))
        (kDim1, kDim2) = k.shape
        for i in range(kDim1):
            d1 = k[i][0]
            d2 = k[i][1]
            Ptemp[d1][d2] = 0
        Ptemp = Ptemp + np.transpose(Ptemp)
        Ptemp = Ptemp / np.sum(Ptemp)
        Ptemp = Ptemp * 4.          # early exaggeration
        Ptemp = np.maximum(Ptemp, 1e-12)
        P[set] = Ptemp
    ## Run iterations
    for iter in range(max_iter):
        sum_Y = np.sum(np.square(Y),1) # Sum of columns squared element-wise
        num = -2. * np.dot(Y, Y.T) # matrix multiplication
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(baselineDim), range(baselineDim)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        # For each data-set
        for set in range(M):
            # Compute pairwise affinities
            # Compute gradient
            Ptemp = P[set]
            Pset = np.array(Ptemp)
            PQ = Pset - Q
            if set == 0:
                for i in range(baselineDim):
                    # Check is NOT NaN
                    if (np.invert(np.isnan(X[set][i,:])).any()):
                        dY[i, :] = float(w[set])*np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0)
            else:
                for i in range(baselineDim):
                    # Check is NOT NaN
                    if (np.invert(np.isnan(X[set][i,:])).any()):
                        dY[i, :] = dY[i,:] + float(w[set])*np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0)
        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
        (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y,0), (baselineDim,1))
        # Compute curent values of cost functions
        for set in range(M):
            PsetTemp = P[set]
            Pset = np.array(PsetTemp)
            if (iter + 1) % 10 ==0:
                C = np.sum(Pset * np.log(Pset / Q))
                print("Iteration %d: error is %f" % (iter + 1, C))
                Ctemp[set] = C
                #z[set] = - C/lambdaParameter - 1
                #w[set] = np.sqrt( abs( (z[set]**2)*sum(np.delete(w**2, set)) ) / abs( 1-z[set]**2 ) )
            # Stop lying about P-Values
            if iter == 100:
                Pset = Pset / 4.
                P[set] = Pset
       if weightUpdating:
            wc = Ctemp.copy()
            if sum(wc)==0:
                wc = np.ones(M)
            wc = wc / sum(wc)
            w = 1-wc
            w = w / sum(w)
        else:
            for set in range(M):
                w[set] = 1/M
        Weights[iter,:] = w
        Errors[iter,:] = Ctemp
        if (iter + 1) % 10 ==0:
            print("Printing weight vector for iteration %d:" % (iter + 1))
            print(*w, sep = ", ")
    # Return solution
    return Y,Weights, Errors
