# -*- coding: utf-8 -*-
"""
Created on Feb 25 2025

@author: Theodoulos Rodosthenous
"""

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
from sklearn.cross_decomposition import CCA
#from sklearn.preprocessing import StandardScaler
#import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


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


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
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


# PCA

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

    return Yout

def multi_SNE(X = np.array([[]]), no_dims = 2, initial_dims = 50, perplexity = 30.0, max_iter = 1000):
    """
        Runs t-SNE on the array(list) X, which includes M datasets
        in the NxD_i for each dataset X_i to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity),
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


    Xpca = multiPCA(X, initial_dims).real
    a = [(0,0),(0,0), (0,0)]
    b = [(0,0),(0,0)]
    c = [(0,0),(0,0)]
    d = [(0,0),(0,0)]
    e = [(0,0),(0,0)]
    f = [(0,0),(0,0)]
    g = [(0,0),(0,0)]
    if M==2:
        Xi = np.array([a,b])
    elif M ==3:
        Xi = np.array([a,b,c])
    elif M ==4:
        Xi = np.array([a,b,c,d])
    elif M ==5:
        Xi = np.array([a,b,c,d,e])
    elif M ==6:
        Xi = np.array([a,b,c,d,e,f])
    elif M ==7:
        Xi = np.array([a,b,c,d,e,f,g])
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

        # Compute curent values of cost functions



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



def mSNE(X = np.array([[]]), no_dims = 2, initial_dims = 50, perplexity = 30.0, max_iter = 1000):
    """
        Runs t-SNE on the array(list) X, which includes M datasets
        in the NxD_i for each dataset X_i to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity),
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
    a = [(0,0),(0,0), (0,0)]
    b = [(0,0),(0,0)]
    c = [(0,0),(0,0)]
    d = [(0,0),(0,0)]
    e = [(0,0),(0,0)]
    f = [(0,0),(0,0)]
    g = [(0,0),(0,0)]
    if M==2:
        Xi = np.array([a,b])
    elif M ==3:
        Xi = np.array([a,b,c])
    elif M ==4:
        Xi = np.array([a,b,c,d])
    elif M ==5:
        Xi = np.array([a,b,c,d,e])
    elif M ==6:
        Xi = np.array([a,b,c,d,e,f])
    elif M ==7:
        Xi = np.array([a,b,c,d,e,f,g])
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

import time

# Missing Data multi-PCA
def missingData_multiPCA(X=np.array([[]]), no_dims=50):
    """
        Runs PCA on M datasets, each being a NxD_i array in order to
        reduce their dimensionality to no_dims dimensions. PCA is applied
        on each dataset separately
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

# Missing data multi-SNE (S-multi-SNE)
def missingData_multiSNE(X = np.array([[]]), no_dims = 2, initial_dims = 50, perplexity = 30.0, max_iter = 1000):
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
    Xpca = missingData_multiPCA(X, initial_dims).real
    a = [(0,0),(0,0), (0,0)]
    b = [(0,0),(0,0)]
    c = [(0,0),(0,0)]
    d = [(0,0),(0,0)]
    e = [(0,0),(0,0)]
    f = [(0,0),(0,0)]
    g = [(0,0),(0,0)]
    if M==2:
        Xi = np.array([a,b])
    elif M ==3:
        Xi = np.array([a,b,c])
    elif M ==4:
        Xi = np.array([a,b,c,d])
    elif M ==5:
        Xi = np.array([a,b,c,d,e])
    elif M ==6:
        Xi = np.array([a,b,c,d,e,f])
    elif M ==7:
        Xi = np.array([a,b,c,d,e,f,g])
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

def autoWeights_multiSNE(X = np.array([[]]), no_dims = 2, initial_dims = 50,
                      perplexity = 30.0, max_iter = 1000, weightUpdating = True, lambdaParameter = 1):
    """
        Runs t-SNE on the array(list) X, which includes M datasets
        in the NxD_i for each dataset X_i to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity),
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
    w = np.ones(M) # Weights on each data-set
    w = w / sum(w) # Sum to 1
    Ctemp = np.zeros(M)
    z = np.zeros(M) #  Used in the automatic update of weights
    Weights = np.zeros((max_iter, M))
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
    a = [(0,0),(0,0), (0,0)]
    b = [(0,0),(0,0)]
    c = [(0,0),(0,0)]
    d = [(0,0),(0,0)]
    e = [(0,0),(0,0)]
    f = [(0,0),(0,0)]
    g = [(0,0),(0,0)]
    if M==2:
        Xi = np.array([a,b])
    elif M ==3:
        Xi = np.array([a,b,c])
    elif M ==4:
        Xi = np.array([a,b,c,d])
    elif M ==5:
        Xi = np.array([a,b,c,d,e])
    elif M ==6:
        Xi = np.array([a,b,c,d,e,f])
    elif M ==7:
        Xi = np.array([a,b,c,d,e,f,g])
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
        if (iter + 1) % 10 ==0:
            print("Printing weight vector for iteration %d:" % (iter + 1))
            print(*w, sep = ", ")


    # Return solution
    return Y,Weights

def manualWeights_multiSNE(X = np.array([[]]), no_dims = 2, initial_dims = 50,
                      perplexity = 30.0, max_iter = 1000, weights = np.array([])):
    """
        Runs t-SNE on the array(list) X, which includes M datasets
        in the NxD_i for each dataset X_i to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity),
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
    Ctemp = np.zeros(M)
    w = weights
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
    a = [(0,0),(0,0), (0,0)]
    b = [(0,0),(0,0)]
    c = [(0,0),(0,0)]
    d = [(0,0),(0,0)]
    e = [(0,0),(0,0)]
    f = [(0,0),(0,0)]
    g = [(0,0),(0,0)]
    if M==2:
        Xi = np.array([a,b])
    elif M ==3:
        Xi = np.array([a,b,c])
    elif M ==4:
        Xi = np.array([a,b,c,d])
    elif M ==5:
        Xi = np.array([a,b,c,d,e])
    elif M ==6:
        Xi = np.array([a,b,c,d,e,f])
    elif M ==7:
        Xi = np.array([a,b,c,d,e,f,g])
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
                Ctemp[set] = C
            # Stop lying about P-Values
            if iter == 100:
                Pset = Pset / 4.
                P[set] = Pset

    # Return solution
    return Y


def manualWeights_missingData_multiSNE(X = np.array([[]]), no_dims = 2, initial_dims = 50,
                      perplexity = 30.0, max_iter = 1000, weights = np.array([])):
    """
        Runs t-SNE on the array(list) X, which includes M datasets
        in the NxD_i for each dataset X_i to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity),
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
    Ctemp = np.zeros(M)
    w = weights
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


    Xpca = missingData_multiPCA(X, initial_dims).real
    a = [(0,0),(0,0), (0,0)]
    b = [(0,0),(0,0)]
    c = [(0,0),(0,0)]
    d = [(0,0),(0,0)]
    e = [(0,0),(0,0)]
    f = [(0,0),(0,0)]
    g = [(0,0),(0,0)]
    if M==2:
        Xi = np.array([a,b])
    elif M ==3:
        Xi = np.array([a,b,c])
    elif M ==4:
        Xi = np.array([a,b,c,d])
    elif M ==5:
        Xi = np.array([a,b,c,d,e])
    elif M ==6:
        Xi = np.array([a,b,c,d,e,f])
    elif M ==7:
        Xi = np.array([a,b,c,d,e,f,g])
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
            # Stop lying about P-Values
            if iter == 100:
                Pset = Pset / 4.
                P[set] = Pset

    # Return solution
    return Y

def autoWeights_missingData_multiSNE(X = np.array([[]]), no_dims = 2, initial_dims = 50,
                      perplexity = 30.0, max_iter = 1000, weightUpdating = True, lambdaParameter = 1):
    """
        Runs t-SNE on the array(list) X, which includes M datasets
        in the NxD_i for each dataset X_i to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity),
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
    w = np.ones(M) # Weights on each data-set
    w = w / sum(w) # Sum to 1
    Ctemp = np.zeros(M)
    z = np.zeros(M) #  Used in the automatic update of weights
    Weights = np.zeros((max_iter, M))
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


    Xpca = missingData_multiPCA(X, initial_dims).real
    a = [(0,0),(0,0), (0,0)]
    b = [(0,0),(0,0)]
    c = [(0,0),(0,0)]
    d = [(0,0),(0,0)]
    e = [(0,0),(0,0)]
    f = [(0,0),(0,0)]
    g = [(0,0),(0,0)]
    if M==2:
        Xi = np.array([a,b])
    elif M ==3:
        Xi = np.array([a,b,c])
    elif M ==4:
        Xi = np.array([a,b,c,d])
    elif M ==5:
        Xi = np.array([a,b,c,d,e])
    elif M ==6:
        Xi = np.array([a,b,c,d,e,f])
    elif M ==7:
        Xi = np.array([a,b,c,d,e,f,g])
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
                z[set] = - C/lambdaParameter - 1
                w[set] = np.sqrt( abs( (z[set]**2)*sum(np.delete(w**2, set)) ) / abs( 1-z[set]**2 ) )
            # Stop lying about P-Values
            if iter == 100:
                Pset = Pset / 4.
                P[set] = Pset

        if weightUpdating:
            w = w / sum(w)
        else:
            for set in range(M):
                w[set] = 1/M

        Weights[iter,:] = w
        if (iter + 1) % 10 ==0:
            print("Printing weight vector for iteration %d:" % (iter + 1))
            print(*w, sep = ", ")


    # Return solution
    return Y,Weights


print("processing..")
import time
Xa_orig = np.loadtxt("data_panc8.txt")
colVector = np.loadtxt("celltype_num_panc8.txt")

Xi = np.array([[(0,0),(0,0), (0,0)],
                      [(0,0)]])

# get transpose
Xa = np.transpose(Xa_orig).copy()

# Prepare colVector
labels = colVector.reshape(colVector.shape[0],1)
## Sample missing labels


label_matrix = np.loadtxt("label_matrix.txt")
nan_index = np.loadtxt("nan_index.txt")
zero_label_matrix = np.loadtxt("zero_label_matrix.txt")
nan_label_matrix = zero_label_matrix.copy()
for i in range(len(nan_index)):
  nan_label_matrix[int(nan_index[i]),:] = np.nan

## Run t-SNE ## 
perp = 2
start_time_tSNE = time.time()
Y_tSNE = tsne(Xa, 2, 50, perp)
end_time_tSNE = time.time()
running_time_tSNE = end_time_tSNE - start_time_tSNE
np.savetxt("Y_tSNE_p2_noLabels_panc8.txt", Y_tSNE)
np.savetxt("running_time_tSNE_p2_noLabels_panc8.txt", [running_time_tSNE])


# True labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(label_matrix)
start_time_multiSNE = time.time()
Y_multiSNE = multi_SNE(Xinput, 2, 50,perp,1000)
end_time_multiSNE  = time.time()
running_time_multiSNE  = end_time_multiSNE  - start_time_multiSNE 
np.savetxt("Y_multiSNE_p2_trueLabels_panc8.txt", Y_multiSNE)
np.savetxt("running_time_multiSNE_p2_trueLabels_panc8.txt", [running_time_multiSNE])

# NaN Labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(nan_label_matrix)
start_time_SmultiSNE = time.time()
Y_SmultiSNE_nan = missingData_multiSNE(Xinput, 2, 50,perp,1000)
end_time_SmultiSNE  = time.time()
running_time_SmultiSNE  = end_time_SmultiSNE  - start_time_SmultiSNE
np.savetxt("Y_SmultiSNE_p2_nanLabelMatrix_panc8.txt", Y_SmultiSNE_nan)
np.savetxt("running_time_SmultiSNE_p2_nanLabelMatrix_panc8.txt", [running_time_SmultiSNE])

# Zero Labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(zero_label_matrix)
start_time_SmultiSNE = time.time()
Y_SmultiSNE_zero = missingData_multiSNE(Xinput, 2, 50,perp,1000)
end_time_SmultiSNE  = time.time()
running_time_SmultiSNE  = end_time_SmultiSNE  - start_time_SmultiSNE
np.savetxt("Y_SmultiSNE_p2_zeroLabelMatrix_panc8.txt", Y_SmultiSNE_zero)
np.savetxt("running_time_SmultiSNE_p2_zeroLabelMatrix_panc8.txt", [running_time_SmultiSNE])


## Run t-SNE ##
perp = 10
start_time_tSNE = time.time()
Y_tSNE = tsne(Xa, 2, 50, perp)
end_time_tSNE = time.time()
running_time_tSNE = end_time_tSNE - start_time_tSNE
np.savetxt("Y_tSNE_p10_noLabels_panc8.txt", Y_tSNE)
np.savetxt("running_time_tSNE_p10_noLabels_panc8.txt", [running_time_tSNE])


# True labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(label_matrix)
start_time_multiSNE = time.time()
Y_multiSNE = multi_SNE(Xinput, 2, 50,perp,1000)
end_time_multiSNE  = time.time()
running_time_multiSNE  = end_time_multiSNE  - start_time_multiSNE
np.savetxt("Y_multiSNE_p10_trueLabels_panc8.txt", Y_multiSNE)
np.savetxt("running_time_multiSNE_p10_trueLabels_panc8.txt", [running_time_multiSNE])

# NaN Labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(nan_label_matrix)
start_time_SmultiSNE = time.time()
Y_SmultiSNE_nan = missingData_multiSNE(Xinput, 2, 50,perp,1000)
end_time_SmultiSNE  = time.time()
running_time_SmultiSNE  = end_time_SmultiSNE  - start_time_SmultiSNE
np.savetxt("Y_SmultiSNE_p10_nanLabelMatrix_panc8.txt", Y_SmultiSNE_nan)
np.savetxt("running_time_SmultiSNE_p10_nanLabelMatrix_panc8.txt", [running_time_SmultiSNE])

# Zero Labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(zero_label_matrix)
start_time_SmultiSNE = time.time()
Y_SmultiSNE_zero = missingData_multiSNE(Xinput, 2, 50,perp,1000)
end_time_SmultiSNE  = time.time()
running_time_SmultiSNE  = end_time_SmultiSNE  - start_time_SmultiSNE
np.savetxt("Y_SmultiSNE_p10_zeroLabelMatrix_panc8.txt", Y_SmultiSNE_zero)
np.savetxt("running_time_SmultiSNE_p10_zeroLabelMatrix_panc8.txt", [running_time_SmultiSNE])


## Run t-SNE ##
perp = 20
start_time_tSNE = time.time()
Y_tSNE = tsne(Xa, 2, 50, perp)
end_time_tSNE = time.time()
running_time_tSNE = end_time_tSNE - start_time_tSNE
np.savetxt("Y_tSNE_p20_noLabels_panc8.txt", Y_tSNE)
np.savetxt("running_time_tSNE_p20_noLabels_panc8.txt", [running_time_tSNE])


# True labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(label_matrix)
start_time_multiSNE = time.time()
Y_multiSNE = multi_SNE(Xinput, 2, 50,perp,1000)
end_time_multiSNE  = time.time()
running_time_multiSNE  = end_time_multiSNE  - start_time_multiSNE
np.savetxt("Y_multiSNE_p20_trueLabels_panc8.txt", Y_multiSNE)
np.savetxt("running_time_multiSNE_p20_trueLabels_panc8.txt", [running_time_multiSNE])

# NaN Labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(nan_label_matrix)
start_time_SmultiSNE = time.time()
Y_SmultiSNE_nan = missingData_multiSNE(Xinput, 2, 50,perp,1000)
end_time_SmultiSNE  = time.time()
running_time_SmultiSNE  = end_time_SmultiSNE  - start_time_SmultiSNE
np.savetxt("Y_SmultiSNE_p20_nanLabelMatrix_panc8.txt", Y_SmultiSNE_nan)
np.savetxt("running_time_SmultiSNE_p20_nanLabelMatrix_panc8.txt", [running_time_SmultiSNE])

# Zero Labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(zero_label_matrix)
start_time_SmultiSNE = time.time()
Y_SmultiSNE_zero = missingData_multiSNE(Xinput, 2, 50,perp,1000)
end_time_SmultiSNE  = time.time()
running_time_SmultiSNE  = end_time_SmultiSNE  - start_time_SmultiSNE
np.savetxt("Y_SmultiSNE_p20_zeroLabelMatrix_panc8.txt", Y_SmultiSNE_zero)
np.savetxt("running_time_SmultiSNE_p20_zeroLabelMatrix_panc8.txt", [running_time_SmultiSNE])


## Run t-SNE ##
perp = 50
start_time_tSNE = time.time()
Y_tSNE = tsne(Xa, 2, 50, perp)
end_time_tSNE = time.time()
running_time_tSNE = end_time_tSNE - start_time_tSNE
np.savetxt("Y_tSNE_p50_noLabels_panc8.txt", Y_tSNE)
np.savetxt("running_time_tSNE_p50_noLabels_panc8.txt", [running_time_tSNE])


# True labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(label_matrix)
start_time_multiSNE = time.time()
Y_multiSNE = multi_SNE(Xinput, 2, 50,perp,1000)
end_time_multiSNE  = time.time()
running_time_multiSNE  = end_time_multiSNE  - start_time_multiSNE
np.savetxt("Y_multiSNE_p50_trueLabels_panc8.txt", Y_multiSNE)
np.savetxt("running_time_multiSNE_p50_trueLabels_panc8.txt", [running_time_multiSNE])

# NaN Labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(nan_label_matrix)
start_time_SmultiSNE = time.time()
Y_SmultiSNE_nan = missingData_multiSNE(Xinput, 2, 50,perp,1000)
end_time_SmultiSNE  = time.time()
running_time_SmultiSNE  = end_time_SmultiSNE  - start_time_SmultiSNE
np.savetxt("Y_SmultiSNE_p50_nanLabelMatrix_panc8.txt", Y_SmultiSNE_nan)
np.savetxt("running_time_SmultiSNE_p50_nanLabelMatrix_panc8.txt", [running_time_SmultiSNE])

# Zero Labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(zero_label_matrix)
start_time_SmultiSNE = time.time()
Y_SmultiSNE_zero = missingData_multiSNE(Xinput, 2, 50,perp,1000)
end_time_SmultiSNE  = time.time()
running_time_SmultiSNE  = end_time_SmultiSNE  - start_time_SmultiSNE
np.savetxt("Y_SmultiSNE_p50_zeroLabelMatrix_panc8.txt", Y_SmultiSNE_zero)
np.savetxt("running_time_SmultiSNE_p50_zeroLabelMatrix_panc8.txt", [running_time_SmultiSNE])



## Run t-SNE ##
perp = 80
start_time_tSNE = time.time()
Y_tSNE = tsne(Xa, 2, 50, perp)
end_time_tSNE = time.time()
running_time_tSNE = end_time_tSNE - start_time_tSNE
np.savetxt("Y_tSNE_p80_noLabels_panc8.txt", Y_tSNE)
np.savetxt("running_time_tSNE_p80_noLabels_panc8.txt", [running_time_tSNE])


# True labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(label_matrix)
start_time_multiSNE = time.time()
Y_multiSNE = multi_SNE(Xinput, 2, 50,perp,1000)
end_time_multiSNE  = time.time()
running_time_multiSNE  = end_time_multiSNE  - start_time_multiSNE
np.savetxt("Y_multiSNE_p80_trueLabels_panc8.txt", Y_multiSNE)
np.savetxt("running_time_multiSNE_p80_trueLabels_panc8.txt", [running_time_multiSNE])

# NaN Labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(nan_label_matrix)
start_time_SmultiSNE = time.time()
Y_SmultiSNE_nan = missingData_multiSNE(Xinput, 2, 50,perp,1000)
end_time_SmultiSNE  = time.time()
running_time_SmultiSNE  = end_time_SmultiSNE  - start_time_SmultiSNE
np.savetxt("Y_SmultiSNE_p80_nanLabelMatrix_panc8.txt", Y_SmultiSNE_nan)
np.savetxt("running_time_SmultiSNE_p80_nanLabelMatrix_panc8.txt", [running_time_SmultiSNE])

# Zero Labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(zero_label_matrix)
start_time_SmultiSNE = time.time()
Y_SmultiSNE_zero = missingData_multiSNE(Xinput, 2, 50,perp,1000)
end_time_SmultiSNE  = time.time()
running_time_SmultiSNE  = end_time_SmultiSNE  - start_time_SmultiSNE
np.savetxt("Y_SmultiSNE_p80_zeroLabelMatrix_panc8.txt", Y_SmultiSNE_zero)
np.savetxt("running_time_SmultiSNE_p80_zeroLabelMatrix_panc8.txt", [running_time_SmultiSNE])


## Run t-SNE ##
perp = 100
start_time_tSNE = time.time()
Y_tSNE = tsne(Xa, 2, 50, perp)
end_time_tSNE = time.time()
running_time_tSNE = end_time_tSNE - start_time_tSNE
np.savetxt("Y_tSNE_p100_noLabels_panc8.txt", Y_tSNE)
np.savetxt("running_time_tSNE_p100_noLabels_panc8.txt", [running_time_tSNE])


# True labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(label_matrix)
start_time_multiSNE = time.time()
Y_multiSNE = multi_SNE(Xinput, 2, 50,perp,1000)
end_time_multiSNE  = time.time()
running_time_multiSNE  = end_time_multiSNE  - start_time_multiSNE
np.savetxt("Y_multiSNE_p100_trueLabels_panc8.txt", Y_multiSNE)
np.savetxt("running_time_multiSNE_p100_trueLabels_panc8.txt", [running_time_multiSNE])

# NaN Labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(nan_label_matrix)
start_time_SmultiSNE = time.time()
Y_SmultiSNE_nan = missingData_multiSNE(Xinput, 2, 50,perp,1000)
end_time_SmultiSNE  = time.time()
running_time_SmultiSNE  = end_time_SmultiSNE  - start_time_SmultiSNE
np.savetxt("Y_SmultiSNE_p100_nanLabelMatrix_panc8.txt", Y_SmultiSNE_nan)
np.savetxt("running_time_SmultiSNE_p100_nanLabelMatrix_panc8.txt", [running_time_SmultiSNE])

# Zero Labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(zero_label_matrix)
start_time_SmultiSNE = time.time()
Y_SmultiSNE_zero = missingData_multiSNE(Xinput, 2, 50,perp,1000)
end_time_SmultiSNE  = time.time()
running_time_SmultiSNE  = end_time_SmultiSNE  - start_time_SmultiSNE
np.savetxt("Y_SmultiSNE_p100_zeroLabelMatrix_panc8.txt", Y_SmultiSNE_zero)
np.savetxt("running_time_SmultiSNE_p100_zeroLabelMatrix_panc8.txt", [running_time_SmultiSNE])

## Run t-SNE ##
perp = 200
start_time_tSNE = time.time()
Y_tSNE = tsne(Xa, 2, 50, perp)
end_time_tSNE = time.time()
running_time_tSNE = end_time_tSNE - start_time_tSNE
np.savetxt("Y_tSNE_p200_noLabels_panc8.txt", Y_tSNE)
np.savetxt("running_time_tSNE_p200_noLabels_panc8.txt", [running_time_tSNE])


# True labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(label_matrix)
start_time_multiSNE = time.time()
Y_multiSNE = multi_SNE(Xinput, 2, 50,perp,1000)
end_time_multiSNE  = time.time()
running_time_multiSNE  = end_time_multiSNE  - start_time_multiSNE
np.savetxt("Y_multiSNE_p200_trueLabels_panc8.txt", Y_multiSNE)
np.savetxt("running_time_multiSNE_p200_trueLabels_panc8.txt", [running_time_multiSNE])

# NaN Labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(nan_label_matrix)
start_time_SmultiSNE = time.time()
Y_SmultiSNE_nan = missingData_multiSNE(Xinput, 2, 50,perp,1000)
end_time_SmultiSNE  = time.time()
running_time_SmultiSNE  = end_time_SmultiSNE  - start_time_SmultiSNE
np.savetxt("Y_SmultiSNE_p200_nanLabelMatrix_panc8.txt", Y_SmultiSNE_nan)
np.savetxt("running_time_SmultiSNE_p200_nanLabelMatrix_panc8.txt", [running_time_SmultiSNE])

# Zero Labels
Xinput = np.copy(Xi)
Xinput[0] = np.copy(Xa)
Xinput[1] = np.copy(zero_label_matrix)
start_time_SmultiSNE = time.time()
Y_SmultiSNE_zero = missingData_multiSNE(Xinput, 2, 50,perp,1000)
end_time_SmultiSNE  = time.time()
running_time_SmultiSNE  = end_time_SmultiSNE  - start_time_SmultiSNE
np.savetxt("Y_SmultiSNE_p200_zeroLabelMatrix_panc8.txt", Y_SmultiSNE_zero)
np.savetxt("running_time_SmultiSNE_p200_zeroLabelMatrix_panc8.txt", [running_time_SmultiSNE])









