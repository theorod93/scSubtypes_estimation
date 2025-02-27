# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:10:07 2020

@author: Theodoulos Rodosthenous
"""

'''
In this script, we include all solutions in multi-view data visualisation, namely:
    (A) multi-SNE
    (B) multi-LLE
    (C) multi-ISOMAP

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

from sklearn.cross_decomposition import CCA
#from sklearn.preprocessing import StandardScaler 
#import matplotlib
import matplotlib.pyplot as plt

#################################################################################################
############ (C) multi-SNE ##########################
#################################################################################################

'''
Multi-view SNE:
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
    P = P * 4.									# early exaggeration
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

# CCA



# t-SNE

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

#################################################################################################
############ (C) multi-ISOMAP ##########################
#################################################################################################


'''
Multi-view ISOMAP:
Input: X = (X_1,..., X_M), where M is the number of views
    X_m \in R^{NxD_m}, for each m = {1,...,M}
Output: Y \in R^{Nxd}, d:usually equals to 2 for good 2D visualisation
Steps: On three separate scenarios:
(A) multi_isomap_graph
    Define graphs G_m~(V, E_m), for each view m = {1,...,M}
    V: Number of n_samples
    E_m: Edges, for which the length is defined by the distance between two vertices
*   Combine all graphs into a single graph \hat{G} by
        (i) SNF
        (ii) Average
    Remaining steps remain the same as standard (single-view) ISOMAP
(B) multi_isomap_path
    Define graphs G_m~(V, E_m), for each view m = {1,...,M}
    V: Number of n_samples
    E_m: Edges, for which the length is defined by the distance between two vertices
    Compute shortest paths D_m = {d_m(i,j), \forall i,j \in V}, for each view m = {1,...,M}, by taking the distances between
    all pairs of points in G_m
*   Combine all shortest paths into a single shortest path measure by
        (i) Minial between all D_m
            (a) Avoiding zero-valued D_m
            (b) Considering zero-valued D_m
        (ii) Average
    Remaining steps remain the same as standard (single-view) ISOMAP
(C) multi_isomap_embeddings
    Define graphs G_m~(V, E_m), for each view m = {1,...,M}
    V: Number of n_samples
    E_m: Edges, for which the length is defined by the distance between two vertices
    Compute shortest paths D_m = {d_m(i,j), \forall i,j \in V}, for each view m = {1,...,M}, by taking the distances between
    all pairs of points in G_m
    Compute d-dimensional embeddings for each D_m into Y_m, for each m = {1,...,M}
*   Combine the final d-dimensional embedding into a single \hat{Y}  by
        (i) Average
'''

import numpy as np
from sklearn.utils.graph import graph_shortest_path

def multi_isomap_graph(X, n_components=2, n_neighbors=6, method = 'average'):
    # Follow the (A) scenario for multi-view ISOMAP
    M = X.shape[0]

    # Compute distance matrix for each view
    print("Computing distance matrix for each view")
    G = X
    for view in range(M):
        g_temp,_ = distance_mat(X[view], n_neighbors)
        G[view] = g_temp
    # Combine distance matrix
    print("Combining the distance matrices")
    if method == 'average':
        G_total = G[0]
        for view in range(1,M):
            G_total = G_total + G[view]
        G_final = G_total/M
    elif method == 'snf':
        print("TODO")
        # # TODO:
    else:
        print("Please provide a method between 'average' and 'snf'")

    # Compute shortest paths from distance matrix
    print("Computing the shortest paths")
    graph = graph_shortest_path(G_final, directed=False)
    graph = -0.5 * (graph ** 2)

    # Return the MDS projection on the shortest paths graph
    return mds(graph, n_components)

def multi_isomap_path(X, n_components=2, n_neighbors=6, method = 'average', zero_valued = False):
    # Follow the (B) scenario for multi-view ISOMAP
    M = X.shape[0]

    all_graphs = X
    print("Computing distance matrix for each view")
    print("Computing the shortest paths for each view")
    for view in range(M):
        # Compute distance matrix for each view
        g_temp,_ = distance_mat(X[view], n_neighbors)
        # Compute shortest paths from distance matrix
        graph_temp = graph_shortest_path(g_temp, directed=False)
        graph_temp = -0.5 * (graph_temp ** 2)
        all_graphs[view] = graph_temp

    # Combine shortest paths
    print("Combining the shortest paths")
    if method == 'average':
        graph_total = all_graphs[0]
        for view in range(1,M):
            graph_total = graph_total + all_graphs[view]
        graphs_final = graph_total/M
    elif method == 'minimal':
        if zero_valued:
            print("TODO")
            # # TODO:
        else:
            print("TODO")
            # # TODO:
        # # TODO:
    else:
        print("Please provide a method between 'average' and 'snf'")


    # Return the MDS projection on the combined shortest paths graph
    return mds(graphs_final, n_components)

def multi_isomap_embeddings(X, n_components=2, n_neighbors=6):
    # Follow the (C) scenario for multi-view ISOMAP
    M = X.shape[0]

    embeddings = X
    print("Computing ISOMAP for each view")
    for view in range(M):
        # Compute ISOMAP for each view
        embedding_temp = isomap(X[view], n_components=n_components, n_neighbors=n_neighbors)
        embeddings[view] = embedding_temp

    # Combine d-dimensional embeddings
    print("Combining d-dimensional embeddings")
    embedding_total = embeddings[0]
    for view in range(1,M):
        embedding_total = embedding_total + embeddings[view]
    embedding_final = embedding_total/M

    # Return the MDS projection on the combined shortest paths graph
    return embedding_final


def isomap(data, n_components=2, n_neighbors=6):
    """
    Dimensionality reduction with isomap algorithm
    :param data: input image matrix of shape (n,m) if dist=False, square distance matrix of size (n,n) if dist=True
    :param n_components: number of components for projection
    :param n_neighbors: number of neighbors for distance matrix computation
    :return: Projected output of shape (n_components, n)
    """
    # Compute distance matrix
    data, _ = distance_mat(data, n_neighbors)

    # Compute shortest paths from distance matrix

    graph = graph_shortest_path(data, directed=False)
    graph = -0.5 * (graph ** 2)

    # Return the MDS projection on the shortest paths graph
    return mds(graph, n_components)

def distance_mat(X, n_neighbors=6):
    """
    Compute the square distance matrix using Euclidean distance
    :param X: Input data, a numpy array of shape (img_height, img_width)
    :param n_neighbors: Number of nearest neighbors to consider, int
    :return: numpy array of shape (img_height, img_height), numpy array of shape (img_height, n_neighbors)
    """
    def dist(a, b):
        return np.sqrt(sum((a - b)**2))

    # Compute full distance matrix
    distances = np.array([[dist(p1, p2) for p2 in X] for p1 in X])

    # Keep only the 6 nearest neighbors, others set to 0 (= unreachable)
    neighbors = np.zeros_like(distances)
    sort_distances = np.argsort(distances, axis=1)[:, 1:n_neighbors+1]
    for k,i in enumerate(sort_distances):
        neighbors[k,i] = distances[k,i]
    return neighbors, sort_distances

def center(K):
    """
    Method to center the distance matrix
    :param K: numpy array of shape mxm
    :return: numpy array of shape mxm
    """
    n_samples = K.shape[0]

    # Mean for each row/column
    meanrows = np.sum(K, axis=0) / n_samples
    meancols = (np.sum(K, axis=1)/n_samples)[:, np.newaxis]

    # Mean across all rows (entire matrix)
    meanall = meanrows.sum() / n_samples

    K -= meanrows
    K -= meancols
    K += meanall
    return K


def mds(data, n_components=2):
    """
    Apply multidimensional scaling (aka Principal Coordinates Analysis)
    :param data: nxn square distance matrix
    :param n_components: number of components for projection
    :return: projected output of shape (n_components, n)
    """

    # Center distance matrix
    center(data)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_val_cov, eig_vec_cov = np.linalg.eig(data)
    eig_pairs = [
        (np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))
    ]
    # Select n_components eigenvectors with largest eigenvalues, obtain subspace transform matrix
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    eig_pairs = np.array(eig_pairs)
    matrix_w = np.hstack(
        [eig_pairs[i, 1].reshape(data.shape[1], 1) for i in range(n_components)]
    )

    # Return samples in new subspace
    return matrix_w

#################################################################################################
############ (C) multi-LLE ##########################
#################################################################################################

"""
Outline:
Input: X -- Multi-view data
Output: Y -- Single-view data, ideally with n_components = 2
Analysis on two approaches, two functions:
(A)
    1. Get weight matrix for each view (W^v)
    2. Averaged weight matrix, based on a coefficient (\alpha, where \sum_v{\alpha^v} = 1)
    3. Find Y by using the averaged weight matrix
(B)
    1. Get weight matrix for each view (W^v)
    2. Find Y^v based on weight matrix W^v, for each view v
    3. Take the averaged output of all Y^v, denoted Y, based on a coefficient (\beta, where \sum_v{\beta^v} = 1)

NOTE: All functions performed in the following code are part of LocallyLinearEmbedding from sklearn.manifold
        i.e. from sklearn.manifold import LocallyLinearEmbedding

    BUT, to test the performance on the method, we will create a new lle() function to be comparable
        with existing tsne() function. Similarly for multi_lle() against multiSNE()
"""




#### Single-view functions ####

def barycenter_weights(X, Z, reg=1e-3):
    """Compute barycenter weights of X from Y along the first axis
    We estimate the weights to assign to each point in Y[i] to recover
    the point X[i]. The barycenter weights sum to 1.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_dim)
    Z : array-like, shape (n_samples, n_neighbors, n_dim)
    reg : float, optional
        amount of regularization to add for the problem to be
        well-posed in the case of n_neighbors > n_dim
    Returns
    -------
    B : array-like, shape (n_samples, n_neighbors)
    Notes
    -----
    See developers note for more information.
    """
    X = check_array(X, dtype=FLOAT_DTYPES)
    Z = check_array(Z, dtype=FLOAT_DTYPES, allow_nd=True)

    n_samples, n_neighbors = X.shape[0], Z.shape[1]
    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)

    # this might raise a LinalgError if G is singular and has trace
    # zero
    for i, A in enumerate(Z.transpose(0, 2, 1)):
        C = A.T - X[i]  # broadcasting
        G = np.dot(C, C.T)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[::Z.shape[1] + 1] += R
        w = solve(G, v, sym_pos=True)
        B[i, :] = w / np.sum(w)
    return B

def barycenter_kneighbors_graph(X, n_neighbors, reg=1e-3, n_jobs=None):
    """Computes the barycenter weighted graph of k-Neighbors for points in X
    Parameters
    ----------
    X : {array-like, NearestNeighbors}
        Sample data, shape = (n_samples, n_features), in the form of a
        numpy array or a NearestNeighbors object.
    n_neighbors : int
        Number of neighbors for each sample.
    reg : float, optional
        Amount of regularization when solving the least-squares
        problem. Only relevant if mode='barycenter'. If None, use the
        default.
    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    Returns
    -------
    A : sparse matrix in CSR format, shape = [n_samples, n_samples]
        A[i, j] is assigned the weight of edge that connects i to j.
    See also
    --------
    sklearn.neighbors.kneighbors_graph
    sklearn.neighbors.radius_neighbors_graph
    """
    knn = NearestNeighbors(n_neighbors + 1, n_jobs=n_jobs).fit(X)
    X = knn._fit_X
    n_samples = X.shape[0]
    ind = knn.kneighbors(X, return_distance=False)[:, 1:]
    data = barycenter_weights(X, X[ind], reg=reg)
    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
    return csr_matrix((data.ravel(), ind.ravel(), indptr),
                      shape=(n_samples, n_samples))

def null_space(M, k, k_skip=1, eigen_solver='arpack', tol=1E-6, max_iter=100,
               random_state=None):
    """
    Find the null space of a matrix M.
    Parameters
    ----------
    M : {array, matrix, sparse matrix, LinearOperator}
        Input covariance matrix: should be symmetric positive semi-definite
    k : integer
        Number of eigenvalues/vectors to return
    k_skip : integer, optional
        Number of low eigenvalues to skip.
    eigen_solver : string, {'auto', 'arpack', 'dense'}
        auto : algorithm will attempt to choose the best method for input data
        arpack : use arnoldi iteration in shift-invert mode.
                    For this method, M may be a dense matrix, sparse matrix,
                    or general linear operator.
                    Warning: ARPACK can be unstable for some problems.  It is
                    best to try several random seeds in order to check results.
        dense  : use standard dense matrix operations for the eigenvalue
                    decomposition.  For this method, M must be an array
                    or matrix type.  This method should be avoided for
                    large problems.
    tol : float, optional
        Tolerance for 'arpack' method.
        Not used if eigen_solver=='dense'.
    max_iter : int
        Maximum number of iterations for 'arpack' method.
        Not used if eigen_solver=='dense'
    random_state : int, RandomState instance, default=None
        Determines the random number generator when ``solver`` == 'arpack'.
        Pass an int for reproducible results across multiple function calls.
        See :term: `Glossary <random_state>`.
    """
    if eigen_solver == 'auto':
        if M.shape[0] > 200 and k + k_skip < 10:
            eigen_solver = 'arpack'
        else:
            eigen_solver = 'dense'

    if eigen_solver == 'arpack':
        random_state = check_random_state(random_state)
        # initialize with [-1,1] as in ARPACK
        v0 = random_state.uniform(-1, 1, M.shape[0])
        try:
            eigen_values, eigen_vectors = eigsh(M, k + k_skip, sigma=0.0,
                                                tol=tol, maxiter=max_iter,
                                                v0=v0)
        except RuntimeError as msg:
            raise ValueError("Error in determining null-space with ARPACK. "
                             "Error message: '%s'. "
                             "Note that method='arpack' can fail when the "
                             "weight matrix is singular or otherwise "
                             "ill-behaved.  method='dense' is recommended. "
                             "See online documentation for more information."
                             % msg)

        return eigen_vectors[:, k_skip:], np.sum(eigen_values[k_skip:])
    elif eigen_solver == 'dense':
        if hasattr(M, 'toarray'):
            M = M.toarray()
        eigen_values, eigen_vectors = eigh(
            M, eigvals=(k_skip, k + k_skip - 1), overwrite_a=True)
        index = np.argsort(np.abs(eigen_values))
        return eigen_vectors[:, index], np.sum(eigen_values)
    else:
        raise ValueError("Unrecognized eigen_solver '%s'" % eigen_solver)


def lle(X=np.array([]), n_components=2, n_neighbors = 5, eigen_solver="auto", n_jobs=None, reg = 1e-3):
    '''
        X is the input matrix
        no_dims is the number of components for the embedding matrix Y
        n_neighbors is the number of neighbours to consider in running k-NN
        eigen_solver is the method use to find the lowest eigenvectors -> Y matrix
        n_jobs is the number of parallel jobs to run for neighbours search (optional)
        reg : float :regularization constant, multiplies the trace of the local covariance matrix of the distances.
    '''

    if eigen_solver not in ('auto', 'arpack', 'dense'):
        raise ValueError("unrecognized eigen_solver '%s'" % eigen_solver)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    nbrs.fit(X)
    X = nbrs._fit_X

    N, d_in = X.shape

    if n_components > d_in:
        raise ValueError("output dimension must be less than or equal "
                         "to input dimension")
    if n_neighbors >= N:
        raise ValueError(
            "Expected n_neighbors <= n_samples, "
            " but n_samples = %d, n_neighbors = %d" %
            (N, n_neighbors)
        )

    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")

    M_sparse = (eigen_solver != 'dense')

    W = barycenter_kneighbors_graph(
            nbrs, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs)

    if M_sparse:
        M = eye(*W.shape, format=W.format) - W
        M = (M.T * M).tocsr()
    else:
        M = (W.T * W - W.T - W).toarray()
        M.flat[::M.shape[0] + 1] += 1  # W = W - I = W - I

    return null_space(M, n_components, k_skip=1, eigen_solver=eigen_solver)


#### Multi-view LLE ####
## (A) -- Averaged Weight Matrix


def multiLLE_weight(X = np.array([[]]), n_components=2, n_neighbors = 5,
                    eigen_solver="auto", n_jobs=None, reg = 1e-3):
    '''
        X is the input matrix
            -- Multi-view data, in the same structure as multi-SNE
        no_dims is the number of components for the embedding matrix Y
        n_neighbors is the number of neighbours to consider in running k-NN
        eigen_solver is the method use to find the lowest eigenvectors -> Y matrix
        n_jobs is the number of parallel jobs to run for neighbours search (optional)
        reg : float :regularization constant, multiplies the trace of the local covariance matrix of the distances.

        In this function, we will follow the (A) solution
        i.e. take the averaged weight matrix out of all views
    '''
    # Now, initialization for each data-set
    dim = X.shape
    V = dim[0] # Number of views
    # Get Weight Matrix for each view
    W = X#.copy(deep=True)
    print("Computing Weight matrix for each view")
    for view in range(V):
        Xtemp = X[view]#.copy(deep=True)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
        nbrs.fit(Xtemp)
        Xtemp = nbrs._fit_X
        N, d_in = Xtemp.shape

        if n_components > d_in:
            raise ValueError("output dimension must be less than or equal "
                             "to input dimension")
        if n_neighbors >= N:
            raise ValueError(
                "Expected n_neighbors <= n_samples, "
                " but n_samples = %d, n_neighbors = %d" %
                (N, n_neighbors)
            )
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")

        Wtemp = barycenter_kneighbors_graph(
            nbrs, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs)
        W[view] = Wtemp#.copy(deep=True)

    # Take the averaged Weight Matrix
    print("Combining the weight matrices")
    W_total = W[0]
    for view in range(1,V):
        W_total =  W[view] + W_total
    W_averaged = W_total / V

    M_sparse = (eigen_solver != 'dense')

    print("Computing the d-dimensional embedding")
    if M_sparse:
        M = eye(*W_averaged.shape, format=W_averaged.format) - W_averaged
        M = (M.T * M).tocsr()
    else:
        M = (W_averaged.T * W_averaged - W_averaged.T - W_averaged).toarray()
        M.flat[::M.shape[0] + 1] += 1  #  = W - I = W - I

    return null_space(M, n_components, k_skip=1, eigen_solver=eigen_solver)


## (B) -- Averaged Embeddings

def multiLLE_embeddings(X = np.array([[]]), n_components=2, n_neighbors = 5,
                    eigen_solver="auto", n_jobs=None, reg = 1e-3):
    '''
        X is the input matrix
            -- Multi-view data, in the same structure as multi-SNE
        no_dims is the number of components for the embedding matrix Y
        n_neighbors is the number of neighbours to consider in running k-NN
        eigen_solver is the method use to find the lowest eigenvectors -> Y matrix
        n_jobs is the number of parallel jobs to run for neighbours search (optional)
        reg : float :regularization constant, multiplies the trace of the local covariance matrix of the distances.

        In this function, we will follow the (A) solution
        i.e. take the averaged weight matrix out of all views
    '''
    # Now, initialization for each data-set
    dim = X.shape
    V = dim[0] # Number of views
    # Get Weight Matrix for each view
    W = X#.copy(deep=True)
    Y = X
    print("Computing LLE for each view")
    for view in range(V):
        Xtemp = X[view]#.copy(deep=True)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
        nbrs.fit(Xtemp)
        Xtemp = nbrs._fit_X
        N, d_in = Xtemp.shape

        if n_components > d_in:
            raise ValueError("output dimension must be less than or equal "
                             "to input dimension")
        if n_neighbors >= N:
            raise ValueError(
                "Expected n_neighbors <= n_samples, "
                " but n_samples = %d, n_neighbors = %d" %
                (N, n_neighbors)
            )
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")

        Wtemp = barycenter_kneighbors_graph(
            nbrs, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs)

        M_sparse = (eigen_solver != 'dense')

        if M_sparse:
            M = eye(*Wtemp.shape, format=Wtemp.format) - Wtemp
            M = (M.T * M).tocsr()
        else:
            M = (Wtemp.T * Wtemp - Wtemp.T - Wtemp).toarray()
            M.flat[::M.shape[0] + 1] += 1  #  = W - I = W - I

        Y[view] = null_space(M, n_components, k_skip=1, eigen_solver=eigen_solver)

    # Take the averaged Weight Matrix
    print("Combining the d-dimensional embeddings")
    Y_total = Y[0][0]
    for view in range(1,V):
        Y_total =  Y[view][0] + Y_total
    Y_averaged = Y_total / V

    return Y_averaged


'''
Following Kayo Yin's solution on LLE
ref: https://towardsdatascience.com/step-by-step-signal-processing-with-machine-learning-manifold-learning-8e1bb192461c
     https://github.com/kayoyin/signal-processing/blob/master/dimensionality_reduction.py

'''

def LLE_kayo(data, n_components=2, n_neighbors=6):
    """
    Dimensionality reduction with FastLLE algorithm
    :param data: input image matrix of shape (n,m)
    :param n_components: number of components for projection
    :param n_neighbors: number of neighbors for the weight extraction
    :return: Projected output of shape (n_components, n)
    """
    # Compute the nearest neighbors
    _, neighbors_idx = distance_mat(data, n_neighbors)

    n = data.shape[0]
    w = np.zeros((n, n))
    for i in range(n):
        # Center the neighbors matrix
        k_indexes = neighbors_idx[i, :]
        neighbors = data[k_indexes, :] - data[i, :]

        # Compute the corresponding gram matrix
        gram_inv = np.linalg.pinv(np.dot(neighbors, neighbors.T))

        # Setting the weight values according to the lagrangian
        lambda_par = 2/np.sum(gram_inv)
        w[i, k_indexes] = lambda_par*np.sum(gram_inv, axis=1)/2
    m = np.subtract(np.eye(n), w)
    values, u = np.linalg.eigh(np.dot(np.transpose(m), m))
    return u[:, 1:n_components+1]

def multi_LLE_kayo(X, n_components=2, n_neighbors=6, method = 'weights'):
    '''
    Dimensionality reduction for Multi-view data
    based on the LLE_kayo function

    Parameters
    ----------
    X : TYPE
        Multi-view data.
    n_components : TYPE, optional
        Number of features to be produced (d).
        The default is 2.
    n_neighbors : TYPE, optional
        Number of Neighbors in creating the Weight matrix.
        The default is 6.
    method : 'weights' or 'embeddings'
        Which method to use in order to get LLE on multi-view data.
        The default is 'weights'.

    Returns
    -------
    Y : d-dimensional embedding (d = n_components)

    '''
    M = X.shape[0]
    W_final = X
    print("Computing the Weight matrices for each view")
    for view in range(M):
        data = X[view]
        n = data.shape[0]
        w = np.zeros((n, n))
        # Compute the nearest neighbors
        _, neighbors_idx = distance_mat(data, n_neighbors)
        for i in range(n):
            # Center the neighbors matrix
            k_indexes = neighbors_idx[i, :]
            neighbors = data[k_indexes, :] - data[i, :]

            # Compute the corresponding gram matrix
            gram_inv = np.linalg.pinv(np.dot(neighbors, neighbors.T))

            # Setting the weight values according to the lagrangian
            lambda_par = 2/np.sum(gram_inv)
            w[i, k_indexes] = lambda_par*np.sum(gram_inv, axis=1)/2
        W_final[view] = w

    if method == 'weights':
        print("Combining the weight matrices")
        # Combine weights
        W_total = W_final[0]
        for view in range(1,M):
            W_total = W_total + W_final[view]
        W_final = W_total/M
        m = np.subtract(np.eye(n), W_final)
        print("Computing the d-dimensional embedding")
        values, u = np.linalg.eigh(np.dot(np.transpose(m), m))
        return u[:, 1:n_components+1]
    elif method == 'embeddings':
        print("Computing the d-dimensional embeddings for each view")
        m = np.subtract(np.eye(n), W_final[0])
        values, u = np.linalg.eigh(np.dot(np.transpose(m), m))
        u_total = u
        for view in range(1,M):
            m = np.subtract(np.eye(n), W_final[view])
            values, u_temp = np.linalg.eigh(np.dot(np.transpose(m), m))
            u_total = u_total + u
        print("Combining the d-dimensional embeddings")
        u_final = u_total / M
        return u_final[:, 1:n_components+1]
    else:
        print("Please provide a viable method, i.e. one of 'weights' or 'embeddings'")
