# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:12:58 2021

@author: Theodoulos Rodosthenous
"""

'''
In this script, we will use standard classification approaches to create a benchmark
in predicting the cell-types of a 'new' data-view, by training the model to the known cell-types of an 'existing' data-view

> Consider all combinations
> Follow the models and parameters as Huang et al. (2021)
> Models from scikit-sklearn
(A) LinearSVC from sklearn.svm
(B) LogisticRegression (solver = ‘lbfgs’) from sklearn.linear_model
(C) RandomForestClassifier() from sklearn.ensemble
(D) MLPClassifier (max_iter = 300) from sklearn.neural_network
(E) SVC (kernel = ‘rbf’, gamma= ‘auto’) from sklearn.svm
(F) DecisionTreeClassifier() from sklearn.tree
(G) AdaBoostClassifier() from sklearn.ensemble
(H) LinearDiscriminantAnalysis() from sklearn.discriminant_analysis
(I) KNeighborsClassifier() from sklearn.neighbors
(J) GaussianNB() from sklearn.naive_bayes
'''

# Basic
import numpy as np
import pandas as pd
# Models
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# Extras
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

# Scrap
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


## See .ipynb -- NOTEBOOK

############ Data
#No transpose
mouse1 = pd.read_csv("data/mouse1_data.txt", sep=" ", header=None)
mouse2 = pd.read_csv("data/mouse2_data.txt", sep=" ", header=None)
mouse3 = pd.read_csv("data/mouse3_data.txt", sep=" ", header=None)
mouseFetal = pd.read_csv("data/mouseFetal_data.txt", sep=" ", header=None)
# And take transpose
mouse1 = np.transpose(pd.read_csv("data/mouse1_data.txt", sep=" ", header=None))
mouse2 = np.transpose(pd.read_csv("data/mouse2_data.txt", sep=" ", header=None))
mouse3 = np.transpose(pd.read_csv("data/mouse3_data.txt", sep=" ", header=None))
mouseFetal = np.transpose(pd.read_csv("data/mouseFetal_data.txt", sep=" ", header=None))

# Read cell-types
'''
celltypesNames_mouse1 = pd.read_csv("data/mouse1_labels.txt", sep=" ", header=None)
celltypesNames_mouse2 = pd.read_csv("data/mouse2_labels.txt", sep=" ", header=None)
celltypesNames_mouse3 = pd.read_csv("data/mouse3_labels.txt", sep=" ", header=None)
celltypesNames_mouseFetal = pd.read_csv("data/mouseFetal_labels.txt", sep=" ", header=None)
'''

celltypes_mouse1 = pd.read_csv("data/mouse1_labels_num.txt", sep=" ", header=None)
celltypes_mouse2 = pd.read_csv("data/mouse2_labels_num.txt", sep=" ", header=None)
celltypes_mouse3 = pd.read_csv("data/mouse3_labels_num.txt", sep=" ", header=None)
celltypes_mouseFetal = pd.read_csv("data/mouseFetal_labels_num.txt", sep=" ", header=None)

## PCA Normalisation ##
# mouse1
pca_mouse1 = PCA(n_components=600)
reduced_mouse1 = pca_mouse1.fit_transform(mouse1)
# mouse2
pca_mouse2 = PCA(n_components=600)
reduced_mouse2 = pca_mouse2.fit_transform(mouse2)
# mouse3
pca_mouse3 = PCA(n_components=600)
reduced_mouse3 = pca_mouse3.fit_transform(mouse3)
# mouseFetal
pca_mouseFetal = PCA(n_components=600)
reduced_mouseFetal = pca_mouseFetal.fit_transform(mouseFetal)

## Training/Test data ##
# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
train_data = mouse1
train_lbl = celltypes_mouse1
test_data = mouse2
test_lbl = celltypes_mouse2

## Methods ##
############ (A) Linear Support Vector Machine ##########################
############ (B) Logistic Regression ##########################
############ (C) Random Forest ##########################
############ (D) Neural network ##########################
############ (E) Support Vector Machine with RBF kernel ##########################
############ (F) Decision Trees ##########################
############ (G) Decision Tree Regression with AdaBoost ##########################
############ (H) Linear Discriminant Analysis ##########################
############ (I) k Nearest Neighbors ##########################
############ (J) Naive Bayes ##########################

# Define function to run Models
def all_models(train_data, train_lbl, test_data, test_lbl):
    ## LINEAR SVM
    # set up model
    # Scaler and regression
    pipe_linearSVM = make_pipeline(StandardScaler(), LinearSVC(tol=1e-5))
    # train
    pipe_linearSVM.fit(train_data, train_lbl)
    # predict
    pred_test_linearSVM = pipe_linearSVM.predict(test_data)
    # accuracy
    acc_test_linearSVM = pipe_linearSVM.score(test_data, test_lbl)
    # other evaluation measures
    prec_linearSVM, rec_linearSVM, f1_linearSVM, sup_linearSVM = precision_recall_fscore_support(test_lbl, pred_test_linearSVM, average='weighted')
    acc_linearSVM = accuracy_score(test_lbl, pred_test_linearSVM)

    ## LOGISTIC REGRESSION
    # set up model
    # Scaler and regression
    pipe_LogisticR = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    # train
    pipe_LogisticR.fit(train_data, train_lbl)
    # predict
    pred_test_LogisticR = pipe_LogisticR.predict(test_data)
    # accuracy
    acc_test_LogisticR = pipe_LogisticR.score(test_data, test_lbl)
    # other evaluation measures
    prec_LogisticR, rec_LogisticR, f1_LogisticR, sup_LogisticR = precision_recall_fscore_support(test_lbl, pred_test_LogisticR, average='weighted')
    acc_LogisticR = accuracy_score(test_lbl, pred_test_LogisticR)

    ## RANDOM FOREST
    # set up model
    # Scaler and regression
    pipe_RF = make_pipeline(StandardScaler(), RandomForestClassifier(max_depth=2, random_state=0))
    # train
    pipe_RF.fit(train_data, train_lbl)
    # predict
    pred_test_RF = pipe_RF.predict(test_data)
    # accuracy
    acc_test_RF = pipe_RF.score(test_data, test_lbl)
    # other evaluation measures
    prec_RF, rec_RF, f1_RF, sup_RF = precision_recall_fscore_support(test_lbl, pred_test_RF, average='weighted')
    acc_RF = accuracy_score(test_lbl, pred_test_RF)

    ## NEURAL NETWORK
    # set up model
    # Scaler and regression
    pipe_NN = make_pipeline(StandardScaler(), MLPClassifier(max_iter=100))
    # train
    pipe_NN.fit(train_data, train_lbl)
    # predict
    pred_test_NN = pipe_NN.predict(test_data)
    # accuracy
    acc_test_NN = pipe_NN.score(test_data, test_lbl)
    # other evaluation measures
    prec_NN, rec_NN, f1_NN, sup_NN = precision_recall_fscore_support(test_lbl, pred_test_NN, average='weighted')
    acc_NN = accuracy_score(test_lbl, pred_test_NN)

    ## SVM w/RBF KERNEL (SVC)
    # set up model
    # Scaler and regression
    pipe_SVC = make_pipeline(StandardScaler(), SVC(kernel = 'rbf', gamma= 'auto'))
    # train
    pipe_SVC.fit(train_data, train_lbl)
    # predict
    pred_test_SVC = pipe_SVC.predict(test_data)
    # accuracy
    acc_test_SVC = pipe_SVC.score(test_data, test_lbl)
    # other evaluation measures
    prec_SVC, rec_SVC, f1_SVC, sup_SVC = precision_recall_fscore_support(test_lbl, pred_test_SVC, average='weighted')
    acc_SVC = accuracy_score(test_lbl, pred_test_SVC)

    ## DECISION TREES
    # set up model
    # Scaler and regression
    pipe_DecisionTrees = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=0))
    # train
    pipe_DecisionTrees.fit(train_data, train_lbl)
    # predict
    pred_test_DecisionTrees = pipe_DecisionTrees.predict(test_data)
    # accuracy
    acc_test_DecisionTrees = pipe_DecisionTrees.score(test_data, test_lbl)
    # other evaluation measures
    prec_DecisionTrees, rec_DecisionTrees, f1_DecisionTrees, sup_DecisionTrees = precision_recall_fscore_support(test_lbl, pred_test_DecisionTrees, average='weighted')
    acc_DecisionTrees = accuracy_score(test_lbl, pred_test_DecisionTrees)

    ## DECISION TREES w/ ADABOOST
    # set up model
    # Scaler and regression
    pipe_DT_AdaBoost = make_pipeline(StandardScaler(), AdaBoostClassifier())
    # train
    pipe_DT_AdaBoost.fit(train_data, train_lbl)
    # predict
    pred_test_DT_AdaBoost = pipe_DT_AdaBoost.predict(test_data)
    # accuracy
    acc_test_DT_AdaBoost = pipe_DT_AdaBoost.score(test_data, test_lbl)
    # other evaluation measures
    prec_DT_AdaBoost, rec_DT_AdaBoost, f1_DT_AdaBoost, sup_DT_AdaBoost = precision_recall_fscore_support(test_lbl, pred_test_DT_AdaBoost, average='weighted')
    acc_DT_AdaBoost = accuracy_score(test_lbl, pred_test_DT_AdaBoost)

    ## LDA
    # set up model
    # Scaler and regression
    pipe_LDA = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    # train
    pipe_LDA.fit(train_data, train_lbl)
    # predict
    pred_test_LDA = pipe_LDA.predict(test_data)
    # accuracy
    acc_test_LDA = pipe_LDA.score(test_data, test_lbl)
    # other evaluation measures
    prec_LDA, rec_LDA, f1_LDA, sup_LDA = precision_recall_fscore_support(test_lbl, pred_test_LDA, average='weighted')
    acc_LDA = accuracy_score(test_lbl, pred_test_LDA)

    # K-NN
    # set up model
    # Scaler and regression
    pipe_KNN = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=10))
    # train
    pipe_KNN.fit(train_data, train_lbl)
    # predict
    pred_test_KNN = pipe_KNN.predict(test_data)
    # accuracy
    acc_test_KNN = pipe_KNN.score(test_data, test_lbl)
    # other evaluation measures
    prec_KNN, rec_KNN, f1_KNN, sup_KNN = precision_recall_fscore_support(test_lbl, pred_test_KNN, average='weighted')
    acc_KNN = accuracy_score(test_lbl, pred_test_KNN)

    ## NAIVE BAYES
    # set up model
    # Scaler and regression
    pipe_NaiveBayes = make_pipeline(StandardScaler(), GaussianNB())
    # train
    pipe_NaiveBayes.fit(train_data, train_lbl)
    # predict
    pred_test_NaiveBayes = pipe_NaiveBayes.predict(test_data)
    # accuracy
    acc_test_NaiveBayes = pipe_NaiveBayes.score(test_data, test_lbl)
    # other evaluation measures
    prec_NB, rec_NB, f1_NB, sup_NB = precision_recall_fscore_support(test_lbl, pred_test_NaiveBayes, average='weighted')
    acc_NB = accuracy_score(test_lbl, pred_test_NaiveBayes)
    # Overall scores
    scores = [acc_test_linearSVM,
              acc_test_LogisticR,
              acc_test_RF,
              acc_test_NN,
              acc_test_SVC,
              acc_test_DecisionTrees,
              acc_test_DT_AdaBoost,
              acc_test_LDA,
              acc_test_KNN,
              acc_test_NaiveBayes]
    # Additional scores
    # Accuracy
    scores_acc = [acc_linearSVM,
              acc_LogisticR,
              acc_RF,
              acc_NN,
              acc_SVC,
              acc_DecisionTrees,
              acc_DT_AdaBoost,
              acc_LDA,
              acc_KNN,
              acc_NB]
    # Precision
    scores_prec = [prec_linearSVM,
              prec_LogisticR,
              prec_RF,
              prec_NN,
              prec_SVC,
              prec_DecisionTrees,
              prec_DT_AdaBoost,
              prec_LDA,
              prec_KNN,
              prec_NB]
    # Recall
    scores_rec = [rec_linearSVM,
              rec_LogisticR,
              rec_RF,
              rec_NN,
              rec_SVC,
              rec_DecisionTrees,
              rec_DT_AdaBoost,
              rec_LDA,
              rec_KNN,
              rec_NB]
    # F1-score
    f1_score = [f1_linearSVM,
              f1_LogisticR,
              f1_RF,
              f1_NN,
              f1_SVC,
              f1_DecisionTrees,
              f1_DT_AdaBoost,
              f1_LDA,
              f1_KNN,
              f1_NB]
    return scores, scores_acc, scores_prec, scores_rec, f1_score


dataViews_names = ["mouse1", "mouse2", "mouse3"]
allData = [mouse1, mouse2, mouse3]
allCelltypes = [celltypes_mouse1,
                celltypes_mouse2,
                celltypes_mouse3]

allScores_df = pd.DataFrame(columns=['LinearSVM', 'LogisticR', 'RF', 'NN', 'SVC',
                                      'DecisionTrees', 'AdaBoost', 'LDA', 'kNN', 'NaiveBayes'])
allAccScores_df = pd.DataFrame(columns=['LinearSVM', 'LogisticR', 'RF', 'NN', 'SVC',
                                      'DecisionTrees', 'AdaBoost', 'LDA', 'kNN', 'NaiveBayes'])
allPrecScores_df = pd.DataFrame(columns=['LinearSVM', 'LogisticR', 'RF', 'NN', 'SVC',
                                      'DecisionTrees', 'AdaBoost', 'LDA', 'kNN', 'NaiveBayes'])
allRecScores_df = pd.DataFrame(columns=['LinearSVM', 'LogisticR', 'RF', 'NN', 'SVC',
                                      'DecisionTrees', 'AdaBoost', 'LDA', 'kNN', 'NaiveBayes'])
allF1Scores_df = pd.DataFrame(columns=['LinearSVM', 'LogisticR', 'RF', 'NN', 'SVC',
                                      'DecisionTrees', 'AdaBoost', 'LDA', 'kNN', 'NaiveBayes'])
train_data = []
test_data = []
for train in range(len(allData)):
    for test in range(len(allData)):
        scores_temp, acc_scores_temp, prec_scores_temp, rec_scores_temp, f1_scores_temp = all_models(train_data = allData[train],
                         train_lbl = allCelltypes[train],
                         test_data = allData[test],
                         test_lbl = allCelltypes[test])
        train_data.append(dataViews_names[train])
        test_data.append(dataViews_names[test])
        allScores_df = allScores_df.append(pd.DataFrame([scores_temp], columns=allScores_df.columns))
        allAccScores_df = allAccScores_df.append(pd.DataFrame([acc_scores_temp], columns=allAccScores_df.columns))
        allPrecScores_df = allPrecScores_df.append(pd.DataFrame([prec_scores_temp], columns=allPrecScores_df.columns))
        allRecScores_df = allRecScores_df.append(pd.DataFrame([rec_scores_temp], columns=allRecScores_df.columns))
        allF1Scores_df = allF1Scores_df.append(pd.DataFrame([f1_scores_temp], columns=allF1Scores_df.columns))


allScores_df['Train'] = train_data
allScores_df['Test'] = test_data
allAccScores_df['Train'] = train_data
allAccScores_df['Test'] = test_data
allPrecScores_df['Train'] = train_data
allPrecScores_df['Test'] = test_data
allRecScores_df['Train'] = train_data
allRecScores_df['Test'] = test_data
allF1Scores_df['Train'] = train_data
allF1Scores_df['Test'] = test_data

## Store data ##

allScores_df.to_csv("allResults_df.csv", index=True)
allAccScores_df.to_csv("allAccScores_df.csv", index=True)
allPrecScores_df.to_csv("allPrecScores_df.csv", index=True)
allRecScores_df.to_csv("allRecScores_df.csv", index=True)
allF1Scores_df.to_csv("allF1Scores_df.csv", index=True)
