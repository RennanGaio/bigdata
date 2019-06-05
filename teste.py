import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import json
import sys
import numpy as np
from numpy.linalg import norm
import math as m
import scipy
import sklearn
from sklearn import preprocessing
from sklearn import neighbors, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
import random

from sklearn.model_selection import GridSearchCV
#from sklearn.grid_search import GridSearchCV
from scipy.spatial.distance import cosine, euclidean

import random
from datetime import datetime
#from helper_functions import *
#from data_functions import *

from sklearn.datasets import load_svmlight_file

import xgboost as xgb

strategie = sys.argv[1]

print "###########process Started##########"
print "using "+ strategie+ " strategie!\n"

file = open("trabalho_final/train_set.csv")

mylines = file.read().split('\n')

list5=[]
list9=[]
alist = []
for i in xrange(1, len(mylines)):
    element = mylines[i].split(";")
    for j in xrange(0, len(element)):
        # print mylines[i][j]
        if element[j] != 'NA' and j == 4:
            list5.append(float(element[j]))

            #alist.append(j)
        elif element[j] != 'NA' and j == 8:
            list9.append(float(element[j]))
            #alist.append(j)
#print set(alist)

median=[0]*10
maxi=[1]*10
media=[0]*10

median[4]=np.median(list5)
median[8]=np.median(list9)
media[4]=np.mean(list5)
media[8]=np.mean(list9)
maxi[4]=max(list5)
maxi[8]=max(list9)

X = []
y = []
signal = 1
for line in mylines:
    if signal == 1:
        signal = 0
        continue
    element = line.split(";")
    try:
        d = [float(x) for x in element[0:-2]]
    except:
        for i in range(0, len(element[0:-2])):
            if element[i] == 'NA':
                if strategie=="Only_0":
                    element[i] = 0
                elif strategie=="media":
                    element[i] = media[i]
                elif strategie=="All_median":
                    element[i]=median[i]
                else:
                    element[i]=random.randrange(maxi[i])
        d = [float(x) for x in element[0:-2]]
    try:
        y.append(int(element[-1]))
    except:
        continue
    X.append(d)

print "TESTEEEEEEEEEEEEEEEEEEEE"
print X
print y


X = np.array(X)
y = np.array(y)

kn=10

kf = KFold(kn, shuffle=True)
metrics = [0,""]

Classifiers=["LR", "KNN", "RF", "NB","XGBoost"]

#this loop will interate for each classifier using diferents kfolds
for classifier in Classifiers:
    for train_index, test_index in kf.split(X):
        print "Classifier: ",classifier
        print "using kfold= ",kn
        print "\n"
        #this will chose the classifier, and use gridSearch to choose the best hyper parameters focussing on reach the best AUC score
        # Linear Regression
        if classifier == "LR":
          Cs = [ 10.0**c for c in xrange(-2, 3, 1) ]
          clf = GridSearchCV(estimator=LogisticRegression(), param_grid=dict(C=Cs), scoring="roc_auc", n_jobs=-1, cv=5, verbose=0)
        # K Neighbors
        elif classifier == "KNN":
          Ks = [ k for k in xrange(1, 15, 2) ]
          clf = GridSearchCV(estimator=neighbors.KNeighborsClassifier(), param_grid=dict(n_neighbors=Ks), scoring="roc_auc", n_jobs=-1, cv=5, verbose=0)
        #Nayve Bayes
        elif classifier == "NB":
          clf = GridSearchCV(estimator=GaussianNB(), param_grid=dict(),scoring="roc_auc", n_jobs=-1, cv=5, verbose=0)
        #random forest
        elif classifier == "RF":
          estimators = [ e for e in xrange(5, 25, 5) ]
          clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=dict(n_estimators=estimators), scoring="roc_auc", n_jobs=-1, cv=5, verbose=0)
        #XGBoost
        elif classifier == "XGBoost":
          clf = xgb.XGBClassifier()
        # elif classifier == "LSVM":
        #   Cs = [ 10.0**c for c in xrange(-2, 3, 1) ]
        #   clf = GridSearchCV(estimator=LinearSVC(), param_grid=dict(C=Cs), scoring="roc_auc", n_jobs=-1, cv=5, verbose=0)
        # elif classifier == "SVM":
        #   Cs = [ 2.0**c for c in xrange(-5, 15, 1) ]
        #   Gs = [ 2.0**g for g in xrange(3, -15, -2) ]
        #   kernels = [ 'rbf', 'poly', 'sigmoid' ]
        #   decision_function_shapes = [ 'ovo', 'ovr' ]
        #   clf = GridSearchCV(estimator=SVC(probability=True), param_grid=dict(kernel=kernels, C=Cs, gamma=Gs, decision_function_shape=decision_function_shapes), scoring="roc_auc", n_jobs=-1, cv=5, verbose=0)

        #this fit will train your classifier to your dataset
        clf.fit(X[train_index],y[train_index])
        #this will return the probability of each example be 0 or 1
        y_pred = clf.predict_proba(X[test_index])
        #this will generate the AUC score
        auc = roc_auc_score(y[test_index], y_pred[:,1])
        print "AUC: ", str(auc)
        print "###########################\n"

        #this will save the greater value, the estimator (model), the train and test set, to reproduce the best model with the real train set file
        if (classifier!="XGBoost"):
            if metrics[0]<auc:
                metrics=[auc, clf.best_estimator_, X[train_index], y[train_index]]
        else:
            if metrics[0]<auc:
                metrics=[auc, "xgboost",X[train_index], y[train_index]]


#generete logs, with statistics
print "best estimator: ", metrics[1]
print "greater AUC: ", metrics[0]
file_name="results/statistics-"+strategie+".txt"
with open(file_name, "a+") as f:
    f.write("###################################\n")
    f.write("strategie "+ strategie)
    f.write("\nbest estimator: "+ str(metrics[1]))
    f.write("\ngreater AUC: "+ str(metrics[0]))
    f.write("\nconjunto de teste: "+ str(metrics[2]))
    f.write("\nconjunto de treino: "+ str(metrics[3]))
    f.write("\n###################################\n")
#he must use the classifier with the best score
if metrics[1]=="xgboost":
    clf_greater=xgb.XGBClassifier()
else:
    clf_greater=metrics[1]
#clf_greater = metrics[1]
clf_greater.fit(metrics[2], metrics[3])



#open the test_set
file_name= "trabalho_final/test_set.csv"
file_test = open(file_name)

print "generating test labels!!\n"
mylines_test = file_test.read().split('\n')
#tratando arquivo de teste
X_test = []
signal = 1
for line in mylines_test:
    if signal == 1:
        signal = 0
        continue
    element = line.split(";")
    try:
        d = [float(x) for x in element[0:-1]]
    except:
        for i in range(0, len(element[0:-1])):
            if element[i] == 'NA':
                if strategie=="Only_0":
                    element[i] = 0
                elif strategie=="All_median":
                    element[i]=median[i]
                else:
                    element[i]=random.randrange(maxi[i])
        d = [float(x) for x in element[0:-1]]
    X_test.append(d)
X_test = X_test[0:-2]
X_test = np.array(X_test)


#generate the prob of each example
probs = clf_greater.predict_proba(X_test)[:,1]
print "test labels probs:"
print probs
file_name= "results/teste"+strategie+".txt"
with open(file_name, "a+") as f:
    for prob in probs:
        f.write(str(prob)+"\n")
