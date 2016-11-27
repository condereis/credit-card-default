from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import lognorm
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
import random
import math


dataset = pd.read_excel('raw.xls', decimal='.', index_col=0)
print(dataset.shape[0])
print(dataset.shape[1])
splitRatio = 1-0.8

rep=1
mat=[[]for x in range(rep)]
ac=[0 for y in range(rep)]
resultado = [[]for x in range(rep)]
Y = dataset['default.payment.next.month']
X = dataset.drop('default.payment.next.month',1)
for q in range(rep):
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=splitRatio,random_state=0)
    print(x_train.shape[0])
    print(x_test.shape[0])
    model = GaussianNB()
    for dim in range(1,20):
        reduced_data = PCA(n_components=dim).fit_transform(x_train)
        score = cross_val_score(model, reduced_data, y_train, cv=5,n_jobs=-1)
        print ('N dimensoes = %i ==> %f +- %f' % (dim,np.mean(score), np.std(score)))
        model.fit(x_train,y_train)
        predicted = model.predict(x_test)
        print(confusion_matrix(y_test, predicted))
    
