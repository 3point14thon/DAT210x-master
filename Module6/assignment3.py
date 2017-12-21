#
# This code is intentionally missing!
# Read the directions on the course lab page!
#
import numpy as np
import matplotlib
import matplotlib as plt

import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

X = pd.read_csv('Datasets/parkinsons.data')
y = X.status
X = X.drop(['name','status'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=7)

prepro = preprocessing.StandardScaler()
prepro.fit(X_train)
X_train = prepro.transform(X_train)
X_test = prepro.transform(X_test)

#test components 4 to 6 and neighbors 2 to 5 inclusive on both
iso = Isomap(n_components=6,n_neighbors=5)
iso.fit(X_train)
X_train = iso.transform(X_train)
X_test = iso.transform(X_test)

best_score = [0,0,0]
for C in np.arange(0.05,2,0.05):
  for gamma in np.arange(0.001,0.1,0.001):
    svc = svm.SVC(C=C, gamma=gamma)
    svc.fit(X_train,y_train)
    score = svc.score(X_test,y_test)
    if score > best_score[0]:
      best_score[0] = score
      best_score[1] = C
      best_score[2] = gamma
print 'The best score was: ', best_score[0]
print 'with a C of: ' , best_score[1]
print 'and a gamma of: ' , best_score[2]
