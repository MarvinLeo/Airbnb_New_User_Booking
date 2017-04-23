import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier


X = pd.read_csv('train_data.csv')
y = pd.read_csv('train_label.csv')

## Encoding y value
X = X.values
y = y.values

encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = y.ravel()
# #


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

# SGD_clf = SGDClassifier(penalty='elasticnet', loss='hinge')
# SGD_clf.fit(X_train, y_train)
# print "The score of SGD elastic training sets is ", SGD_clf.score(X_train, y_train)
# print 'The score of SGD elastic Test sets is', SGD_clf.score(X_test, y_test)
#
# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)
# print "The score of training sets is ", clf.score(X_train, y_train)
# print 'The score of Test sets is', clf.score(X_test, y_test)

clf_labels = np.array(['LogisticRegression',
                       'RandomForest',
                       'Ridge',
                       'svc_rbf',
                       'svc_lin'])#, 'svr_poly'])
# errvals = np.array([])

clfs = {
    'LogisticRegression': LogisticRegression(),
    'Ridge': RidgeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
    #'SVC_rbf': SVC(kernel='rbf')
    #'SVC_lin': SVC(kernel='linear')
}
## give up on SVM because the data set is too big, the svm is very slow

print 'start...........'
for name, clf in clfs.items():
    clf.fit(X_train, y_train)
    print name, 'finished'
    this_err = accuracy_score(y_test, clf.predict(X_test))
    #print "got error %0.2f" % this_err
    print name, 'score:', this_err