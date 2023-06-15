import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, minmax_scale

rng = np.random.RandomState(2023)
def pure_if(x_train):
    dim = len(x_train[0])
    clf = IsolationForest(random_state=rng, max_features=dim)
    clf.fit(x_train)
    y_test_scores = clf.decision_function(x_train).reshape(-1,1)
    y_test_scores = minmax_scale(-y_test_scores)
    return y_test_scores