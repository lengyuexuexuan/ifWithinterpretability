import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, minmax_scale

rng = np.random.RandomState(2023)


def ifWithInter(x_train):
    dim = len(x_train[0])
    y_test_all = []
    y_test_all = np.array(y_test_all)
    for i in range(dim):
        tmp_train = x_train[:, i].reshape(-1, 1)
        clf = IsolationForest(random_state=rng)
        clf.fit(tmp_train)
        y_test = clf.decision_function(tmp_train).reshape(-1, 1)
        if y_test_all.size == 0:
            y_test_all = y_test
        else:
            y_test_all = np.hstack((y_test_all, y_test))

    y_test_all = y_test_all.sum(axis=1) * -1
    y_test_all = minmax_scale(y_test_all)
    return y_test_all


def ifWithInter_max(x_train):
    dim = len(x_train[0])
    y_test_all = []
    y_test_all = np.array(y_test_all)
    for i in range(dim):
        tmp_train = x_train[:, i].reshape(-1, 1)
        clf = IsolationForest(random_state=rng)
        clf.fit(tmp_train)
        y_test = clf.decision_function(tmp_train).reshape(-1, 1)
        if y_test_all.size == 0:
            y_test_all = y_test
        else:
            y_test_all = np.hstack((y_test_all, y_test))
    y_test_all = y_test_all * -1
    index_max = np.argmax(y_test_all, axis=1)
    y_test_all = y_test_all[range(y_test_all.shape[0]), index_max]
    y_test_all = minmax_scale(y_test_all)
    return y_test_all

def ifWithInter1(x_train):
    dim = len(x_train[0])
    y_test_all = []
    y_test_all = np.array(y_test_all)
    for i in range(dim):
        tmp_train = x_train[:, i].reshape(-1, 1)
        clf = IsolationForest(random_state=rng)
        clf.fit(tmp_train)
        y_test = clf.decision_function(tmp_train).reshape(-1, 1)
        if y_test_all.size == 0:
            y_test_all = y_test
        else:
            y_test_all = np.hstack((y_test_all, y_test))
    y_test_all = y_test_all * -1
    y_test_res = y_test_all.sum(axis=1)
    y_test_res = minmax_scale(y_test_res)

    return y_test_all, y_test_res