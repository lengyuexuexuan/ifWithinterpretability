from pyod.models.ecod import ECOD
from sklearn.preprocessing import minmax_scale


def ecod_test(x_train):
    ecod = ECOD()
    ecod.fit(x_train)
    scores = ecod.decision_function(x_train)
    scores = minmax_scale(scores)
    return scores