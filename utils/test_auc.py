from Algorithm import *
from sklearn.metrics import *

from Algorithm import ifWIthinter, iForest, ecod


def test_auc(input, output):


    print("myif 均值")
    myif_score = ifWIthinter.ifWithInter(input)
    myif_avg_res = roc_auc_score(output, myif_score)
    print(myif_avg_res)

    print("myif 最大值")
    myif_max_score = ifWIthinter.ifWithInter_max(input)
    myif_max_res = roc_auc_score(output, myif_max_score)
    print(myif_max_res)

    print("ecod")
    ecod_score = ecod.ecod_test(input)
    ecod_res = roc_auc_score(output,ecod_score)
    print(ecod_res)

    print("if")
    if_score = iForest.pure_if(input)
    if_res = roc_auc_score(output, if_score)
    print(if_res)

    return myif_avg_res, myif_max_res, if_res, ecod_res