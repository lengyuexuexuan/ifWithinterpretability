"""
  测试隔离森林和ECOD的组合算法在其他数据集上和ECOD算法和隔离森林算法的ROC、AP的区别

"""
import pandas as pd
from scipy.io import arff
import numpy as np


def get_data():
    data, a = arff.loadarff(r"D:\开题论文\数据\semantic\Cardiotocography\Cardiotocography_withoutdupl_norm_22.arff")
    input = []
    output = []
    np.random.shuffle(data)
    for sub_data in data:
        tmp = []
        for i in range(21):
            tmp.append(sub_data[i])
        input.append(tmp)
        if sub_data[21] == b'yes':
            output.append(1)
        else:
            output.append(0)
    return np.array(input), np.array(output)

