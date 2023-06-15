"""
  测试隔离森林和ECOD的组合算法在其他数据集上和ECOD算法和隔离森林算法的ROC、AP的区别

"""
import random

import numpy as np
from scipy.io import arff
import scipy.io


rng = np.random.RandomState(42)

def get_data():

    datas = scipy.io.loadmat(r'D:\开题论文\数据\other\vowels.mat')
    input = datas['X']
    out_put = datas['y']
    all_data = np.hstack((input,out_put))
    np.random.shuffle(all_data)
    input = all_data[:, :12]
    output = all_data[:, 12]

    input = np.array(input)
    output = np.array(output).reshape(-1, 1)
    return input,output