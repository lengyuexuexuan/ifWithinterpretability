"""
  测试隔离森林和ECOD的组合算法在其他数据集上和ECOD算法和隔离森林算法的ROC、AP的区别

"""
import random
import numpy as np
from scipy.io import arff
from utils import test_auc


rng = np.random.RandomState(42)


def get_data():
    data, a = arff.loadarff(r"D:\开题论文\数据\semantic\Pima\Pima_withoutdupl_norm_35.arff")
    input = []
    output = []
    np.random.shuffle(data)
    for sub_data in data:
        tmp = []
        for i in range(1, 9):
            tmp.append(sub_data[i])
        input.append(tmp)
        if sub_data[9] == b'yes':
            output.append(1)
        else:
            output.append(0)
    return np.array(input), np.array(output)



if __name__ == '__main__':
    input, output = get_data()
    test_auc.test_auc(input,output)



