import random
from scipy.io import arff
import numpy as np
rng = np.random.RandomState(42)

def get_data():
    data, a = arff.loadarff(r"D:\开题论文\数据\semantic\Annthyroid\Annthyroid_withoutdupl_norm_07.arff")
    input = []
    output = []
    np.random.shuffle(data)
    for sub_data in data:
        tmp = []
        for i in range(21):
            tmp.append(sub_data[i])
        input.append(tmp)
        if sub_data[22] == b'yes':
            output.append(1)
        else:
            output.append(0)
    return np.array(input), np.array(output)
