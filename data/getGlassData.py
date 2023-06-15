import numpy as np
import pandas as pd


def getGlass():
    data = pd.read_csv(r'D:\开题论文\小论文\数据集\glass.data', header=None,sep=',')
    data = data.to_numpy()
    data = np.delete(data,0,axis=1)

    output = data[:,len(data[1])-1]
    data = np.delete(data,len(data[0])-1,axis=1)
    input = []
    new_output = []
    for i in range(len(data)):
        if (output[i] == 5) | (output[i] == 6):
            continue
        else:
            input.append(data[i].tolist())
            if output[i] == 7:
                new_output.append(1)
            else:
                new_output.append(0)

    return np.array(input), np.array(new_output)
