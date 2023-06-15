import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale

from Algorithm import ifWIthinter
from data.generateData import generate

def draw_pic(data1,data2,data3,data4):

    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 13
    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=17)  # fontsize of the figure title

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax = ax.flatten()
    print(len(data1))
    labels = ['D1','D2','D3','D4']
    ax[0].bar(labels,data1)
    ax[1].bar(labels, data2)
    ax[2].bar(labels, data3)
    ax[3].bar(labels, data4)
    ax[0].set_title("Outlier1")
    ax[1].set_title("Outlier2")
    ax[2].set_title("Outlier3")
    ax[3].set_title("Outlier4")
    plt.show()



if __name__ == '__main__':
    x = generate()
    feature_sort,myif_score = ifWIthinter.ifWithInter1(x)

    sort = []
    for data in feature_sort:
        sort.append(np.argsort((data)))

    fea_st1 = minmax_scale(feature_sort[0])
    fea_st2 = minmax_scale(feature_sort[1])
    fea_st3 = minmax_scale(feature_sort[1001])
    fea_st4 = minmax_scale(feature_sort[1002])

    draw_pic(fea_st1,fea_st2, fea_st3, fea_st4)
    print(sort[0])
    print(minmax_scale(feature_sort[0]))
    print(minmax_scale(feature_sort[1]))
    print(minmax_scale(feature_sort[1001]))
    print(minmax_scale(feature_sort[1002]))