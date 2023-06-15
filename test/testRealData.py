import operator

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import minmax_scale
from data.getGlassData import getGlass
from Algorithm import ifWIthinter

import matplotlib.pyplot as plt


def draw_pic(res):
    x_label = ['1st','2st','3st','4st','5st','6st','7st','8st','9st']
    plt.figure(figsize=(8, 5), dpi=450)
    buttom1 = np.add(res[0],res[1])
    buttom2 = np.add(buttom1,res[2])
    buttom3 = np.add(buttom2,res[3])
    buttom4 = np.add(buttom3,res[4])
    buttom5 = np.add(buttom4,res[5])
    buttom6 = np.add(buttom5,res[6])
    buttom7 = np.add(buttom6,res[7])



    plt.bar(x_label, res[0], color='red', label='RI')
    plt.bar(x_label, res[1],bottom=res[0],color='blue', label='NA')
    plt.bar(x_label, res[2], bottom=buttom1, color='green', label='MG')
    plt.bar(x_label, res[3], bottom=buttom2, color='yellow', label='AL')
    plt.bar(x_label, res[4], bottom=buttom3, color='pink', label='SI')
    plt.bar(x_label, res[5], bottom=buttom4, color='Gray', label='K')
    plt.bar(x_label, res[6], bottom=buttom5, color='orange', label='CA')
    plt.bar(x_label, res[7], bottom=buttom6, color='brown', label='BA')
    plt.bar(x_label, res[8], bottom=buttom7, color='purple', label='Fe')
    plt.legend(bbox_to_anchor=(1, 0.5), loc=3, borderaxespad=0)

    plt.xlabel('Rank')
    plt.ylabel('Nums')
    plt.title("Feature ranking of 27 outliers")
    plt.show()


def get_precision(myif_score,y):
    auc = roc_auc_score(y, myif_score)
    print(auc)


def draw_oneDim(x):
    BIGGER_SIZE = 13
    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=17)  # fontsize of the figure title

    mg_data = x[:,2]
    al_data = x[:,3]
    ba_data = x[:,7]
    ca_data = x[:,6]
    ri_data = x[:,0]


    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()
    ax[0].scatter(ca_data[:164],mg_data[:164],c="b",label="Normal data")
    ax[0].scatter(ca_data[164:], mg_data[164:], c="r", label="Outliers")
    ax[0].set_xlabel("CA")
    ax[0].set_ylabel("MG")

    ax[1].scatter(ca_data[:164], al_data[:164], c="b", label="Normal data")
    ax[1].scatter(ca_data[164:], al_data[164:], c="r", label="Outliers")
    ax[1].set_xlabel("CA")
    ax[1].set_ylabel("AL")

    ax[2].scatter(ca_data[:164], ba_data[:164], c="b", label="Normal data")
    ax[2].scatter(ca_data[164:], ba_data[164:], c="r", label="Outliers")
    ax[2].set_xlabel("CA")
    ax[2].set_ylabel("BA")

    ax[3].scatter(ca_data[:164], ri_data[:164], c="b", label="Normal data")
    ax[3].scatter(ca_data[164:], ri_data[164:], c="r", label="Outliers")
    ax[3].set_xlabel("CA")
    ax[3].set_ylabel("RI")

    plt.show()




if __name__ == '__main__':
    x,y = getGlass()
    # feature_sort,myif_score = ifWIthinter.ifWithInter1(x)
    # print(x.shape)
    # sort = []
    # for data in feature_sort[163:]:
    #     sort.append(np.argsort(-data).tolist())
    #
    # res = []
    # score = [0]*9
    # for i in range(9):
    #     dic = {}
    #     for k in range(9):
    #         dic[k] = 0
    #     for j in range(len(sort)):
    #         if sort[j][i] in dic.keys():
    #             dic[sort[j][i]] = dic[sort[j][i]] + 1
    #         else:
    #             dic[sort[j][i]] = 1
    #         score[sort[j][i]] += i
    #
    #     res.append(list(dict(sorted(dic.items(),key=operator.itemgetter(0))).values()))
    # res = np.array(res)
    # res = np.transpose(res)
    # draw_pic(res)
    # print(score)
    # get_precision(myif_score,y)
    draw_oneDim(x)