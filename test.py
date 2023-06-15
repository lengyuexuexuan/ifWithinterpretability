from data import PageBlocks,test_Annthyroid,test_Cardiotocography,test_pima,vowels
from utils import test_auc

if __name__ == '__main__':

    ifi_sum_max = 0
    ifi_sum = 0
    if_sum = 0
    ecod_sum = 0
    count = 10
    for i in range(count):
        #input, output = PageBlocks.get_data()
        #input, output = test_pima.get_data()
        input, output = test_Annthyroid.get_data()
        #input, output = vowels.get_data()
        #input, output = test_Cardiotocography.get_data()
        ifi_res, ifi_res_max, if_res, ecod_res = test_auc.test_auc(input, output)
        ifi_sum += ifi_res
        ifi_sum_max += ifi_res_max
        if_sum += if_res
        ecod_sum += ecod_res
    print("----均值myif----")
    print(ifi_sum / count)
    print("----最大值myif----")
    print(ifi_sum_max / count)

    print("------if-----")
    print(if_sum / count)
    print("-----ecod----")
    print(ecod_sum / count)