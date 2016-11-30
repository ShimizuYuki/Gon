# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet,\
     optimizers,Chain,serializers
import chainer.functions  as F
import sys

import CNNcifar10Train


if __name__=="__main__":

    unitSet   = [512]     #中間層ユニット数

    N = 1000           #テストするサンプル数
    N_train = 500     #訓練データの数
    N_test = N - N_train #テストデータの数

    tate = 4
    yoko = 4
    n_config = tate*yoko

    category= ["Airplane","Automobile",
               "Bird","Cat","Deer","Dog",
               "Frog","Horse","Ship","Truck"]

    KeepTimeList = ["2016-10-03-21-54",
                    "2016-10-04-00-32",
                    "2016-10-04-03-21",
                    "2016-10-04-06-34",
                    "2016-10-04-09-50",
                    "2016-10-04-13-12",
                    "2016-10-05-13-07"]

    for val in range(6,7):
        if val == 2:
            NetName = "conv2"
            Title=NetName
            KeepTime = ""
        elif val == 7:
            NetName = "Cifar_Liner"
            KeepTime = KeepTimeList[val-1]
        else:
            NetName = "Cifar_conv%d"%(val)
            KeepTime = KeepTimeList[val-1]

        Cifar10 =  CNNcifar10Train.Cifar10CNN(N,N_train,N_test,unitSet,NetName)
        serializers.load_npz\
        ('./modelkeep/'+NetName+'_Model_'+KeepTime,Cifar10.model)
        #Cifar10.draw_1image(0)
        Cifar10.draw_answerChack(category,tate,yoko)
        #Cifar10.predict(n_config)
        plt.show()
