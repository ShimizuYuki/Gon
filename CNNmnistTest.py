
# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet,\
     optimizers,Chain,serializers
import chainer.functions  as F
import sys

import CNNmnistTrain


if __name__=="__main__":

    unitSet   = [512]     #中間層ユニット数

    N = 600          #サンプル数
    N_train = 500    #訓練データの数
    N_test = N - N_train #テストデータの数

    tate = 4
    yoko = 4
    n_config = tate*yoko

    KeepTimeList  = ["2016-10-05-03-34",
                    "2016-10-05-04-44",
                    "2016-10-05-06-09",
                    "2016-10-05-07-46",
                    "2016-10-05-13-47",
                    "2016-10-05-15-10",
                    "2016-10-05-12-47"]
    for val in range(6,7):
        if val == 7:
            NetName = "Mnist_Liner"
            KeepTime = KeepTimeList[val-1]
            #Title = NetName.split('_')[1]
        else:
            NetName = "Mnist_conv%d"%(val)
            KeepTime = KeepTimeList[val-1]
        Mnist = CNNmnistTrain.MnistCNN(N,N_train,N_test,unitSet,NetName)

        serializers.load_npz\
        ('./modelkeep/'+NetName+'_Model_'+KeepTime,Mnist.model)

        Mnist.draw_1image(0)
        Mnist.draw_answerChack(tate,yoko)
        #Mnist.predict(n_config)
        plt.show()
