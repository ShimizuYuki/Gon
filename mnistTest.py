# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet,\
     optimizers,Chain,serializers
import chainer.functions  as F
import sys

import Dataload
import mnistTrain

#np.random.seed(0)

if __name__=="__main__":

    unitSet   = [900]     #中間層ユニット数

    N = 2500           #テストする画像数
    N_train = 2000
    N_test = N - N_train
    tate = 5
    yoko =5
    n_config = tate*yoko

    #(N , N_train , N_test , unitSet)
    Mnist = mnistTrain.MnistNN(N,N_train,N_test,unitSet)

    serializers.load_npz('./modelkeep/Mnist_900unit_Model',Mnist.model)

    #Mnist.draw_digit(0)
    Mnist.draw_answerChack(tate,yoko)
    Mnist.predict(n_config)
    plt.show()
