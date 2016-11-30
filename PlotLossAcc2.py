# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet,\
     optimizers,Chain,serializers
import chainer.functions  as F
import sys
import time

import ChainerNN

import datetime
import Dataload
import re

Name = "Mnist"

if Name == "Cifar" :
    KeepTimeList = ["2016-10-03-21-54",
                    "2016-10-04-00-32",
                    "2016-10-04-03-21",
                    "2016-10-04-06-34",
                    "2016-10-04-09-50",
                    "2016-10-04-13-12",
                    "2016-10-05-13-07"]

elif Name == "Mnist" :
    KeepTimeList = ["2016-10-05-03-34",
                    "2016-10-05-04-44",
                    "2016-10-05-06-09",
                    "2016-10-05-07-46",
                    "2016-10-05-13-47",
                    "2016-10-05-15-10",
                    "2016-10-05-12-47"]


n_epoch   = 20
batchsize = 100
N = 60000         #サンプル数
N_train = 50000    #訓練データの数
N_test = N - N_train #テストデータの数

plt.figure(figsize=(20,10))
plt.subplot(2,2,1)
plt.title(Name+"_Train_Loss")
plt.subplot(2, 2, 2)
plt.title(Name+"_Train_Accuracy")
plt.subplot(2, 2, 3)
plt.title(Name+"_Test_Loss")
plt.subplot(2, 2, 4)
plt.title(Name+"_Test_Accuracy")

for val in range(1,8):
    if Name == "Cifar":
        if val == 2:
            NetName = "conv2"
            Title=NetName
            KeepTime = ""
        elif val == 7:
            NetName = "Cifar_Liner"
            KeepTime = KeepTimeList[val-1]
            #Title = NetName.split('_')[1]
            Title = "Fully_Conected"
        else:
            NetName = "Cifar_conv%d"%(val)
            KeepTime = KeepTimeList[val-1]
            Title = NetName.split('_')[1]

    elif Name == "Mnist":
        if val == 7:
            NetName = "Mnist_Liner"
            KeepTime = KeepTimeList[val-1]
            #Title = NetName.split('_')[1]
            Title = "Fully_Conected"
        else :
            NetName = "Mnist_conv%d"%(val)
            KeepTime = KeepTimeList[val-1]
            Title = NetName.split('_')[1]


    TrainLossFile = open("LossAcc/"\
    +NetName+"_Train_Loss"+KeepTime+".txt",'r')
    TrainLoss = TrainLossFile.readline()
    TrainLoss = TrainLoss.split(',')
    TrainLoss = map(float,TrainLoss)
    TrainLoss = np.array(TrainLoss)
    TrainLossFile.close()

    TrainAccFile = open("LossAcc/"\
    +NetName+"_Train_Acc"+KeepTime+".txt",'r')
    TrainAcc = TrainAccFile.readline()
    TrainAcc = TrainAcc.split(',')
    TrainAcc = map(float,TrainAcc)
    TrainAcc = np.array(TrainAcc)
    TrainAccFile.close()


    TestLossFile = open("LossAcc/"\
    +NetName+"_Test_Loss"+KeepTime+".txt",'r')
    TestLoss = TestLossFile.readline()
    TestLoss = TestLoss.split(',')
    TestLoss = map(float,TestLoss)
    TestLoss = np.array(TestLoss)
    TestLossFile.close()


    TestAccFile = open("LossAcc/"\
    +NetName+"_Test_Acc"+KeepTime+".txt",'r')
    TestAcc = TestAccFile.readline()
    TestAcc = TestAcc.split(',')
    TestAcc = map(float,TestAcc)
    TestAcc = np.array(TestAcc)
    TestAccFile.close()

    PlotTrainLoss = np.zeros(n_epoch)
    PlotTrainAcc = np.zeros(n_epoch)
    PlotTestLoss = np.zeros(n_epoch)
    PlotTestAcc = np.zeros(n_epoch)

    x1 = len(TrainLoss)/n_epoch
    x2 = len(TestLoss)/n_epoch


    j=0
    for i in range(0,len(TrainLoss),x1):
        ave = TrainLoss[i:i+x1].mean()
        PlotTrainLoss[j] = ave
        ave = TrainAcc[i:i+x1].mean()
        PlotTrainAcc[j] = ave
        j+=1

    j=0
    for i in range(0,len(TestLoss),x2):
        ave = TestLoss[i:i+x2].mean()
        PlotTestLoss[j] = ave
        ave = TestAcc[i:i+x2].mean()
        PlotTestAcc[j] = ave
        j+=1
    plt.subplot(2, 2, 1)
    plt.plot(range(len(PlotTrainLoss)), PlotTrainLoss\
    ,label = Title)
    #plt.legend(loc='upper right')

    plt.subplot(2, 2, 2)
    plt.plot(range(len(PlotTrainAcc)), PlotTrainAcc\
    ,label = Title)
    #plt.legend(loc='lower right')

    plt.subplot(2, 2, 3)
    plt.plot(range(len(PlotTestLoss)), PlotTestLoss\
    ,label = Title)
    #plt.legend(loc='uper right')

    plt.subplot(2, 2, 4)
    plt.plot(range(len(PlotTestAcc)), PlotTestAcc\
    ,label = Title)
    #plt.legend(loc='lower right')

    print "%s's Finally Accuracy=%f"\
            %(NetName,PlotTrainAcc[-1])
    print


#plt.show()
