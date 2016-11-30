
# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet,\
     optimizers,Chain,serializers
import chainer.functions  as F
import sys

import Color_CNN
import Dataload


class Cifar10CNN(Color_CNN.ColorCNN):
    def __init__(self,N,Ntr,Nte,unitSet,NetName):
        super(Cifar10CNN,self).__init__(N,Ntr,Nte,unitSet,NetName)


    #(入力のチャネル数,出力のチャネル数,フィルタサイズ)
    def set_Model_Opti(self):
        if self.NetName == 'Cifar_conv1':
            self.model = FunctionSet(
                 conv1=F.Convolution2D(3, 32, 3, pad=1),
                 l1=F.Linear(8192, 512),
                 l2=F.Linear(512, 10))
        elif self.NetName == 'Cifar_conv2':
            self.model = FunctionSet(
                conv1=F.Convolution2D(3, 32, 3, pad=1),
                conv2=F.Convolution2D(32, 32, 3, pad=1),
                l1=F.Linear(8192, 512),
                l2=F.Linear(512, 10))
        elif self.NetName == 'Cifar_conv3':
            self.model = FunctionSet(
                conv1=F.Convolution2D(3, 32, 3, pad=1),
                conv2=F.Convolution2D(32, 32, 3, pad=1),
                conv3=F.Convolution2D(32, 32, 3, pad=1),
                l1=F.Linear(2048, 512),
                l2=F.Linear(512, 10))
        elif self.NetName == 'Cifar_conv4':
            self.model = FunctionSet(
                conv1=F.Convolution2D(3, 32, 3, pad=1),
                conv2=F.Convolution2D(32, 32, 3, pad=1),
                conv3=F.Convolution2D(32, 32, 3, pad=1),
                conv4=F.Convolution2D(32, 32, 3, pad=1),
                l1=F.Linear(2048, 512),
                l2=F.Linear(512, 10))
        elif self.NetName == 'Cifar_conv5':
            self.model = FunctionSet(
                conv1=F.Convolution2D(3, 32, 3, pad=1),
                conv2=F.Convolution2D(32, 32, 3, pad=1),
                conv3=F.Convolution2D(32, 32, 3, pad=1),
                conv4=F.Convolution2D(32, 32, 3, pad=1),
                conv5=F.Convolution2D(32, 32, 3, pad=1),
                l1=F.Linear(512, 512),
                l2=F.Linear(512, 10))
        elif self.NetName == 'Cifar_conv6':
            self.model = FunctionSet(
                conv1=F.Convolution2D(3, 32, 3, pad=1),
                conv2=F.Convolution2D(32, 32, 3, pad=1),
                conv3=F.Convolution2D(32, 32, 3, pad=1),
                conv4=F.Convolution2D(32, 32, 3, pad=1),
                conv5=F.Convolution2D(32, 32, 3, pad=1),
                conv6=F.Convolution2D(32, 32, 3, pad=1),
                l1=F.Linear(512, 512),
                l2=F.Linear(512, 10))
        elif self.NetName == 'Cifar_conv7':
            self.model = FunctionSet(
                conv1=F.Convolution2D(3, 32, 3, pad=1),
                conv2=F.Convolution2D(32, 32, 3, pad=1),
                conv3=F.Convolution2D(32, 32, 3, pad=1),
                conv4=F.Convolution2D(32, 32, 3, pad=1),
                conv5=F.Convolution2D(32, 32, 3, pad=1),
                conv6=F.Convolution2D(32, 32, 3, pad=1),
                conv7=F.Convolution2D(32, 32, 3, pad=1),
                l1=F.Linear(128, 128),
                l2=F.Linear(128, 10))
        elif self.NetName == 'Cifar_conv8':
             self.model = FunctionSet(
                conv1=F.Convolution2D(3, 32, 3, pad=1),
                conv2=F.Convolution2D(32, 32, 3, pad=1),
                conv3=F.Convolution2D(32, 32, 3, pad=1),
                conv4=F.Convolution2D(32, 32, 3, pad=1),
                conv5=F.Convolution2D(32, 32, 3, pad=1),
                conv6=F.Convolution2D(32, 32, 3, pad=1),
                conv7=F.Convolution2D(32, 32, 3, pad=1),
                conv8=F.Convolution2D(32, 32, 3, pad=1),
                l1=F.Linear(128, 128),
                l2=F.Linear(128, 10))
        elif self.NetName == 'Cifar_Liner':
            self.model = FunctionSet(\
                l1=F.Linear(32*32*3, self.unitSet[0]),
                l2=F.Linear(self.unitSet[0], self.unitSet[0]),
                l3=F.Linear(self.unitSet[0], 10)
                )

        #Optimaizerの設定
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    def forward(self,x_data, y_data, train=True):
        if self.NetName == "Cifar_conv1":
            x, t = Variable(x_data), Variable(y_data)
            h = F.max_pooling_2d(F.relu(self.model.conv1(x)), 2)
            h = F.dropout(F.relu(self.model.l1(h)), train=train)
            y = self.model.l2(h)
        elif self.NetName == "Cifar_conv2":
            x, t = Variable(x_data), Variable(y_data)
            h = F.relu(self.model.conv1(x))
            h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
            h = F.dropout(F.relu(self.model.l1(h)), train=train)
            y = self.model.l2(h)
        elif self.NetName == "Cifar_conv3":
            x, t = Variable(x_data), Variable(y_data)
            h = F.relu(self.model.conv1(x))
            h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
            h = F.max_pooling_2d(F.relu(self.model.conv3(h)), 2)
            h = F.dropout(F.relu(self.model.l1(h)), train=train)
            y = self.model.l2(h)
        elif self.NetName == "Cifar_conv4":
            x, t = Variable(x_data), Variable(y_data)
            h = F.relu(self.model.conv1(x))
            h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
            h = F.relu(self.model.conv3(h))
            h = F.max_pooling_2d(F.relu(self.model.conv4(h)), 2)
            h = F.dropout(F.relu(self.model.l1(h)), train=train)
            y = self.model.l2(h)
        elif self.NetName == "Cifar_conv5":
            x, t = Variable(x_data), Variable(y_data)
            h = F.relu(self.model.conv1(x))
            h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
            h = F.relu(self.model.conv3(h))
            h = F.max_pooling_2d(F.relu(self.model.conv4(h)), 2)
            h = F.max_pooling_2d(F.relu(self.model.conv5(h)), 2)
            h = F.dropout(F.relu(self.model.l1(h)), train=train)
            y = self.model.l2(h)
        elif self.NetName == "Cifar_conv6":
            x, t = Variable(x_data), Variable(y_data)
            h = F.relu(self.model.conv1(x))
            h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
            h = F.relu(self.model.conv3(h))
            h = F.max_pooling_2d(F.relu(self.model.conv4(h)), 2)
            h = F.relu(self.model.conv5(h))
            h = F.max_pooling_2d(F.relu(self.model.conv6(h)), 2)
            h = F.dropout(F.relu(self.model.l1(h)), train=train)
            y = self.model.l2(h)
        elif self.NetName == "Cifar_conv7":
            x, t = Variable(x_data), Variable(y_data)
            h = F.relu(self.model.conv1(x))
            h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
            h = F.relu(self.model.conv3(h))
            h = F.max_pooling_2d(F.relu(self.model.conv4(h)), 2)
            h = F.relu(self.model.conv5(h))
            h = F.max_pooling_2d(F.relu(self.model.conv6(h)), 2)
            h = F.max_pooling_2d(F.relu(self.model.conv7(h)), 2)
            h = F.dropout(F.relu(self.model.l1(h)), train=train)
            y = self.model.l2(h)
        elif self.NetName == "Cifar_conv8":
            x, t = Variable(x_data), Variable(y_data)
            h = F.relu(self.model.conv1(x))
            h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
            h = F.relu(self.model.conv3(h))
            h = F.max_pooling_2d(F.relu(self.model.conv4(h)), 2)
            h = F.relu(self.model.conv5(h))
            h = F.max_pooling_2d(F.relu(self.model.conv6(h)), 2)
            h = F.relu(self.model.conv7(h))
            h = F.max_pooling_2d(F.relu(self.model.conv8(h)), 2)
            h = F.dropout(F.relu(self.model.l1(h)), train=train)
            y = self.model.l2(h)
        elif self.NetName == 'Cifar_Liner':
            x, t = Variable(x_data), Variable(y_data)
            h = F.dropout(F.relu(self.model.l1(x)),train=train)
            h = F.dropout(F.relu(self.model.l2(h)),train=train)
            h = self.model.l3(h)
            y  = h
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t),y

if __name__=="__main__":
    #DataName_Netの特徴で記入
    NetName = "Cifar_conv1"
    #AnswerCheckでだけ使う
    category= ["Airplane","Automobile",
               "Bird","Cat","Deer","Dog",
               "Frog","Horse","Ship","Truck"]

    #訓練だけなら 0 テストもするなら 1
    howloop = 1

    # 学習の繰り返し回数
    n_epoch   = 20
    n_display = n_epoch

    # 確率的勾配降下法で学習させる際の１回分のバッチサイズ
    batchsize = 100

    N = 60000        #サンプル数
    N_train = 50000    #訓練データの数
    N_test = N - N_train #テストデータの数

    unitSet   = [512]     #中間層ユニット数

    tate = 5
    yoko = 5
    n_config = tate*yoko
    n_predict = 10

    for i in range(9,10):
        if i == 9:
            NetName = "Cifar_Liner"
        else :
            NetName = "Cifar_conv%d"%(i)
        Cifar10 =  Cifar10CNN(N,N_train,N_test,unitSet,NetName)
        Cifar10.train_loop(howloop,n_epoch,batchsize,n_display)

        #Cifar10.draw_1image(0)

        #Cifar10.draw_answerChack(caategory,tate,yoko)
        Cifar10.predict(n_predict)
        #plt.show()
