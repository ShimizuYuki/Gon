# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet,\
     optimizers,Chain,serializers
import chainer.functions  as F
import sys

#import Dataload
import ChainerNN

class MnistCNN(ChainerNN.NN):
    def __init__(self,N,Ntr,Nte,unitSet,NetName):
        super(MnistCNN,self).__init__(N,Ntr,Nte,unitSet,NetName)

        self.x_train \
        = self.x_train.reshape((self.N_train, 1, 28, 28))

        self.x_test \
        =  self.x_test.reshape((self.N_test, 1, 28, 28))

        print self.x_train.shape

    #(入力のチャネル数,出力のチャネル数,フィルタサイズ)
    def set_Model_Opti(self):
        if self.NetName == 'Mnist_conv1':
            self.model = FunctionSet(
                 conv1=F.Convolution2D(1, 32, 3, pad=1),
                 l1=F.Linear(6272, 512),
                 l2=F.Linear(512, 10))
        elif self.NetName == 'Mnist_conv2':
            self.model = FunctionSet(
                conv1=F.Convolution2D(1, 32, 3, pad=1),
                conv2=F.Convolution2D(32, 32, 3, pad=1),
                l1=F.Linear(6272, 512),
                l2=F.Linear(512, 10))
        elif self.NetName == 'Mnist_conv3':
            self.model = FunctionSet(
                conv1=F.Convolution2D(1, 32, 3, pad=1),
                conv2=F.Convolution2D(32, 32, 3, pad=1),
                conv3=F.Convolution2D(32, 32, 3, pad=1),
                l1=F.Linear(1568, 512),
                l2=F.Linear(512, 10))
        elif self.NetName == 'Mnist_conv4':
            self.model = FunctionSet(
                conv1=F.Convolution2D(1, 32, 3, pad=1),
                conv2=F.Convolution2D(32, 32, 3, pad=1),
                conv3=F.Convolution2D(32, 32, 3, pad=1),
                conv4=F.Convolution2D(32, 32, 3, pad=1),
                l1=F.Linear(1568, 512),
                l2=F.Linear(512, 10))
        elif self.NetName == 'Mnist_conv5':
            self.model = FunctionSet(
                conv1=F.Convolution2D(1, 32, 3, pad=1),
                conv2=F.Convolution2D(32, 32, 3, pad=1),
                conv3=F.Convolution2D(32, 32, 3, pad=1),
                conv4=F.Convolution2D(32, 32, 3, pad=1),
                conv5=F.Convolution2D(32, 32, 3, pad=1),
                l1=F.Linear(512, 512),
                l2=F.Linear(512, 10))
        elif self.NetName == 'Mnist_conv6':
            self.model = FunctionSet(
                conv1=F.Convolution2D(1, 32, 3, pad=1),
                conv2=F.Convolution2D(32, 32, 3, pad=1),
                conv3=F.Convolution2D(32, 32, 3, pad=1),
                conv4=F.Convolution2D(32, 32, 3, pad=1),
                conv5=F.Convolution2D(32, 32, 3, pad=1),
                conv6=F.Convolution2D(32, 32, 3, pad=1),
                l1=F.Linear(512, 512),
                l2=F.Linear(512, 10))
        elif self.NetName == 'Mnist_conv7':
            self.model = FunctionSet(
                conv1=F.Convolution2D(1, 32, 3, pad=1),
                conv2=F.Convolution2D(32, 32, 3, pad=1),
                conv3=F.Convolution2D(32, 32, 3, pad=1),
                conv4=F.Convolution2D(32, 32, 3, pad=1),
                conv5=F.Convolution2D(32, 32, 3, pad=1),
                conv6=F.Convolution2D(32, 32, 3, pad=1),
                conv7=F.Convolution2D(32, 32, 3, pad=1),
                l1=F.Linear(128, 128),
                l2=F.Linear(128, 10))
        elif self.NetName == 'Mnist_conv8':
            self.model = FunctionSet(
                conv1=F.Convolution2D(1, 32, 3, pad=1),
                conv2=F.Convolution2D(32, 32, 3, pad=1),
                conv3=F.Convolution2D(32, 32, 3, pad=1),
                conv4=F.Convolution2D(32, 32, 3, pad=1),
                conv5=F.Convolution2D(32, 32, 3, pad=1),
                conv6=F.Convolution2D(32, 32, 3, pad=1),
                conv7=F.Convolution2D(32, 32, 3, pad=1),
                conv8=F.Convolution2D(32, 32, 3, pad=1),
                l1=F.Linear(128, 128),
                l2=F.Linear(128, 10))
        elif self.NetName == 'Mnist_Liner':
            self.model = FunctionSet(\
                l1=F.Linear(28*28, self.unitSet[0]),
                l2=F.Linear(self.unitSet[0], self.unitSet[0]),
                l3=F.Linear(self.unitSet[0], 10)
                )

        #Optimaizerの設定
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    def forward(self,x_data, y_data, train=True):
        if self.NetName == "Mnist_conv1":
            x, t = Variable(x_data), Variable(y_data)
            h = F.max_pooling_2d(F.relu(self.model.conv1(x)), 2)
            h = F.dropout(F.relu(self.model.l1(h)), train=train)
            y = self.model.l2(h)
        elif self.NetName == "Mnist_conv2":
            x, t = Variable(x_data), Variable(y_data)
            h = F.relu(self.model.conv1(x))
            h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
            h = F.dropout(F.relu(self.model.l1(h)), train=train)
            y = self.model.l2(h)
        elif self.NetName == "Mnist_conv3":
            x, t = Variable(x_data), Variable(y_data)
            h = F.relu(self.model.conv1(x))
            h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
            h = F.max_pooling_2d(F.relu(self.model.conv3(h)), 2)
            h = F.dropout(F.relu(self.model.l1(h)), train=train)
            y = self.model.l2(h)
        elif self.NetName == "Mnist_conv4":
            x, t = Variable(x_data), Variable(y_data)
            h = F.relu(self.model.conv1(x))
            h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
            h = F.relu(self.model.conv3(h))
            h = F.max_pooling_2d(F.relu(self.model.conv4(h)), 2)
            h = F.dropout(F.relu(self.model.l1(h)), train=train)
            y = self.model.l2(h)
        elif self.NetName == "Mnist_conv5":
            x, t = Variable(x_data), Variable(y_data)
            h = F.relu(self.model.conv1(x))
            h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
            h = F.relu(self.model.conv3(h))
            h = F.max_pooling_2d(F.relu(self.model.conv4(h)), 2)
            h = F.max_pooling_2d(F.relu(self.model.conv5(h)), 2)
            h = F.dropout(F.relu(self.model.l1(h)), train=train)
            y = self.model.l2(h)
        elif self.NetName == "Mnist_conv6":
            x, t = Variable(x_data), Variable(y_data)
            h = F.relu(self.model.conv1(x))
            h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
            h = F.relu(self.model.conv3(h))
            h = F.max_pooling_2d(F.relu(self.model.conv4(h)), 2)
            h = F.relu(self.model.conv5(h))
            h = F.max_pooling_2d(F.relu(self.model.conv6(h)), 2)
            h = F.dropout(F.relu(self.model.l1(h)), train=train)
            y = self.model.l2(h)
        elif self.NetName == "Mnist_conv7":
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
        elif self.NetName == "Mnist_conv8":
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
        elif self.NetName == 'Mnist_Liner':
            x, t = Variable(x_data), Variable(y_data)
            h = F.dropout(F.relu(self.model.l1(x)),train=train)
            h = F.dropout(F.relu(self.model.l2(h)),train=train)
            h = self.model.l3(h)
            y  = h

        return F.softmax_cross_entropy(y, t), F.accuracy(y, t),y

if __name__=="__main__":
    NetName = "Mnist_conv1"

    #訓練だけなら 0 テストもするなら 1
    howloop = 1

    # 学習の繰り返し回数
    n_epoch   = 10
    n_display = n_epoch

    # 確率的勾配降下法で学習させる際の１回分のバッチサイズ
    batchsize = 100

    N = 70000           #サンプル数
    N_train =60000    #訓練データの数
    N_test = N - N_train #テストデータの数

    unitSet   = [512]     #中間層ユニット数

    tate = 5
    yoko = 5
    n_config = tate*yoko
    n_predict =10

    for i in range(1,2):
        if i == 9:
            NetName = "Mnist_Liner"
        else :
            NetName = "Mnist_conv%d"%(i)
        Mnist =  MnistCNN(N,N_train,N_test,unitSet,NetName)
        #Mnist.train_loop(howloop,n_epoch,batchsize,n_display)
        #Mnist.draw_1image(0)
        #Mnist.draw_answerChack(tate,yoko)
        #Mnist.predict(n_config)
        #plt.show()
