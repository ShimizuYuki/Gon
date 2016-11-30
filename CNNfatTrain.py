
# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet,\
     optimizers,Chain,serializers
import chainer.functions  as F
import sys

import ChainerNN
import Dataload

import cv2

class FatCNN(ChainerNN.NN):
    def __init__(self,N,Ntr,Nte,unitSet,NetName):
        self.height=80
        self.width=40
        super(FatCNN,self).__init__\
        (N,Ntr,Nte,unitSet,NetName)

        #x_train (N_train,1,h,w) float32
        #color_x_train (N_train,h,w,3) uint8

    def train_test_dataLoad(self):
        self.x_train,self.x_test,\
        self.color_x_train,self.color_x_test,\
        self.y_train,self.y_test\
        =Dataload.Load\
        (self.N,self.N_train,self.DataName,\
        self.height,self.width)

    #(入力のチャネル数,出力のチャネル数,フィルタサイズ)
    def set_Model_Opti(self):
        if self.NetName == 'Fat_conv1':
            self.model = FunctionSet(
                 conv1=F.Convolution2D(1,32, 3, pad=1),
                 l1=F.Linear(25600, 512),
                 l2=F.Linear(512, 2))
        elif self.NetName == 'Fat_conv2':
            self.model = FunctionSet(
                conv1=F.Convolution2D(1, 32, 3, pad=1),
                conv2=F.Convolution2D(32, 32, 3, pad=1),
                l1=F.Linear(6272, 512),
                l2=F.Linear(512, 2))
        elif self.NetName == 'Fat_conv3':
            self.model = FunctionSet(
                conv1=F.Convolution2D(1, 32, 3, pad=1),
                conv2=F.Convolution2D(32, 32, 3, pad=1),
                conv3=F.Convolution2D(32, 32, 3, pad=1),
                l1=F.Linear(1568, 512),
                l2=F.Linear(512, 2))
        elif self.NetName == 'Fat_conv4':
            self.model = FunctionSet(
                conv1=F.Convolution2D(1, 32, 3, pad=1),
                conv2=F.Convolution2D(32, 32, 3, pad=1),
                conv3=F.Convolution2D(32, 32, 3, pad=1),
                conv4=F.Convolution2D(32, 32, 3, pad=1),
                l1=F.Linear(1568, 512),
                l2=F.Linear(512, 2))
        elif self.NetName == 'Fat_conv5':
            self.model = FunctionSet(
                conv1=F.Convolution2D(1, 32, 3, pad=1),
                conv2=F.Convolution2D(32, 32, 3, pad=1),
                conv3=F.Convolution2D(32, 32, 3, pad=1),
                conv4=F.Convolution2D(32, 32, 3, pad=1),
                conv5=F.Convolution2D(32, 32, 3, pad=1),
                l1=F.Linear(1600, 512),
                l2=F.Linear(512, 2))
        elif self.NetName == 'Fat_conv6':
            self.model = FunctionSet(
                conv1=F.Convolution2D(1, 32, 3, pad=1),
                conv2=F.Convolution2D(32, 32, 3, pad=1),
                conv3=F.Convolution2D(32, 32, 3, pad=1),
                conv4=F.Convolution2D(32, 32, 3, pad=1),
                conv5=F.Convolution2D(32, 32, 3, pad=1),
                conv6=F.Convolution2D(32, 32, 3, pad=1),
                l1=F.Linear(512, 512),
                l2=F.Linear(512, 2))
        elif self.NetName == 'Fat_conv7':
            self.model = FunctionSet(
                conv1=F.Convolution2D(1, 32, 3, pad=1),
                conv2=F.Convolution2D(32, 32, 3, pad=1),
                conv3=F.Convolution2D(32, 32, 3, pad=1),
                conv4=F.Convolution2D(32, 32, 3, pad=1),
                conv5=F.Convolution2D(32, 32, 3, pad=1),
                conv6=F.Convolution2D(32, 32, 3, pad=1),
                conv7=F.Convolution2D(32, 32, 3, pad=1),
                l1=F.Linear(128, 128),
                l2=F.Linear(128, 2))
        elif self.NetName == 'Fat_conv8':
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
                l2=F.Linear(128, 2))
        elif self.NetName == 'Fat_Liner':
            self.model = FunctionSet(\
                l1=F.Linear(28*28, self.unitSet[0]),
                l2=F.Linear(self.unitSet[0], self.unitSet[0]),
                l3=F.Linear(self.unitSet[0], 2)
                )
        #Optimaizerの設定
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    def forward(self,x_data, y_data, train=True):
        if self.NetName == "Fat_conv1":
            x, t = Variable(x_data), Variable(y_data)
            h = F.max_pooling_2d(F.relu(self.model.conv1(x)), 2)
            h = F.dropout(F.relu(self.model.l1(h)), train=train)
            y = self.model.l2(h)
        elif self.NetName == "Fat_conv2":
            x, t = Variable(x_data), Variable(y_data)
            h = F.relu(self.model.conv1(x))
            h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
            h = F.dropout(F.relu(self.model.l1(h)), train=train)
            y = self.model.l2(h)
        elif self.NetName == "Fat_conv3":
            x, t = Variable(x_data), Variable(y_data)
            h = F.relu(self.model.conv1(x))
            h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
            h = F.max_pooling_2d(F.relu(self.model.conv3(h)), 2)
            h = F.dropout(F.relu(self.model.l1(h)), train=train)
            y = self.model.l2(h)
        elif self.NetName == "Fat_conv4":
            x, t = Variable(x_data), Variable(y_data)
            h = F.relu(self.model.conv1(x))
            h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
            h = F.relu(self.model.conv3(h))
            h = F.max_pooling_2d(F.relu(self.model.conv4(h)), 2)
            h = F.dropout(F.relu(self.model.l1(h)), train=train)
            y = self.model.l2(h)
        elif self.NetName == "Fat_conv5":
            x, t = Variable(x_data), Variable(y_data)
            h = F.relu(self.model.conv1(x))
            h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
            h = F.relu(self.model.conv3(h))
            h = F.max_pooling_2d(F.relu(self.model.conv4(h)), 2)
            h = F.max_pooling_2d(F.relu(self.model.conv5(h)), 2)
            h = F.dropout(F.relu(self.model.l1(h)), train=train)
            y = self.model.l2(h)
        elif self.NetName == "Fat_conv6":
            x, t = Variable(x_data), Variable(y_data)
            h = F.relu(self.model.conv1(x))
            h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
            h = F.relu(self.model.conv3(h))
            h = F.max_pooling_2d(F.relu(self.model.conv4(h)), 2)
            h = F.relu(self.model.conv5(h))
            h = F.max_pooling_2d(F.relu(self.model.conv6(h)), 2)
            h = F.dropout(F.relu(self.model.l1(h)), train=train)
            y = self.model.l2(h)
        elif self.NetName == "Fat_conv7":
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
        elif self.NetName == "Fat_conv8":
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
        elif self.NetName == 'Fat_Liner':
            x, t = Variable(x_data), Variable(y_data)
            h = F.dropout(F.relu(self.model.l1(x)),train=train)
            h = F.dropout(F.relu(self.model.l2(h)),train=train)
            h = self.model.l3(h)
            y  = h
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t),y

    def draw_1image(self,cate,idx=30):
        plt.figure(figsize=(10,10))
        plt.imshow(self.to_plot(self.color_x_train[idx]))
        plt.axis('off')

        plt.title(cate[self.y_train[idx]],size=15)
        #plt.tick_params(labelbottom="off")
        #plt.tick_params(labelleft="off")

    def draw_answerChack(self,cate,tate=5,yoko=5):
        print '-----AnswerCheckStart------'
        x,recog,ans,acc = self.config(tate*yoko)
        plt.figure(figsize=(10,10))
        for idx in range(tate*yoko):
            plt.subplot(tate, yoko, idx+1)
            plt.imshow\
            (self.to_plot(self.color_x_train[idx]))

            plt.axis('off')
            #plt.tick_params(labelbottom="off")
            #plt.tick_params(labelleft="off")

            AnswerCategory=cate[ans[idx]]
            ConfigCategory=cate[recog[idx]]

            plt.title("ans=%s,recog=%s"\
                %(AnswerCategory,ConfigCategory),size=8)
            plt.tick_params(labelbottom="off")
            plt.tick_params(labelleft="off")
            print "ans:%d,predict:%d"%(ans[idx],recog[idx])
        print "acc of draw_answerChack = %f"%(acc.data)
        print

    def to_plot(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    def to_grayscale(self,img):
        grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return grayed
    def blur(self,img):
        filtered = cv2.GaussianBlur(img, (11, 11), 0,0)
        return filtered
    def threshold(self,img):
        th1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
        return th1
if __name__=="__main__":
    #DataName_Netの特徴で記入
    NetName = "Fat_conv1"
    #AnswerCheckでだけ使う
    cate= ["Fat","Not_Fat"]

    #訓練だけなら 0 テストもするなら 1
    howloop = 1

    # 学習の繰り返し回数
    n_epoch   = 10
    n_display = n_epoch

    # 確率的勾配降下法で学習させる際の１回分のバッチサイズ
    batchsize = 100

    N = 188        #サンプル数
    N_train = 160    #訓練データの数
    N_test = N - N_train #テストデータの数

    unitSet   = [512]     #中間層ユニット数

    tate = 5
    yoko = 5
    n_config = tate*yoko
    n_predict = 10

    for i in range(5,6):
        if i == 9:
            NetName = "Fat_Liner"
        else :
            NetName = "Fat_conv%d"%(i)
        Fat =  FatCNN(N,N_train,N_test,unitSet,NetName)
        Fat.train_loop(howloop,n_epoch,batchsize,n_display)

        #Fat.draw_1image(cate,0)

        Fat.draw_answerChack(cate,tate,yoko)
        #Fat.predict(n_predict)
        plt.show()
