
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


class ColorCNN(ChainerNN.NN):
    def __init__(self,N,Ntr,Nte,unitSet,NetName):
        super(ColorCNN,self).__init__(N,Ntr,Nte,unitSet,NetName)

        self.x_train \
        = self.x_train.reshape((self.N_train, 3, 32, 32))

        self.x_test \
        =  self.x_test.reshape((self.N_test, 3, 32, 32))

    def draw_1image(self,number=30):
        size = 32
        x = self.x_test
        Z = x[number:number+1].reshape(3,size,size).transpose(1,2,0)
        plt.figure(figsize=(5, 5))
        plt.imshow(Z)
        #目盛りをなくす
        plt.axis('off')
        #plt.pcolor(Z)
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        #画像右側にバーをつける
        #plt.colorbar()
    #カテゴリーの指定忘れずに
    def draw_answerChack(self,cate,tate=5,yoko=5):
        print '-----AnswerCheckStart------'
        x,recog,ans,acc = self.config(tate*yoko)
        cnt=0
        plt.figure(figsize=(10,10))
        # Show Predict Result
        for idx in range(tate*yoko):
            # Forwarding for prediction
            cnt+=1
            size = 32
            plt.subplot(tate, yoko, cnt)
            # convert from vector to 32x32 matrix
            Z = x[idx:idx+1].reshape(3,size,size).transpose(1,2,0)
            plt.imshow(Z)
            plt.axis('off')

            AnswerCategory=cate[ans[idx]]
            ConfigCategory=cate[recog[idx]]

            plt.title("ans=%s,recog=%s"\
                %(AnswerCategory,ConfigCategory),size=8)
            plt.tick_params(labelbottom="off")
            plt.tick_params(labelleft="off")
            print "ans:%d,predict:%d"%(ans[idx],recog[idx])
        print "acc of draw_answerChack = %f"%(acc.data)
        print
