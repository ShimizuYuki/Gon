# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.externals import joblib
from chainer import cuda, Variable, FunctionSet,\
     optimizers,Chain,serializers
import chainer.functions  as F
import cv2

import sys
import time
import datetime
import os
import random

import imgKanren
import Dataload
#np.random.seed(0)
class NN(object):
  def __init__(self,NetName,conv):
    self.NetName = NetName
    self.conv=conv
    print
    print "-----This is conv",conv,"_Net-----"
    self.StartTime = time.clock()
  #(入力のチャネル数,出力のチャネル数,フィルタサイズ)
  def set_Model_Opti(self,cate,h,w):
    N_output = len(cate)
    if self.conv == 1:
      self.model = FunctionSet(
        conv1=F.Convolution2D(1, h, 3, pad=1),
        #l1=F.Linear(6272, 512),
	l1=F.Linear(71250, 512),
        l2=F.Linear(512, N_output))
    elif self.conv == 2:
      self.model = FunctionSet(
        conv1=F.Convolution2D(1, h, 3, pad=1),
        conv2=F.Convolution2D(h, h, 3, pad=1),
        l1=F.Linear(71250, 1000),
        l2=F.Linear(1000, N_output))
    elif self.conv == 3:
      self.model = FunctionSet(
        conv1=F.Convolution2D(1, h, 3, pad=1),
        conv2=F.Convolution2D(h, h, 3, pad=1),
        conv3=F.Convolution2D(h, h, 3, pad=1),
        l1=F.Linear(1568, 512),
        l2=F.Linear(512, N_output))
    elif self.conv == 4:
      self.model = FunctionSet(
        conv1=F.Convolution2D(1, h, 3, pad=1),
        conv2=F.Convolution2D(h, h, 3, pad=1),
        conv3=F.Convolution2D(h, h, 3, pad=1),
        conv4=F.Convolution2D(h, h, 3, pad=1),
        l1=F.Linear(1568, 512),
        l2=F.Linear(512, N_output))
    elif self.conv == 5:
      self.model = FunctionSet(
        conv1=F.Convolution2D(1, h, 3, pad=1),
        conv2=F.Convolution2D(h, h, 3, pad=1),
        conv3=F.Convolution2D(h, h, 3, pad=1),
        conv4=F.Convolution2D(h, h, 3, pad=1),
        conv5=F.Convolution2D(h, h, 3, pad=1),
        l1=F.Linear(5250, 5000),
        l2=F.Linear(5000, N_output))
    elif self.conv == 6:
      self.model = FunctionSet(
        conv1=F.Convolution2D(1, h, 3, pad=1),
        conv2=F.Convolution2D(h, h, 3, pad=1),
        conv3=F.Convolution2D(h, h, 3, pad=1),
        conv4=F.Convolution2D(h, h, 3, pad=1),
        conv5=F.Convolution2D(h, h, 3, pad=1),
        conv6=F.Convolution2D(h, h, 3, pad=1),
        #l1=F.Linear(512, 512),
	l1=F.Linear(5250, 5000),
        l2=F.Linear(5000, N_output))
    elif self.conv == 7:
      self.model = FunctionSet(
        conv1=F.Convolution2D(1, h, 3, pad=1),
        conv2=F.Convolution2D(h, h, 3, pad=1),
        conv3=F.Convolution2D(h, h, 3, pad=1),
        conv4=F.Convolution2D(h, h, 3, pad=1),
        conv5=F.Convolution2D(h, h, 3, pad=1),
        conv6=F.Convolution2D(h, h, 3, pad=1),
        conv7=F.Convolution2D(h, h, 3, pad=1),
        l1=F.Linear(128, 128),
        l2=F.Linear(128, N_output))
    elif self.conv == 8:
      self.model = FunctionSet(
        conv1=F.Convolution2D(1, h, 3, pad=1),
        conv2=F.Convolution2D(h, h, 3, pad=1),
        conv3=F.Convolution2D(h, h, 3, pad=1),
        conv4=F.Convolution2D(h, h, 3, pad=1),
        conv5=F.Convolution2D(h, h, 3, pad=1),
        conv6=F.Convolution2D(h, h, 3, pad=1),
        conv7=F.Convolution2D(h, h, 3, pad=1),
        conv8=F.Convolution2D(h, h, 3, pad=1),
        l1=F.Linear(128, 128),
        l2=F.Linear(128, N_output))
    else :
      self.model = FunctionSet(\
          l1=F.Linear(h*w, 1200),
          l2=F.Linear(1200, 1200),
          l3=F.Linear(1200, N_output)
          )

    #Optimaizerの設定
    self.optimizer = optimizers.Adam()
    self.optimizer.setup(self.model)
  def forward(self,x_data, y_data, train=True):
    if self.conv==1:
      x, t = Variable(x_data), Variable(y_data)
      h = F.max_pooling_2d(F.relu(self.model.conv1(x)), 2)
      h = F.dropout(F.relu(self.model.l1(h)), train=train)
      y = self.model.l2(h)
    elif self.conv==2:
      x, t = Variable(x_data), Variable(y_data)
      h = F.relu(self.model.conv1(x))
      h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
      h = F.dropout(F.relu(self.model.l1(h)), train=train)
      y = self.model.l2(h)
    elif self.conv==3:
      x, t = Variable(x_data), Variable(y_data)
      h = F.relu(self.model.conv1(x))
      h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
      h = F.max_pooling_2d(F.relu(self.model.conv3(h)), 2)
      h = F.dropout(F.relu(self.model.l1(h)), train=train)
      y = self.model.l2(h)
    elif self.conv==4:
      x, t = Variable(x_data), Variable(y_data)
      h = F.relu(self.model.conv1(x))
      h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
      h = F.relu(self.model.conv3(h))
      h = F.max_pooling_2d(F.relu(self.model.conv4(h)), 2)
      h = F.dropout(F.relu(self.model.l1(h)), train=train)
      y = self.model.l2(h)
    elif self.conv==5:
      x, t = Variable(x_data), Variable(y_data)
      h = F.relu(self.model.conv1(x))
      h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
      h = F.relu(self.model.conv3(h))
      h = F.max_pooling_2d(F.relu(self.model.conv4(h)), 2)
      h = F.max_pooling_2d(F.relu(self.model.conv5(h)), 2)
      h = F.dropout(F.relu(self.model.l1(h)), train=train)
      y = self.model.l2(h)
    elif self.conv==6:
      x, t = Variable(x_data), Variable(y_data)
      h = F.relu(self.model.conv1(x))
      h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
      h = F.relu(self.model.conv3(h))
      h = F.max_pooling_2d(F.relu(self.model.conv4(h)), 2)
      h = F.relu(self.model.conv5(h))
      h = F.max_pooling_2d(F.relu(self.model.conv6(h)), 2)
      h = F.dropout(F.relu(self.model.l1(h)), train=train)
      y = self.model.l2(h)
    elif self.conv==7:
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
    elif self.conv==8:
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
    else:
      x, t = Variable(x_data), Variable(y_data)
      h = F.dropout(F.relu(self.model.l1(x)),train=train)
      h = F.dropout(F.relu(self.model.l2(h)),train=train)
      h = self.model.l3(h)
      y  = h
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t),y

  #--------------------------------------------------------------------------

  def train_loop(self,howloop,n_epoch,batchsize,\
                 x_train,y_train,x_test,y_test):
    N_train = len(x_train)
    N_test = len(x_test)
    N = N_train+N_test
    train_loss = []
    train_acc  = []
    test_loss = []
    test_acc  = []
    # Learning loop
    for epoch in xrange(1, n_epoch+1):
      '''
      訓練
      '''
      sum_trainloss = 0
      sum_trainacc = 0
      # 0〜Nまでのデータをバッチサイズごとに使って学習
      for i in xrange(0, N_train, batchsize):
        x_batch = x_train[i:i+batchsize]
        y_batch = y_train[i:i+batchsize]
        # 勾配を初期化
        self.optimizer.zero_grads()
        # 順伝播させて誤差と精度を算出
        loss,acc,y = self.forward(x_batch,y_batch)
        # 誤差のみ
        #loss = self.forward(x_batch, y_batch,train=False)
        # 誤差逆伝播で勾配を計算
        loss.backward()
        self.optimizer.update()
        train_loss.append(loss.data)
        train_acc.append(acc.data)
        sum_trainloss += loss.data * batchsize
        sum_trainacc  += acc.data  * batchsize
        print 'epoch:%d'%(epoch)+u"のTrain %f"%((i+batchsize)*100.0/N_train)+u"% 終了"
      mean_trainloss = sum_trainloss / N_train
      mean_trainacc = sum_trainacc / N_train
      print "----------------------------------------------"
      Jikan1=str(datetime.datetime.now()).split()
      Jikan2=Jikan1[1].split(":")
      Jikan3=[Jikan1[0],Jikan2[0],Jikan2[1]]
      Jikan='-'.join(Jikan3)+str(epoch)
      #モデルの保存
      serializers.save_npz\
      ('./modelkeep/'+self.NetName+"-conv%d"%(self.conv)+'_Model_'+Jikan,self.model)
      print u"モデルを保存しました"
      print "----------------------------------------------"
      '''
      テスト
      '''
      #テストデータで誤差と正解精度を算出
      if (howloop==1):
        sum_testloss = 0
        sum_testacc     = 0
        for i in xrange(0, N_test, batchsize):
          x_batch = x_test[i:i+batchsize]
          y_batch = y_test[i:i+batchsize]
          # 順伝播させて誤差と精度を算出
          loss,acc,y\
          =self.forward(x_batch,y_batch,train=False)
          test_loss.append(loss.data)
          test_acc.append(acc.data)
          sum_testloss += loss.data * batchsize
          sum_testacc  += acc.data * batchsize
          print 'epoch:%d'%(epoch)+u"のTest %f"%((i+batchsize)*100.0/N_test)+u"% 終了"
        mean_testloss = sum_testloss / N_test
        mean_testacc  = sum_testacc  / N_test

        print "----------------------------------------------"
        print "----------------------------------------------"
        print "conv",self.conv,',epoch',epoch,u"終了"
        # 訓練データの誤差と、正解精度を表示
        print 'train mean loss=%f'%(mean_trainloss),\
              'train mean acc=%f'%(mean_trainacc)
        # テストデータの誤差と、正解精度を表示
        print 'test mean loss=%f'%(mean_testloss),\
              'test mean acc=%f'%(mean_testacc)
        print ('Time = %f'%(time.clock()-self.StartTime))
        print "----------------------------------------------"
        print "----------------------------------------------"
      else:
        print "----------------------------------------------"
        print "----------------------------------------------"
        print "conv",self.conv,',epoch',epoch,u"終了"
        # 訓練データの誤差と、正解精度を表示
        print 'train mean loss=%f'%(mean_trainloss),\
              'train mean acc=%f'%(mean_trainacc)
        print "----------------------------------------------"
        print "----------------------------------------------"

    #Loss,Accの保存
    self.LossAccKeep(train_loss\
    ,"LossAcc/"+self.NetName+"-conv%d"%(self.conv)+"_Train_Loss"+Jikan+".txt")
    self.LossAccKeep(train_acc\
    ,"LossAcc/"+self.NetName+"-conv%d"%(self.conv)+"_Train_Acc"+Jikan+".txt")
    self.LossAccKeep(test_loss\
    ,"LossAcc/"+self.NetName+"-conv%d"%(self.conv)+"_Test_Loss"+Jikan+".txt")
    self.LossAccKeep(test_acc\
    ,"LossAcc/"+self.NetName+"-conv%d"%(self.conv)+"_Test_Acc"+Jikan+".txt")
  def LossAccKeep(self,List,filename):
    fo = open(filename,'w')
    List_str = map(str,List)
    NotList = "\t".join(List_str)
    fo.write(NotList)
    fo.flush()
    fo.close()

  #--------------------------------------------------------------------------

  def draw_answerCheck(self,gx,y,cx,cate,img_num,tate=4,yoko=4):
    print '-----DrawAnswerCheckStart------'
    if len(gx)<tate*yoko:
      print u"len(gx)<tate*yokoになってるよ"
      print  "len(gx)=", len(gx)
      print  "tate*yoko=", tate*yoko
      sys.exit()
    gx = gx[0:img_num]
    cx = cx[0:img_num]
    ans_array = y[0:img_num]
    #認識した画像のクラスのナンバーリスト
    recog_array = self.config(gx,ans_array)[0]
    title_list = []
    for i in range(img_num):
      ans   = cate[ans_array[i]]
      recog = cate[recog_array[i]]
      title = "ans=%s,recog=%s"%(ans,recog)
      title_list.append(title)
    print "----------------------------------------------"
    imgKanren.draw_imgSet(cx,title_list,img_num,tate,yoko)

  def draw_config(self,gx,y,cx,cate,img_num,tate=1,yoko=1):
    if len(gx)<img_num:
      print u"len(gx)<img_numになってるよ"
      print  "len(gx)=", len(gx)
      print  "img_num=", img_num
      sys.exit()
    gx = gx[0:img_num]
    cx = cx[0:img_num]
    ans_array = y[0:img_num]
    #認識した画像のクラスのナンバーリスト
    recog_array = self.config(gx,ans_array)[0]
    title_list = []
    for i in range(img_num):
      recog = cate[recog_array[i]]
      title = "recog=%s"%(recog)
      title_list.append(title)
    imgKanren.draw_imgSet(cx,title_list,img_num,tate,yoko,False)

  def answerCheck(self,gx,y,n_predict):
    print '-----AnswerCheckStart------'
    if len(gx)<n_predict:
      print u"len(gx)<n_predictになってるよ"
      print  "len(gx)=", len(gx)
      print  "n_predict=", n_predict
      sys.exit()
    gx = gx[0:n_predict]
    ans_array = y[0:n_predict]
    recog_array,acc = self.config(gx,ans_array)
    #認識した画像のクラスのナンバーリスト
    recog_array,acc = self.config(gx,ans_array)
    for idx in range(n_predict):
      print "ans:%d,predict:%d"%(ans_array[idx],recog_array[idx])
    print "acc of predict = %f"%(acc.data)
    print "----------------------------------------------"
    return recog_array

  def config(self,x,ans_array):
    acc,recog_array=self.forward(x,ans_array,False)[1:3]
    #認識した画像のクラスのナンバーリスト
    recog_array = recog_array.data.argmax(axis=1)
    return recog_array,acc


if __name__ == '__main__':
  print "----------------------------------------------"
  #訓練だけなら 0 テストもするなら 1
  howloop = 1
  # 学習の繰り返し回数
  n_epoch   = 1
  # 確率的勾配降下法で学習させる際の１回分のバッチサイズ
  batchsize = 10
  #---------------------------------------------------------------------------
  #Hozondir = "dumpHumanNotHumanmizumasi72"
  #cate=["Human","NotHuman"]
  Hozondir = "dumpHumanNotHuman7-3"
  cate=["Human","NotHuman"]
  #Hozondir = "dumpCharhanGyouzaPasta2016-10-29-11-23"
  #cate = ["Charhan","Gyouza","Pasta"]
  NetName='-'.join(cate)
  #N_train = 22100
  #N_test = 9500
  #N_train = 10
  #N_test = 10
  N_train = 129679
  N_test = 55577
  conv=5
  #----------------------------------------------------------------------------
  tate = 3
  yoko = 3
  img_num = tate*yoko
  n_predict = 9


  #-----ロードするデータ名を格納するリスト------------------------------------
  nL =["color_x","s_labelset","gray_x","x_train","x_test","y_train","y_test"]
  color_x,s_labelset,gray_x,x_train,x_test,y_train,y_test\
  =imgKanren.unpickleImg(Hozondir,nL)
  color_x = color_x.astype(np.float32)
  gray_x = gray_x.astype(np.float32)
  x_train = x_train.astype(np.float32)
  x_test = x_test.astype(np.float32)
  y_train = y_train.astype(np.int32)
  y_test = y_test.astype(np.int32)
  h,w = color_x.shape[2:4]

  x_train = x_train[0:N_train]
  x_test  = x_test[0:N_test]
  y_train = y_train[0:N_train]
  y_test  = y_test[0:N_test]

  print u"実際にNNに突っ込む訓練,テストデータのShapes"
  print "x_train.shape : ",x_train.shape
  print "x_test.shape : ",x_test.shape
  print "y_train.shape : ",y_train.shape
  print "y_test.shape : ",y_test.shape
  print "----------------------------------------------"


  #------NN構築------------------------------------------------------------------------
  HumanCNN = NN(NetName,conv)
  HumanCNN.set_Model_Opti(cate,h,w)
  #HumanCNN.forward(x_train, y_train, train=False)

  HumanCNN.train_loop(howloop,n_epoch,batchsize,\
                 x_train,y_train,x_test,y_test)

  #(gx,y,cx,cate,img_num,tate,yoko)
  #HumanCNN.draw_answerCheck(x_train,y_train,color_x,cate,img_num,tate,yoko)
  #HumanCNN.draw_config(x_train,y_train,color_x,cate,img_num,tate,yoko)
  HumanCNN.answerCheck(x_train,y_train,n_predict)
  plt.show()
