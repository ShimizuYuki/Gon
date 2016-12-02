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
import ChainerImg
#np.random.seed(0)
if __name__ == '__main__':
  #--------------------------------------------------------------------
  #変更する
  cate=["Human","NotHuman"]
  if cate[0]=="Human":
    NetName='-'.join(cate)
    model_name = "Human-NotHuman-conv5_Model_(D)test100-e2"
    conv = 5
  h = 75
  w = 50
  tate = 4
  yoko = 4
  img_num=tate*yoko
  read_num = 10
  N_array=np.zeros((len(cate)),dtype=np.int32)+read_num
  #N_array=[20,1]

  #--------------------------------------------------------------------
  print "----------------------------------------------"
  temple=u'Input a number\n0->Both of config and answerCheck\n1->Only config\n>>>'
  n = input(temple)
  #raw_inputはなんでもありで文字型として受け取る
  print "----------------------------------------------"

  #Both of config and answerCheck
  if n == 0:
    folder_list=cate
    data_list,label_list = imgKanren.makeDataList(folder_list,N_array)
    dataset,labelset = imgKanren.gattaiResize(data_list,label_list,h,w)
    s_dataset,s_labelset = imgKanren.shuffle(dataset,labelset)

    color_x , gray_x = imgKanren.preparNNdata(s_dataset)

    s_labelset = s_labelset.astype(np.int32)
    color_x = color_x.astype(np.float32)
    gray_x = gray_x.astype(np.float32)

    imgCNN=ChainerImg.NN(NetName,conv)
    imgCNN.set_Model_Opti(cate,h,w)
    serializers.load_npz('./ModelKeep/'+model_name,imgCNN.model)
    imgCNN.draw_answerCheck(gray_x,s_labelset,color_x,cate,img_num,tate,yoko)

  #Only config
  elif n==1:
    folder_list = [NetName+"-ConfigTest"]
    img_num=len(os.listdir('./DataKeep/Picture/'+folder_list[0]+'/'))
    a=np.sqrt(img_num)
    # 切り上げ (大きい側の整数に丸める)
    a = np.ceil(a)
    tate = a
    yoko = a
    print u"%sフォルダ内の画像をConfig"%(folder_list[0])
    print "----------------------------------------------"
    data_list,label_list = imgKanren.makeDataList(folder_list,N_array)
    dataset,labelset = imgKanren.gattaiResize(data_list,label_list,h,w)

    #引数が s_datasetではない
    color_x , gray_x = imgKanren.preparNNdata(dataset)

    labelset = labelset.astype(np.int32)
    color_x = color_x.astype(np.float32)
    gray_x = gray_x.astype(np.float32)
    imgCNN=ChainerImg.NN(NetName,conv)
    imgCNN.set_Model_Opti(cate,h,w)
    serializers.load_npz('./ModelKeep/'+model_name,imgCNN.model)

    imgCNN.draw_config(gray_x,labelset,data_list[0],cate,img_num,tate,yoko)
  plt.show()

  #--------------------------------------------------------------------
