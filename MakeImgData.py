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
#np.random.seed(0)
if __name__ == '__main__':

  print "----------------------------------------------"
  temple=u'Input a number\n0->Input folder_list\n1->No input\n>>>'
  n = input(temple)
  #raw_inputはなんでもありで文字型として受け取る
  if n == 0:
    folder_list = raw_input(u'folder_list\n>>> ').split()
    print "----------------------------------------------"
  else:
    folder_list=["Human","NotHuman"]
    #folder_list=["Jikken"]
  cate=folder_list
  #何枚の画像を読み込むか(フォルダ内の画像数より大きいとフォルダ内の画像すべてが読み込まれる)
  #read_num = 8
  read_num = [100,100]
  mizumasi_num = 1
  #読み込ませるpos,negの枚数
  N_array=np.zeros((len(folder_list)),dtype=np.int32)+read_num
  #N_array=[6,10]
  #各フォルダの画像数の合計✖フォルダ数✖水増し数（144）=63360
  #NNに突っ込むサンプル数N（できたdatasetより大きいとN=len(dataset)）
  #N = read_num*len(folder_list)*mizumasi_num
  N = (N_array[0]+N_array[1])*mizumasi_num
  #トレーニングデータの数
  N_train = (N*7)/10


  #resizeする時の画像の高さ h,幅 w
  h = 75
  w = 50
  #mizumasi_num = 72
  #if mizumasi_num !=144 and mizumasi_num != 72:
    #print u"mizumasi_numは72か144にしてください"
    #sys.exit()

  imgKanren.makeData(folder_list,N_array,h,w,N,N_train,mizumasi_num)
