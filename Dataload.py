# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet,\
     optimizers,Chain,serializers
import chainer.functions  as F
import sys

import cv2

import pickle

# mnist.data => 70,000件の784次元ベクトルデータ
def mnistLoad(N,N_train):
    mnist = fetch_mldata('MNIST original',\
            data_home = '.')
    x = mnist.data
    # 教師データ
    label = mnist.target
    dataset = np.column_stack((x, label))
    np.random.shuffle(dataset) #データ点の順番をシャッフル

    x = dataset[:, :28*28]
    label = dataset[:, 28*28:]
    label = label.ravel()

    x   = x.astype(np.float32)
    label = label.astype(np.int32)
    x  /= 255     # 0-1のデータに変換

    #データ分割（訓練とテスト）
    x_train, x_test , x_notUse = np.split(x, [N_train,N])
    y_train, y_test , y_notUse = np.split(label, [N_train,N])
    return x_train,x_test,y_train,y_test

# CIFAR-10
# 80 million tiny imagesのサブセット
# Alex Krizhevsky, Vinod Nair, Geoffrey Hintonが収集
# 32x32のカラー画像60000枚
# 10クラスで各クラス6000枚
# 50000枚の訓練画像と10000枚（各クラス1000枚）のテスト画像
# クラスラベルは排他的
# PythonのcPickle形式で提供されている

def cifarLoad(N,N_train,Ni=32*32,No=10):

    a=0

    train_data = []
    train_target = []

    datadir = "DataKeep/cifar10"
    # 訓練データをロード
    for i in range(1, 6):
        d = unpickle("%s/data_batch_%d" % (datadir, i))
        train_data.extend(d["data"])
        train_target.extend(d["labels"])

    # テストデータをロード
    d = unpickle("%s/test_batch" % (datadir))
    test_data = d["data"]
    test_target = d["labels"]

    # ラベル名をロード
    #label_names = unpickle("%s/batches.meta" % (datadir))["label_names"]


    if a==0:
        x = np.vstack((train_data,test_data))
        label = np.hstack((train_target,test_target))

        dataset = np.column_stack((x, label))
        np.random.shuffle(dataset) #データ点の順番をシャッフル
        x = dataset[:, :32*32*3]
        label = dataset[:, 32*32*3:]
        label = label.ravel()

        x   = x.astype(np.float32)
        label = label.astype(np.int32)
        x  /= 255     # 0-1のデータに変換

        #データ分割（訓練とテスト）
        x_train, x_test , x_notUse = np.split(x, [N_train,N])
        y_train, y_test , y_notUse = np.split(label, [N_train,N])
        return x_train,x_test,y_train,y_test

    if a==1:

        # データはfloat32、ラベルはint32のndarrayに変換
        train_data = np.array(train_data, dtype=np.float32)
        train_target = np.array(train_target, dtype=np.int32)
        test_data = np.array(test_data, dtype=np.float32)
        test_target = np.array(test_target, dtype=np.int32)

        # 画像のピクセル値を0-1に正規化
        train_data /= 255.0
        test_data /= 255.0

        return train_data, test_data, train_target, test_target

def FatLoad(N,N_train,height,width):
    for i in range(N/2):
        #BeforeAfter = cv2.imread('./DataKeep/Picture/BeforeAfter/BeforeAfter%d.jpg'%(i+1))
        Before =cv2.imread\
        ('./DataKeep/Picture/Before/Before%d.jpg'%(i+1))
        After =cv2.imread\
        ('./DataKeep/Picture/After/After%d.jpg'%(i+1))

        Before=cv2.resize(Before,(width,height))
        After=cv2.resize(After,(width,height))

        Before = Before.reshape(height*width*3)
        After = After.reshape(height*width*3)
        Before = np.hstack((Before,np.array([0])))
        After = np.hstack((After,np.array([1])))
        BeforeAfter=np.vstack((Before,After))
        if i == 0:
            dataset = BeforeAfter
        elif i >= 1:
            dataset = np.vstack((dataset,BeforeAfter))

    np.random.shuffle(dataset)

    #プロット用のカラー画像準備
    color_x = dataset[:,:height*width*3]
    color_x = color_x.reshape(N,height,width,3)
    color_x = color_x.astype(np.uint8)

    #NNに突っ込むグレースケール画像準備
    x = np.zeros((N,height,width))
    for i in range(N):
        #x[i]=to_grayscale(color_x[i]).reshape(1,height,width)
        x[i]=cv2.cvtColor(color_x[i],cv2.COLOR_BGR2GRAY)

    x=x.reshape(N,1,height,width)

    label = dataset[:,height*width*3:]
    label = label.ravel()

    x  = x.astype(np.float32)
    label = label.astype(np.int32)
    x  /= 255     # 0-1のデータに変換

    #データ分割（訓練とテスト）
    x_train, x_test , x_notUse = np.split(x, [N_train,N])
    color_x_train, color_x_test , x_notUse = np.split(color_x, [N_train,N])
    y_train, y_test , y_notUse = np.split(label, [N_train,N])

    return x_train,x_test,\
           color_x_train, color_x_test,\
           y_train,y_test

def Load(N,N_train,DataName,height=80,width=40):
    if DataName == "Mnist":
        return mnistLoad(N,N_train)
    elif DataName == "Cifar":
        return cifarLoad(N,N_train)
    elif DataName == "Fat":
        return FatLoad(N,N_train,height,width)

def to_plot(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def to_grayscale(img):
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayed
def blur(img):
    filtered = cv2.GaussianBlur(img, (11, 11), 0,0)
    return filtered
def threshold(img):
    th1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    return th1

def unpickle(f):
    import cPickle
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d

def unpickleImg(Hozondir):
    print u"unpickle開始"
    nL =["m_data_list", "m_label_list", "dataset", "labelset", \
    "s_dataset", "s_labelset", "color_x" , "gray_x" , "x", \
    "x_train", "x_test", "y_train", "y_test"]
    unpickle_num = len(nL)
    aL = []
    for i in range(unpickle_num):
        filename = nL[i]+'.dump'
        fo = open('./DataKeep/'+Hozondir+'/'+filename, 'r')
        d=pickle.load(fo)
        aL.append(d)
        print nL[i]+u"のunpickle完了"
    print u"全てのunpickle完了"
    print "----------------------------------------------"
    print "Each Info"
    print "len(m_data_list) : " , len(aL[0])
    print "len(m_label_list) : " , len(aL[1])
    for j in range(2,len(aL)):
        print nL[j]+".shape : ",aL[j].shape
    print "----------------------------------------------"
    return aL


if __name__ == '__main__':
    print "----------------------------------------------"
    Hozondir = "dumpCharhanGyouza"
    m_data_list, m_label_list, dataset, labelset, \
    s_dataset, s_labelset, color_x , gray_x , x, \
    x_train, x_test, y_train, y_test\
    =unpickleImg(Hozondir)
