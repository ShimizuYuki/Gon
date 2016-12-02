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

import Dataload
#np.random.seed(0)

#--------------------------------------------------------------------------

def to_plot(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def to_grayscale(img):
  grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return grayed
def threshold(img):
  th1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
          cv2.THRESH_BINARY,11,2)
  return th1

#--------------------------------------------------------------------------

def makeImgSet(N,FolderName,label):
  try :
    imgset = []
    img_list = os.listdir('./DataKeep/Picture/'+FolderName+'/')
    random.shuffle(img_list)
    kakutyousi_list = []
  except OSError:
    print "\n\n"+FolderName+\
    u"フォルダが見つかりません!!!\n\n"
    raise

  #print img_list
  print FolderName,u'フォルダ'
  val = 0
  for img_name in (img_list):
    kakutyousi = img_name.split(".")[1]
    if kakutyousi not in kakutyousi_list:
      kakutyousi_list.append(kakutyousi)
    if '.jpg' in img_name or '.jpeg' in img_name \
        or '.png' in img_name or '.JPG' in img_name:
      img =cv2.imread\
      ('./DataKeep/Picture/'+FolderName+'/'+img_name)
      #print u"画像?"+img_name+u"を読み込みました"
      val+=1
      imgset.append(img)
      if val==N:
        break
  print u"存在する拡張子:",kakutyousi_list
  print len(imgset),u'枚画像読み込み完了'
  print "----------------------------------------------"
  label=np.zeros((len(imgset)))+label
  return imgset,label

def makeDataList(folder_list,N_array):
  class_num = len(folder_list)
  data_list = []
  label_list = []
  print "----------------------------------------------"
  for i in xrange(class_num):
    imgset , label = makeImgSet(N_array[i],folder_list[i], i)
    data_list.append(imgset)
    label_list.append(label)
  print u"全フォルダー内の画像読み込み完了"
  print "len(data_list) : ",len(data_list)
  for i in xrange(class_num):
    print "len(data_list[%d]) : "%(i),len(data_list[i])
    print "len(label_list[%d]) : "%(i),len(label_list[i])
  print "data_list[0][0].shape : ",data_list[0][0].shape
  print "----------------------------------------------"
  return data_list , label_list

def gattaiResize(data_list,label_list,h,w):
  class_num = len(data_list)
  for i in xrange(class_num):
    imgset=[0]*len(data_list[i])
    imgset[:]=data_list[i][:]
    label = label_list[i]
    for j in xrange(len(imgset)):
      imgset[j]=cv2.resize(imgset[j],(w,h))
    imgset = np.array(imgset)
    print i+1,u"つ目のフォルダの画像resize完了"
    print "imgset.shape : ",imgset.shape
    print "----------------------------------------------"
    if i== 0:
      dataset = imgset
      g_label = label
    elif i >= 1:
      dataset = np.vstack((dataset,imgset))
      g_label = np.hstack((g_label,label))
  print u"全ての画像gattai完了"
  print "dataset.shape : ",dataset.shape
  print "----------------------------------------------"
  return dataset,g_label

def shuffle(dataset,g_label):
  print u"Shuffle開始"
  print "g_label[0:10] : ",g_label[0:10]
  N,h,w,c = dataset.shape
  s_dataset = np.zeros_like(dataset)
  s_dataset[:]=dataset[:]
  s_label = np.zeros_like(g_label)
  s_label[:]=g_label[:]

  s_dataset = s_dataset.reshape(N,h*w*c)
  if s_label.shape[0] != N:
    print u"ラベルとdatasetのサイズが不一致"
  dataset_label = np.column_stack((s_dataset,s_label))
  np.random.shuffle(dataset_label)
  s_dataset = dataset_label[:,:h*w*c]
  s_label = dataset_label[:,h*w*c:]

  s_dataset = s_dataset.reshape(N,h,w,c)

  s_label = s_label.ravel()
  print u"Shuffle完了"
  print "s_label[0:10] : ",s_label[0:10]
  print "----------------------------------------------"
  return s_dataset,s_label

def preparNNdata(dataset):
  N,h,w = dataset.shape[0:3]
  #プロット用のカラー画像準備
  color_x = np.zeros_like(dataset)
  color_x[:] = dataset[:]
  color_x=color_x.astype(np.uint8)
  #グレースケール画像準備
  gray_x=np.zeros((N,h,w))
  for i in xrange(N):
    gray_x[i]=cv2.cvtColor(color_x[i],cv2.COLOR_BGR2GRAY)
  gray_x = gray_x.reshape(N,1,h,w)
  color_x = color_x.transpose(0,3,1,2)
  #gray_x  /= 255.0
  #color_x  /= 255.0
  print u"NNに入れるデータ準備完了"
  print "color_x.shape : ",color_x.shape
  print "gray_x.shape : ",gray_x.shape
  print "----------------------------------------------"
  return color_x , gray_x

def TrainTestBunkatu(N,N_train,x,label):
  x_train, x_test , x_notUse = np.split(x, [N_train,N])
  y_train, y_test , y_notUse = np.split(label, [N_train,N])
  print u"トレーニングとテストデータ準備完了"
  print "x_train.shape : ",x_train.shape
  print "x_test.shape : ",x_test.shape
  print "y_train.shape : ",y_train.shape
  print "y_test.shape : ",y_test.shape
  print "----------------------------------------------"
  return x_train,x_test,\
         y_train,y_test

#-------------------------------------------------------------------------

def Flip(img):
  new_img = np.zeros_like(img)
  new_img[:] = img[:]
  row,col,ch= new_img.shape
  hflip_img = cv2.flip(new_img, 1)
  vflip_img = cv2.flip(new_img, 0)
  return hflip_img,vflip_img

def SPnoise(img):
  new_img = np.zeros_like(img)
  new_img[:] = img[:]
  row,col,ch= new_img.shape

  s_vs_p = 0.5
  amount = 0.004
  sp_img = new_img.copy()

  # 塩モード
  num_salt = np.ceil(amount * new_img.size * s_vs_p)
  coords = [np.random.randint(0, i-1 , int(num_salt)) for i in new_img.shape]
  sp_img[coords[:-1]] = (255,255,255)

  # 胡椒モード
  num_pepper = np.ceil(amount* new_img.size * (1. - s_vs_p))
  coords = [np.random.randint(0, i-1 , int(num_pepper)) for i in new_img.shape]
  sp_img[coords[:-1]] = (0,0,0)
  return sp_img

def noise(img):
  new_img = np.zeros_like(img)
  new_img[:] = img[:]
  row,col,ch= new_img.shape
  mean = 0
  sigma = 15
  gauss = np.random.normal(mean,sigma,(row,col,ch))
  gauss = gauss.reshape(row,col,ch)
  gauss_img = new_img + gauss

  return gauss_img

def blur(img):
  new_img = np.zeros_like(img)
  new_img[:] = img[:]
  blured = cv2.blur(new_img, (10, 10))
  return blured

def gamma(img):
  new_img = np.zeros_like(img)
  new_img[:] = img[:]

  # ガンマ変換ルックアップテーブル
  gamma1 = 0.75
  gamma2 = 1.5

  LUT_G1 = np.arange(256, dtype = 'uint8' )
  LUT_G2 = np.arange(256, dtype = 'uint8' )

  for i in range(256):
    LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
    LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)
  # 変換
  new_img=new_img.astype(np.uint8)
  high_cont_img = cv2.LUT(new_img, LUT_G1)
  low_cont_img = cv2.LUT(new_img, LUT_G2)
  return high_cont_img,low_cont_img

def contrast(img):
  new_img = np.zeros_like(img)
  new_img[:] = img[:]
  # ルックアップテーブルの生成
  min_table = 50
  max_table = 205
  diff_table = max_table - min_table

  LUT_HC = np.arange(256, dtype = 'uint8' )
  LUT_LC = np.arange(256, dtype = 'uint8' )

  # ハイコントラストLUT作成
  for i in range(0, min_table):
    LUT_HC[i] = 0
  for i in range(min_table, max_table):
    LUT_HC[i] = 255 * (i - min_table) / diff_table
  for i in range(max_table, 255):
    LUT_HC[i] = 255

  # ローコントラストLUT作成
  for i in range(256):
    LUT_LC[i] = min_table + i * (diff_table) / 255

  # 変換
  new_img=new_img.astype(np.uint8)
  high_cont_img = cv2.LUT(new_img, LUT_HC)
  low_cont_img = cv2.LUT(new_img, LUT_LC)
  return high_cont_img,low_cont_img

def imgMizumasi(img,mizumasi_num):
  mizumasi = [img]
  f_img , vf_img=  Flip(mizumasi[0])
  mizumasi.append(f_img)
  #mizumasi.append(vf_img)
  repeat = len(mizumasi)
  for val in range(repeat):
    sp_img =  SPnoise(mizumasi[val])
    mizumasi.append(sp_img)

  repeat = len(mizumasi)
  for val in range(repeat):
    n_img =  noise(mizumasi[val])
    mizumasi.append(n_img)

  if mizumasi_num==144:
    repeat = len(mizumasi)
    for val in range(repeat):
      b_img =  blur(mizumasi[val])
      mizumasi.append(b_img)

  repeat = len(mizumasi)
  for val in range(repeat):
    h_img , l_img = gamma(mizumasi[val])
    mizumasi.append(h_img)
    mizumasi.append(l_img)

  repeat = len(mizumasi)
  for val in range(repeat):
    h_img , l_img = contrast(mizumasi[val])
    mizumasi.append(h_img)
    mizumasi.append(l_img)

  return mizumasi

#生データ(resizeされていない)を水増しして，imwrite
def dataListMizumasi(data_list,folder_list):
  mizumasi_data_list = []
  mizumasi_label_list = []
  for folder in folder_list:
    Hozonsaki = folder+'_mizumasi'
    if Hozonsaki not in os.listdir('./DataKeep/Picture/') :
      os.mkdir('./DataKeep/Picture/'+Hozonsaki)

    f = folder_list.index(folder)
    imgset = data_list[f]
    mizumasi_imgset =[]
    for img in imgset:
      mizumasi144 = imgMizumasi(img)
      for src in mizumasi144:
        jikan = str(time.clock()).replace('.','-',1)
        cv2.imwrite\
        ('./DataKeep/Picture/'+Hozonsaki+'/'+jikan+'.jpg'%(),src)
        mizumasi_imgset.append(src)
    mizumasi_label = np.zeros((len(mizumasi_imgset))) + f
    mizumasi_data_list.append(mizumasi_imgset)
    mizumasi_label_list.append(mizumasi_label)
    print folder+u"フォルダー内の画像水増し完了"
    print "----------------------------------------------"
  print u"全フォルダー内の画像水増し完了"
  print "len(data_list) : ",len(mizumasi_data_list)
  for i in xrange(len(data_list)):
    print "len(data_list[%d]) : "%(i),len(mizumasi_data_list[i])
    print "len(label_list[%d]) : "%(i),len(mizumasi_label_list[i])

  print "data_list[0][0].shape : ",mizumasi_data_list[0][0].shape
  print "----------------------------------------------"
  return mizumasi_data_list , mizumasi_label_list

#resizeされたデータを水増し。imwriteしない
def dataSetMizumasi(dataset,N_array,mizumasi_num):
  print u"datasetの水増し開始，画像数は%dです"%(len(dataset))
  dataset = dataset.astype(np.uint8)
  i=0
  for img in dataset:
    mizumasi144 = imgMizumasi(img,mizumasi_num)
    mizumasi144 = np.array(mizumasi144)
    if i== 0:
      m_dataset = mizumasi144
    elif i >= 1:
      m_dataset = np.vstack((m_dataset,mizumasi144))
    i+=1
    print i,u"枚目の水増し完了"
  print u"datasetの水増し完了"
  for j in xrange(len(N_array)):
    m_label=np.zeros((N_array[j]))+j
    if j== 0:
      m_labelset = m_label
    elif j >= 1:
      m_labelset = np.hstack((m_labelset,m_label))
  print u"labelsetの水増し完了"
  print "m_dataset.shape : ",m_dataset.shape
  print "m_labelset.shape : ",m_labelset.shape
  print "----------------------------------------------"
  return m_dataset , m_labelset

#-------------------------------------------------------------------------

def dump(aL,nL,Hozondir):
  print u"dump開始"
  dump_num = len(aL)
  if Hozondir not in os.listdir('../imgNNdata/') :
    os.mkdir('../imgNNdata/'+Hozondir)
  for i in xrange(dump_num):
    filename = nL[i]+'.pkl'
    #fo = open('../imgNNdata/'+Hozondir+'/'+filename,'wb')
    fo = os.path.join('../imgNNdata/'+Hozondir+'/', filename)
    #os.mkdir('../imgNNdata/'+Hozondir+'/'+filename)
    #fo = os.listdir('../imgNNdata/'+Hozondir+'/'+filename)
    #joblib.dump(aL[i], fo ,compress=3)
    joblib.dump(aL[i], fo )
    #joblib.dump(aL[i], filename )
    #fo.flush()
    #fo.close()
    print nL[i]+u"のdump完了"
    print nL[i]+".shape : ",aL[i].shape
  print u"全てのdump完了"
  print "----------------------------------------------"

def unpickleImg(Hozondir,nL):
  print u"unpickle開始"
  unpickle_num = len(nL)
  aL = []
  for i in range(unpickle_num):
    filename = nL[i]+'.pkl'
    #fo = open('../imgNNdata/'+Hozondir+'/'+filename,'rb')
    #fo = './DataKeep/'+Hozondir+'/'+filenam
    fo = os.path.join('../imgNNdata/'+Hozondir+'/', filename)
    d=joblib.load(fo)
    #fo.close()
    aL.append(d)
    print nL[i]+u"のunpickle完了"
    print nL[i]+".shape : ",aL[i].shape
  print u"全てのunpickle完了"
  print "----------------------------------------------"
  return aL

#-------------------------------------------------------------------------

def draw_img(img,title):
  if np.max(img) <= 1.0:
    img*=255
  img=img.astype(np.uint8)
  plt.title(title,size=10)
  plt.tick_params(labelbottom="off")
  plt.tick_params(labelleft="off")
  if len(img.shape)==3 :
    if img.shape[0]==1:
      h,w=img.shape[1:]
      img=img.reshape(h,w)
      draw_img(img,title)
    elif img.shape[0]==3:
      img = img.transpose(1,2,0)
      draw_img(img,title)
    else:
      plt.imshow(to_plot(img))
  else:
      plt.gray()
      plt.imshow(img)

def draw_imgSet(imgset,title_list,img_num,tate=1,yoko=3,random=True):
  plt.figure(figsize=(10,10))
  r = np.random.permutation(len(imgset))
  for idx in xrange(img_num):
    if random==True:
      img = imgset[r[idx]]
      title = title_list[r[idx]]
    #ランダムに画像をピックアップしない場合
    elif random==False:
        img = imgset[idx]
        title = title_list[idx]
    plt.subplot(tate, yoko, idx+1)
    draw_img(img,title)

#-------------------------------------------------------------------------

def Cut_img(img):
  Dir = "./haarcascadesWin/"
  #cascade = cv2.CascadeClassifier(Dir+'haarcascade_frontalface_default.xml')
  #cascade = cv2.CascadeClassifier(Dir+'haarcascade_eye.xml')
  cascade = cv2.CascadeClassifier(Dir+'haarcascade_fullbody.xml')
  #cascade = cv2.CascadeClassifier(Dir+'haarcascade_upperbody.xml')
  #cascade = cv2.CascadeClassifier(Dir+'haarcascade_lowerbody.xml')

  #drawed_img = img
  drawed_img = np.zeros_like(img)
  drawed_img[:] = img[:]
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


  bodys= cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(3, 3))
  if len(bodys) > 0:
    print ("%d Mituketa!!"%(len(bodys)))
    for (x,y,w,h) in bodys:
      cv2.rectangle(drawed_img,(x,y),(x+w,y+h),(255,0,0),2)
      roi_gray = img[y:y+h, x:x+w]
      roi_color = drawed_img[y:y+h, x:x+w]
  else:
      print("No!!!")
  return drawed_img

def Cut_imgSet(imgset,Cut_num):
  Cut_imgset = []
  for i in xrange(Cut_num):
    Cut_imgset.append(Cut_img(imgset[i]))
  return Cut_imgset

#--------------------------------------------------------------------------
def makeData(folder_list,N_array,h,w,N,N_train,mizumasi_num):
  #テストストデータの数
  #18360
  N_test = N-N_train
  data_list,label_list = makeDataList(folder_list,N_array)
  for i in xrange(len(N_array)):
    N_array[i]=len(data_list[i])
  for i in xrange(len(folder_list)):
    if i==0:
      Hozondir = folder_list[i]
    else:
      Hozondir += folder_list[i]
  #-------------------------------------------------------------------------
  #水増し!!!!
  N_array = N_array*mizumasi_num
  ##data_list, label_list = dataListMizumasi(data_list,folder_list)
  dataset,labelset = gattaiResize(data_list,label_list,h,w)
  #dataset,labelset = dataSetMizumasi(dataset,N_array,mizumasi_num)
  #--------------------------------------------------------------------------
  s_dataset,s_labelset = shuffle(dataset,labelset)
  color_x , gray_x = preparNNdata(s_dataset)
  if N > len(color_x):
    N=len(color_x)
    N_train = (N*7)/10

  x_train,x_test,y_train,y_test=TrainTestBunkatu(N,N_train,gray_x,s_labelset)
  #draw_imgSet(s_dataset,s_labelset,cate,4,4)
  #plt.show()
  Jikan1=str(datetime.datetime.now()).split()
  Jikan2=Jikan1[1].split(":")
  Jikan3=[Jikan1[0],Jikan2[0],Jikan2[1]]
  Jikan='-'.join(Jikan3)
  dumpHozondir = "dump" + Hozondir+Jikan
  nL =["color_x","s_labelset","gray_x","x_train","x_test","y_train","y_test"]
  aL = [color_x, s_labelset, gray_x,  x_train , x_test , y_train , y_test]
  dump(aL,nL,dumpHozondir)

if __name__ == '__main__':
  print "----------------------------------------------"
  nL =["color_x","s_labelset","gray_x","x_train","x_test","y_train","y_test"]
  #nL =["x_train"]
  Hozondir = "dumpHumanNoHumantmizumasi72"
  #Hozondir = "dumpCharhanPlain72"
  color_x,s_labelset,gray_x,x_train,x_test,y_train,y_test\
  =unpickleImg(Hozondir,nL)
  s_labelset = s_labelset.astype(np.int32)
  color_x = color_x.astype(np.float32)
  gray_x = gray_x.astype(np.float32)
  x_train = x_train.astype(np.float32)
  x_test = x_test.astype(np.float32)
  y_train = y_train.astype(np.int32)
  y_test = y_test.astype(np.int32)
  cate=["Human","NotHuman"]

  title_list = []
  for i in range(len(s_labelset)):
    ans   = cate[s_labelset[i]]
    title_list.append(ans)
  draw_imgSet(gray_x,title_list,4,4)
  #draw_imgSet(gray_x,title_list,4,4,False)

  plt.show()
