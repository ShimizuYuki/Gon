# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet,\
     optimizers,Chain,serializers
import chainer.functions  as F
import sys
import time
import datetime

fo = open("A.txt",'w')

A = [1,2,3,4]*20


A = map(str,A)
A = ",".join(A)
'''
for kaigyou in range(5,len(A),5):
    A.insert(kaigyou,"\n")
'''
fo.write(A)
fo.flush()
fo.close()

fo2 = open("A.txt",'r')
B = fo2.readline()
B = B.strip().split(',')
B = map(float,B)
fo2.close()

a = "okuyama"
b = "takahumi"

NetName = "Cifar_conv4"
Jikan1=str(datetime.datetime.now()).split()
Jikan2=Jikan1[1].split(":")
Jikan3=[Jikan1[0],Jikan2[0],Jikan2[1]]
Jikan='-'.join(Jikan3)
print\
('./modelkeep/'+NetName+'_Mode_'+Jikan)
