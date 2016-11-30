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
import Dataload

#np.random.seed(0)

class NN(object):
    def __init__(self,N,Ntr,Nte,unitSet,NetName):
        self.NetName = NetName
        self.DataName = NetName.split('_')[0]
        print
        print "-----This is %s_Net-----"%(self.NetName)
        self.N = N
        self.N_train = Ntr
        self.N_test = Nte
        self.unitSet = unitSet

        self.train_test_dataLoad()
        self.set_Model_Opti()

        self.StartTime = time.clock()

    def train_test_dataLoad(self):
        self.x_train,self.x_test,self.y_train,self.y_test\
        =Dataload.Load(self.N,self.N_train,self.DataName)

    def set_Model_Opti(self):
        self.model = FunctionSet(\
            l1=F.Linear(28*28, self.unitSet[0]),
            l2=F.Linear(self.unitSet[0], self.unitSet[0]),
            l3=F.Linear(self.unitSet[0], 10)
            )
        #Optimaizerの設定
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    def forward(self,x_data, y_data, train=True):
        x = Variable(x_data)
        t = Variable(y_data)
        z1 = F.dropout(F.relu(self.model.l1(x)),train=train)
        z2 = F.dropout(F.relu(self.model.l2(z1)),train=train)
        z3 = self.model.l3(z2)
        y  = z3
        # 多クラス分類なので誤差関数としてソフトマックス関数の
        # 交差エントロピー関数を用いて、誤差を導出
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t),y

    def train_loop(self,howloop,n_epoch,batchsize,n_display):
        train_loss = []
        train_acc  = []
        test_loss = []
        test_acc  = []

        l1_W = []
        l2_W = []
        l3_W = []

        # Learning loop
        for epoch in xrange(1, n_epoch+1):
            '''
            訓練
            '''
            sum_trainloss = 0
            sum_trainacc = 0
            # 0〜Nまでのデータをバッチサイズごとに使って学習
            for i in xrange(0, self.N_train, batchsize):
                x_batch = self.x_train[i:i+batchsize]
                y_batch = self.y_train[i:i+batchsize]

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
            mean_trainloss = sum_trainloss / self.N_train
            mean_trainacc = sum_trainacc / self.N_train

            # 学習したパラメーターを保存
            #l1_W.append(self.model.l1.W)
            #l2_W.append(self.model.l2.W)
            #l3_W.append(model.l3.W)
            if (epoch % (n_epoch/n_display)==0):
                print self.NetName + ' epoch : ', epoch
                # 訓練データの誤差と、正解精度を表示
                print 'train mean loss=%f'%(mean_trainloss),\
                      'train mean acc=%f'%(mean_trainacc)
            '''
            テスト
            '''
            #テストデータで誤差と正解精度を算出
            if (howloop==1):
                sum_testloss = 0
                sum_testacc     = 0
                for i in xrange(0, self.N_test, batchsize):
                    x_batch = self.x_test[i:i+batchsize]
                    y_batch = self.y_test[i:i+batchsize]

                    # 順伝播させて誤差と精度を算出
                    loss,acc,y\
                    =self.forward(x_batch,y_batch,train=False)

                    # 誤差のみ
                    #loss\
                    #= self.forward(x_batch,y_batch,train=False)

                    test_loss.append(loss.data)
                    test_acc.append(acc.data)
                    sum_testloss += loss.data * batchsize
                    sum_testacc  += acc.data * batchsize

                mean_testloss = sum_testloss / self.N_test
                mean_testacc  = sum_testacc  / self.N_test

                if (epoch % (n_epoch/n_display)==0):
                    # テストデータの誤差と、正解精度を表示
                    print 'test mean loss=%f'%(mean_testloss),\
                          'test mean acc=%f'%(mean_testacc)
                    print ('Time = %f'%(time.clock()-self.StartTime))

        Jikan1=str(datetime.datetime.now()).split()
        Jikan2=Jikan1[1].split(":")
        Jikan3=[Jikan1[0],Jikan2[0],Jikan2[1]]
        Jikan='-'.join(Jikan3)

        #モデルの保存
        serializers.save_npz\
        ('./modelkeep/'+self.NetName+'_Model_'+Jikan,self.model)

        #Loss,Accの保存
        self.LossAccKeep(train_loss\
        ,"LossAcc/"+self.NetName+"_Train_Loss"+Jikan+".txt")
        self.LossAccKeep(train_acc\
        ,"LossAcc/"+self.NetName+"_Train_Acc"+Jikan+".txt")
        self.LossAccKeep(test_loss\
        ,"LossAcc/"+self.NetName+"_Test_Loss"+Jikan+".txt")
        self.LossAccKeep(test_acc\
        ,"LossAcc/"+self.NetName+"_Test_Acc"+Jikan+".txt")

    def LossAccKeep(self,List,filename):
        fo = open(filename,'w')
        List_str = map(str,List)
        NonList = ",".join(List_str)
        fo.write(NonList)
        fo.flush()
        fo.close()

    def draw_1image(self,number=30):
        size = 28

        X, Y = np.meshgrid(range(size+1),range(size+1))
        xxx = self.x_train[number:number+1]
        Z = xxx.reshape(size,size)

        Z = Z[::-1,:]

        plt.figure(figsize=(5, 5))
        plt.xlim(0,27)
        plt.ylim(0,27)
        plt.imshow(Z)
        #plt.pcolor(Z)
        plt.gray()
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        #画像右側にバーをつける
        #plt.colorbar()

    def config(self,n_config):
        #P = np.random.permutation(N_test)

        x = self.x_train[0:n_config]
        ans   = self.y_train[0:n_config]

        loss,acc,recog\
        =self.forward(x,ans,False)
        recog = recog.data.argmax(axis=1)

        return x,recog,ans,acc

    def draw_answerChack(self,tate=5,yoko=5):
        print '-----AnswerCheckStart------'
        x,recog,ans,acc = self.config(tate*yoko)
        cnt=0
        plt.figure(figsize=(6,6))
        # Show Predict Result
        for idx in range(tate*yoko):
            # Forwarding for prediction
            cnt+=1
            size = 28
            plt.subplot(tate, yoko, cnt)
            # convert from vector to 32x32 matrix
            Z = x[idx:idx+1].reshape(size,size)
            plt.imshow(Z)
            plt.axis('off')

            plt.title("ans=%d,recog=%d"\
                    %(ans[idx],recog[idx]),size=8)
            plt.tick_params(labelbottom="off")
            plt.tick_params(labelleft="off")
            print "ans:%d,predict:%d"%(ans[idx],recog[idx])
        print "acc of draw_answerChack = %f"%(acc.data)
        print

    def predict(self,n_predict=100):
        print
        P = np.random.permutation(n_predict)
        #P = np.arange(100)
        ans = self.y_train
        loss,acc,recog\
        =self.forward(self.x_train,self.y_train,False)

        recog = recog.data.argmax(axis=1)
        for idx in P[:n_predict]:
            print "ans:%d,predict:%d"%(ans[idx],recog[idx])
        print "acc of predict = %f"%(acc.data)
