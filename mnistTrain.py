# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet,\
     optimizers,Chain,serializers
import chainer.functions  as F
import sys

import ChainerNN


#np.random.seed(0)

class MnistNN(ChainerNN.NN):
    def __init__(self,N,Ntr,Nte,unitSet,NetName):
        super(Cifar10CNN,self).__init__(N,Ntr,Nte,unitSet,NetName)
        self.N = N
        self.N_train = Ntr
        self.N_test = Nte

        self.unitSet = unitSet

        self.train_test_dataLoad()
        self.set_Model_Opti()

    def train_test_dataLoad(self):
        self.x_train,self.x_test,self.y_train,self.y_test\
        =Dataload.mnistLoad(self.N,self.N_train)

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
            l1_W.append(self.model.l1.W)
            l2_W.append(self.model.l2.W)
            #l3_W.append(model.l3.W)
            if (epoch % (n_epoch/n_display)==0):
                print ('epoch : ', epoch)
                # 訓練データの誤差と、正解精度を表示
                print ('train mean loss=%f'%(mean_trainloss),\
                      'train mean acc=%f'%(mean_trainacc))
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
                    loss,acc,y = self.forward(x_batch,y_batch)

                    test_loss.append(loss.data)
                    test_acc.append(acc.data)
                    sum_testloss += loss.data * batchsize
                    sum_testacc  += acc.data * batchsize

                mean_testloss = sum_testloss / self.N_test
                mean_testacc  = sum_testacc  / self.N_test

                if (epoch % (n_epoch/n_display)==0):
                    # テストデータの誤差と、正解精度を表示
                    print ('test mean loss=%f'%(mean_testloss),\
                          'test mean acc=%f'%(mean_testacc))
                    print ('Time = %f'%(time.clock()-self.StartTime))



    def draw_digit(self,number=30):
        size = 28
        x = self.x_test
        Z = x[number:number+1].reshape(size,size)
        plt.figure(figsize=(5, 5))
        plt.imshow(Z)
        #目盛りをなくす
        plt.axis('off')
        #plt.pcolor(Z)
        plt.gray()
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        #画像右側にバーをつける
        #plt.colorbar()

    def config(self,n_config):
        #P = np.random.permutation(N_test)

        x = self.x_test[0:n_config]
        ans   = self.y_test[0:n_config]

        loss,acc,recog\
        =self.forward(x,ans,False)
        recog = recog.data.argmax(axis=1)
        return x,recog,ans,acc

    def predict(self,n_config=25):
        print ('-----PredictStart-----')
        x,recog,ans,acc  = self.config(n_config)
        for idx in range(n_config):
            print ("ans:%d,predict:%d"%(ans[idx],recog[idx]))
        print ('acc of predict = %f'%(acc.data))
        print

    def draw_answerChack(self,tate=5,yoko=5):
        print ('-----AnswerCheckStart-----')
        x,recog,ans,acc = self.config(tate*yoko)
        cnt=0
        plt.figure(figsize=(10,10))
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
            plt.gray()
            plt.title("ans=%d,recog=%d"\
                    %(ans[idx],recog[idx]),size=8)
            plt.tick_params(labelbottom="off")
            plt.tick_params(labelleft="off")
            print ("ans:%d,predict:%d"%(ans[idx],recog[idx]))
        print ("acc of draw_answerChack = %f"%(acc.data))
        print

if __name__=="__main__":
    NetName = "Mnist_Liner"

    #訓練だけなら 0 テストもするなら 1
    howloop = 1

    # 学習の繰り返し回数
    n_epoch   = 2
    # 確率的勾配降下法で学習させる際の１回分のバッチサイズ
    batchsize = 100

    n_display = 2


    N = 7000           #サンプル数
    N_train =6000     #訓練データの数
    N_test = N - N_train #テストデータの数

    unitSet   = [900]     #中間層ユニット数

    tate = 5
    yoko = 5
    n_config = tate*yoko

    Mnist =  MnistNN(N,N_train,N_test,unitSet,NetName)

    #Mnist.train_loop(howloop,n_epoch,batchsize,n_display)
    Mnist.draw_digit(3)
    #Mnist.draw_answerChack(tate,yoko)
    #Mnist.predict(n_config)
    plt.show()

    #serializers.save_npz('./modelkeep/Mnist_900unit_Model',Mnist.model)
