
# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold='nan')
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet,\
     optimizers,Chain, serializers
import chainer.functions  as F
import sys

def loadData(N,N_train,N_test,N_inputs,N_outputs): 
    mnist = fetch_mldata('MNIST original',\
            data_home = '/Users/okuyamatakashi/pyworks/')

    # mnist.data : 70,000件の784次元ベクトルデータ
    x = mnist.data
    
    # 教師データ
    label = mnist.target
    
    dataset = np.column_stack((x, label))
    np.random.shuffle(dataset) #データ点の順番をシャッフル

    x = dataset[:, :N_inputs]
    label = dataset[:, N_inputs:]
    label = label.ravel()

    x   = x.astype(np.float32)
    label = label.astype(np.int32)
    x  /= 255     # 0-1のデータに変換

    #データ分割（訓練とテスト）
    x_train, x_test , x_notUse = np.split(x, [N_train,N])
    #y_train, y_test , y_notUse = np.split(label, [N_train,N])    
    return x_train,x_test


def setModel(N_inpputs,unitset,N_outputs):
    model = FunctionSet(l1=F.Linear(N_inputs, unitset[0]),
                    l2=F.Linear(unitset[0], N_outputs)
                    )
    return model


def getY(x):
    x = Variable(x)
    z1 = F.relu(model.l1(x))
    z2 = F.relu(model.l2(z1))
    y  = z2
    return y
    
def forward(x_data, y_data, train=True):
    t = Variable(y_data)
    x = Variable(x_data)
    y = getY(x_data)
    # 多クラス分類なので誤差関数としてソフトマックス関数の
    # 交差エントロピー関数を用いて、誤差を導出
    return F.mean_squared_error(y, t)


def TrainTest_loop(n_epoch,batchsize,n_display,N_train,N_test,\
                   x_train,x_test,y_train,y_test):
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
        for i in xrange(0, N_train, batchsize):
            x_batch = x_train[i:i+batchsize]
            y_batch = y_train[i:i+batchsize]

            # 勾配を初期化
            optimizer.zero_grads()

            # 順伝播させて誤差と精度を算出
            #loss, acc  = forward(x_batch,y_batch)

            # 誤差のみ
            loss = forward(x_batch, y_batch, train=False)

            # 誤差逆伝播で勾配を計算
            loss.backward()
            optimizer.update()

            train_loss.append(loss.data)
            #train_acc.append(acc.data)
            sum_trainloss += loss.data * batchsize
            #sum_trainacc  += acc.data  * batchsize
        mean_trainloss = sum_trainloss / N_train
        #mean_trainacc = sum_trainacc / N_train

        # 学習したパラメーターを保存
        l1_W.append(model.l1.W)
        l2_W.append(model.l2.W)
        #l3_W.append(model.l3.W)

        '''
        テスト
        '''
        # テストデータで誤差と、正解精度を算出し汎化性能を確認
        sum_testloss = 0
        sum_testacc     = 0
        for i in xrange(0, N_test, batchsize):
            x_batch = x_test[i:i+batchsize]
            y_batch = y_test[i:i+batchsize]

            # 順伝播させて誤差と精度を算出
            #loss, acc = forward(x_batch, y_batch, train=False)

            # 誤差のみ
            loss = forward(x_batch, y_batch, train=False)

            test_loss.append(loss.data)
            #test_acc.append(acc.data)
            sum_testloss += loss.data * batchsize
            #sum_testacc  += acc.data * batchsize

        mean_testloss = sum_testloss / N_test
        #mean_testacc  = sum_testacc  / N_test

        if (epoch % (n_epoch/n_display)==0):
            print 'epoch : ', epoch
            # 訓練データの誤差と、正解精度を表示
            print 'train mean loss=%f'\
                    %(mean_trainloss)
            # テストデータの誤差と、正解精度を表示
            print 'test mean loss=%f'\
                    %(mean_testloss)
            print loss.data

def Train_loop(n_epoch, batchsize, n_display, N_train,\
               x_train,y_train):
    train_loss = []
    train_acc  = []
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
        for i in xrange(0, N_train, batchsize):
            x_batch = x_train[i:i+batchsize]
            y_batch = y_train[i:i+batchsize]

            # 勾配を初期化
            optimizer.zero_grads()

            # 順伝播させて誤差と精度を算出
            loss, acc = error(x_batch,y_batch)

            # 誤差逆伝播で勾配を計算
            loss.backward()
            optimizer.update()


            train_loss.append(loss.data)
            train_acc.append(acc.data)
            sum_trainloss += loss.data * batchsize
            sum_trainacc  += acc.data  * batchsize
        mean_trainloss = sum_trainloss / N_train
        mean_trainacc = sum_trainacc / N_train
        
        if (epoch % (n_epoch/n_display)==0):
            print 'epoch : ', epoch
            # 訓練データの誤差と、正解精度を表示
            print 'train mean loss=%f, accuracy=%f'\
                    %(mean_trainloss,mean_trainacc)

    # 学習したパラメーターを保存
    l1_W.append(model.l1.W)
    #l2_W.append(model.l2.W)
    #l3_W.append(model.l3.W)

def draw_digit(data,P,tate=10,yoko=10,cnt=0):
    # Show Predict Result
    for idx in P[:tate*yoko]:
        # Forwarding for prediction
        xxx = data[idx:idx+1]
        cnt+=1
        size = 28
        plt.subplot(tate*2, yoko, cnt)
        # convert from vector to 28x28 matrix
        Z = xxx.reshape(size,size)          
        # flip vertical
        Z = Z[::-1,:] 
        plt.xlim(0,27)
        plt.ylim(0,27)
        plt.pcolor(Z)
        #plt.title("ans=%d, recog=%d"%(ans,recog), size=8)
        plt.gray()
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")

if __name__=="__main__":
    
    N = 3000            #サンプル数
    N_train = 2000      #訓練データの数
    N_test = N - N_train #テストデータの数

    N_inputs  = 784 #入力の数
    unitset   = [64]            #中間層ユニット数
    N_outputs = N_inputs        #出力数

    # 学習の繰り返し回数
    n_epoch   = 100
    # 確率的勾配降下法で学習させる際の１回分のバッチサイズ
    batchsize = 100

    n_display = 10
    n_predict = 100

    x_train,x_test\
    = loadData(N,N_train,N_test,N_inputs,N_outputs)
        
    model = setModel(N_inputs,unitset,N_outputs)

    #Optimaizerの設定
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    train_only = True

    #Train_loop(n_epoch,batchsize,n_display,N_train,x_train,y_train)

    TrainTest_loop(n_epoch,batchsize,n_display,N_train,N_test,\
                       x_train,x_test,x_train,x_test)
    
    
    P = np.random.permutation(N_train)
    tate = 10
    yoko = 10

    encodeX = getY(x_train).data

    x = Variable(x_train)


    cnt=0
    
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(9,9))
    draw_digit(x_train,P,tate,yoko,cnt)
    cnt += tate*yoko
    draw_digit(encodeX,P,tate,yoko,cnt)
    #plt.show()
    #serializers.save_npz('AutoEncoder_model',model)
