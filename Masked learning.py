import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from OnehotEncode import encode
import random
from CodeModel import Codemodel
from SoftmaxClassifier import softMax
def custom_loss_L(y_true, y_pred):
    loss=tf.keras.losses.BinaryCrossentropy()
    return loss(y_true=y_true,y_pred=y_pred)



train_set=pd.read_csv('./全基因组数据集副本.csv',header=None)
print(train_set.iloc[:,1].value_counts())
train_set=train_set.sample(frac=1, random_state=42)
train_set = train_set.reset_index(drop=True)
data=train_set.iloc[:,0]
em=encode(276)

codemodel=Sequential()
for i in range(4):
    codemodel.add(Codemodel(channel=128,k=6,d=pow(2,i)))
decodemodel=Sequential()
for i in range(4):
    decodemodel.add(Codemodel(channel=32,k=6,d=pow(2,i)))

classifer=softMax(classes=4)

simple_encoder_decoder = Sequential([codemodel,decodemodel])


def selectIndex(rate):
    indexList=[]
    rate=int(276*rate)
    for i in range(rate):
        while(True):
            index=random.randint(0,275)
            if index not in indexList:
                indexList.append(index)
                break
    return indexList


def DNAmask(sequence,indexList):
    Base=[[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    for j in range(int(len(indexList)*0.8)):
        index=random.randint(0,len(indexList)-1)
        sequence[:,indexList[index]]=Base[0]
        del indexList[index]
    for k in range(int(len(indexList)/2)):
        index=random.randint(0,len(indexList)-1)
        sequence[:,indexList[index]]=Base[random.randint(1,3)]
        del indexList[index]
    return sequence

train_loss = tf.keras.metrics.Mean(name='train_loss')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)
for epoch in range(50):
    print('epoch',epoch)
    f=0
    while(f<len(data)):
        with tf.GradientTape() as tape:
            with tf.device('/GPU:0'):
                Tdata=em.onehot(data[f:f+512])
                loss=0
                index=selectIndex(0.4)
                mask=DNAmask(Tdata.copy(),index.copy())
                label = Tdata[:,index]
                mask=simple_encoder_decoder(mask,training=True)
                mask=tf.gather(mask,index,axis=1)
                pre=classifer(mask,training=True)
                loss+=custom_loss_L(y_pred=pre,y_true=label)
                f+=512
                trainable_variables=simple_encoder_decoder.trainable_variables+classifer.trainable_variables
                gradients = tape.gradient(loss, trainable_variables)
                optimizer.apply_gradients(zip(gradients, trainable_variables))
                train_loss(loss)
                print(train_loss.result())

    codemodel.save_weights('./TCN6.h5')


