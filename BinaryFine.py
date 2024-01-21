import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from OnehotEncode import encode
import random
from CodeModel import Codemodel
from ModeClassifier import FLinemodel
from tensorflow import keras
import numpy as np


import random
def cutDna(sequence):
    sequence=sequence.split()
    length=random.randint(50,276)
    begin=random.randint(0,len(sequence)-length)
    sequence=sequence[begin:begin+length]
    return ' '.join(sequence)


train_set=pd.read_csv('./Complete Genome.csv',header=None)
Ntrain_set=pd.read_csv('./Negative Genome.csv',header=None)
Ntrain_set.iloc[:,1]=4
test_set=pd.read_csv('./Test.csv',header=None)
Alltest_set=test_set[test_set.iloc[:,1]==4].copy()




print(test_set.iloc[:,1].value_counts())
print(Alltest_set.iloc[:,1].value_counts())
print(len(test_set))
print(train_set.iloc[:,1].value_counts())

train_set=train_set.sample(frac=1, random_state=42)
train_set = train_set.reset_index(drop=True)
test_set = test_set.reset_index(drop=True)
Alltest_set = Alltest_set.reset_index(drop=True)
classes=2
encoder=encode(276)





optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9)
codemodel=Sequential()
for i in range(4):
    codemodel.add(Codemodel(channel=128,k=6,d=pow(2,i)))

import os
dummy_input = tf.zeros((32, 276 ,4)) 
_ = codemodel(dummy_input,training=False)
if os.path.exists('./TCN6.h5'):
    codemodel.load_weights('./TCN6.h5')

SLmodel=Sequential()
Smodel=Sequential()
for i in range(4):
    Smodel.add(Codemodel(channel=32,k=6,d=pow(2,i)))

optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9)
SLmodel.add(Smodel)
SLmodel.add(FLinemodel(classes=classes,seqlen=276,hidden_size=64))
SLmodel.compile(optimizer=optimizer2,
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

def Test(Testdata):
    Testlabel=Testdata.iloc[:,1]
    Testlabel=np.where(Testlabel==4,1,0)
    ans=[]
    for i in range(0,len(Testdata),5000):
        TestdataCopy=encoder.onehot(Testdata[i:i+5000].iloc[:,0])
        TestdataCopy=codemodel(TestdataCopy,training=False)
        prediction=SLmodel(TestdataCopy,training=False)
        for pre in prediction:
            ans.append(pre)
    ans=np.array(ans)
    
    error=0
    All=[0]*2
    for f in Testlabel:
        All[f]+=1
    res=[0]*2
    for i in range(len(ans)):
        if np.argmax(ans[i])!=Testlabel[i]:
            res[Testlabel[i]]+=1
            error+=1
    print("准确率",1-error/len(Testdata))
    rcall=[]
    for f in range(len(res)):
        if All[f]!=0:
           rcall.append(1-res[f]/All[f])
    print("召回率",rcall)

def Fine(Tdata,Ndata):
        Tdata=pd.concat((Tdata,Ndata),axis=0)
        cut=Tdata.copy()
        for i in range(len(cut)):
            cut.iloc[i,0]=cutDna(cut.iloc[i,0])
        labels=Tdata.iloc[:,1]
        labels=np.where(labels==4,1,0)
        labels=keras.utils.to_categorical(labels,num_classes=2)
        cut=encoder.onehot(cut.iloc[:,0])
        cut=codemodel(cut,training=False)
        SLmodel.fit(cut,labels,epochs=1,batch_size=2048)
def Fin():
        for epoch in range(3):
                print("第",epoch,"epoch")
                j=0
                for i in range(0,len(train_set),1024):
                    Fine(train_set[i:i+1024],Ntrain_set[j:j+1024])
                    j+=1024
                    if j>=len(Ntrain_set):
                         j=0
                SLmodel.save_weights('./inandOut.h5')

Fin()