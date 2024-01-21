import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from OnehotEncode import encode
import random
from CodeModel import Codemodel
from ModeClassifier import FLinemodel
import numpy as np
from tensorflow import keras
def cutDna(sequence):
    sequence=sequence.split()
    length=random.randint(50,len(sequence))
    begin=random.randint(0,len(sequence)-length)
    sequence=sequence[begin:begin+length]
    return ' '.join(sequence)

def Test(Testdata):
    Testlabel=Testdata.iloc[:,1]
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

train_set=pd.read_csv('./Complete Genome.csv',header=None)
print(train_set.iloc[:,1].value_counts())
train_set=train_set.sample(frac=1, random_state=42)
train_set = train_set.reset_index(drop=True)
classes=4
encoder=encode(276)
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
optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)
SLmodel.add(Smodel)
SLmodel.add(FLinemodel(classes=classes,hidden_size=64,seqlen=276))
SLmodel.compile(optimizer=optimizer2,
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
def Fine(Tdata):
        cut=Tdata.copy()
        for i in range(len(cut)):
            cut.iloc[i,0]=cutDna(cut.iloc[i,0])
        labels=Tdata.iloc[:,1]
        labels=keras.utils.to_categorical(labels,num_classes=classes)
        cut=encoder.onehot(cut.iloc[:,0])
        cut=codemodel(cut,training=False)
        SLmodel.fit(cut,labels,epochs=1,batch_size=1024)
        
def Fin():
    # 在每个训练步骤中记录损失值
        for epoch in range(20):
                print("第",epoch,"epoch")
                for i in range(0,len(train_set),1024):
                    Fine(train_set[i:i+1024])
                SLmodel.save_weights('./SLmodel.h5')
Fin()