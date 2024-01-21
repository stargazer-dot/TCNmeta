import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout,Bidirectional,Embedding,LSTM,GlobalAveragePooling2D,Activation,GlobalAveragePooling1D,GlobalMaxPooling1D
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM
class FLinemodel(Model):
    def __init__(self,classes,seqlen,hidden_size):
        super(FLinemodel, self).__init__()
        
        self.gp=GlobalAveragePooling1D()

        self.softmax=Dense(classes,activation='softmax')
        self.softmaxAt=Dense(seqlen,activation='softmax')


        self.d1=Dense(64,activation='relu')

        self.wq = tf.Variable(tf.random.normal([hidden_size,4], stddev=0.1) ,trainable=True)
        self.wk = tf.Variable(tf.random.normal([hidden_size,4], stddev=0.1) ,trainable=True)
        self.wv = tf.Variable(tf.random.normal([hidden_size,4], stddev=0.1) ,trainable=True)

    def call(self, x):

        Q=tf.matmul(x,self.wq)
        K=tf.matmul(x,self.wk)
        V=tf.matmul(x,self.wv)
        K=tf.transpose(K,[0, 2 ,1])
        QK=tf.matmul(Q,K)
        QK=self.softmaxAt(QK)

        x=tf.matmul(QK,V)

        x=self.softmax(x)

        x=self.gp(x)

        
        return x
        


