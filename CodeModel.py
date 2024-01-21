import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling1D, Flatten, Dense,BatchNormalization,GlobalAveragePooling2D,Dropout,Bidirectional,Embedding,Conv1D,Activation
from tensorflow.keras import Model
class Codemodel(Model):
    def __init__(self,channel,k,d):
        super(Codemodel, self).__init__()
        self.conv1 = Conv1D(filters=channel, kernel_size=k,padding='same',dilation_rate=d)
        self.conv2 = Conv1D(filters=channel, kernel_size=k,padding='same',dilation_rate=d)
        self.conv1_1 = Conv1D(filters=channel*2, kernel_size=1, activation='relu',padding='same')
       
        self.bn1=BatchNormalization()
        self.bn2=BatchNormalization()

        self.relu1=Activation('relu')
        self.relu2=Activation('relu')


    def call(self, x):
        xr=tf.reverse(x,axis=[1])

        f=tf.identity(x)

        x = self.conv1(x)
        
        x=self.bn1(x)

        x=self.relu1(x)
        
        x = self.conv2(x)
        
        x=self.bn2(x)

        x=self.relu2(x)

        xr = self.conv1(xr)
        
        xr=self.bn1(xr)

        xr=self.relu1(xr)
        
        xr = self.conv2(xr)
        
        xr=self.bn2(xr)

        xr=self.relu2(xr)

        xr=tf.reverse(x,axis=[1])

        x=tf.concat((x,xr),axis=-1)

        f=self.conv1_1(f)

        return x+f
        


