import tensorflow as tf
from keras.layers import Dense
from tensorflow.keras import Model
class softMax(Model):
    def __init__(self,classes):
        super(softMax, self).__init__()
        self.softmax = Dense(classes,activation='softmax')
    def call(self, x):
        x=self.softmax(x)
        return x
        


