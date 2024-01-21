from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
class encode:
   def __init__(self,maxlen) -> None:
      self.tokenizer = Tokenizer(num_words=5)
      self.tokenizer.fit_on_texts(['A T C G'])
      self.maxlen=maxlen
   def onehot(self,data):
            sequences = self.tokenizer.texts_to_sequences(data)
            for i in range(len(sequences)):
               for j in range(len(sequences[i])):
                    sequences[i][j]-=1
               sequences[i]=to_categorical(sequences[i],num_classes=4)
               if len(sequences[i])<self.maxlen:
                    zero=np.zeros((self.maxlen-len(sequences[i]),4))
                    sequences[i]=np.concatenate((sequences[i],zero),axis=0)
            return np.array(sequences)
