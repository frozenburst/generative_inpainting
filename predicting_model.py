# Reference to https://stackoverflow.com/questions/59970935/keras-predict-in-a-process-freezes

from tensorflow import keras

from multiprocessing.managers import BaseManager
from multiprocessing import Lock

class KerasModelForThreads():
    def __init__(self):
        self.lock = Lock()
        self.model = None

    def load_model_and_predict(self, model_pth, x):
        self.model = keras.models.load_model(model_pth)
        #with self.lock:
        return self.model.predict(x, steps=1)

    def predict_output(self, x_pred):
        with self.lock:
            return self.model.predict(x_pred, steps=1)
            #return (self.model.predict(x_pred) + self.const.offset)[0][0]

    def summary(self):
        return self.model.summary()

class KerasManager(BaseManager):
    pass

KerasManager.register('KerasModelForThreads', KerasModelForThreads)
