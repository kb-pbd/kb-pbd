import keras
import keras.layers as layers
import tensorflow as tf
import tensorflow_addons as tfa
from config import *

class GraspRecognitionModel:
    def __init__(self,seqlen_designed,initial_learning_rate=0.001,decay_steps=100,decay_rate=0.9,staircase=True):
        self.seqlen_designed = seqlen_designed # input is the designed sequnce length of video.
        self.initial_learning_rate=initial_learning_rate
        self.decay_steps=decay_steps
        self.decay_rate=decay_rate
        self.staircase=staircase
        self.model = None
        self.build_model()

    def build_model(self):
        self.model = keras.models.Sequential()
        self.model.add(layers.Masking(mask_value=0.,input_shape=(self.seqlen_designed,63)))
        self.model.add(layers.LSTM(hidden_layer_num,activation='tanh'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(class_num, activation='softmax'))

        # def decayed_learning_rate(step): 
        #   return initial_learning_rate * decay_rate ^ (step / decay_steps)
        
        opt = keras.optimizers.Adam(learning_rate=self.decayed_learning_rate(self.initial_learning_rate,
            self.decay_steps,self.decay_rate,self.staircase))
        opt = tfa.optimizers.MovingAverage(opt)
        self.model.compile(optimizer=opt,loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
    
    def decayed_learning_rate(self,initial_learning_rate,decay_steps,decay_rate,staircase):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps,
            decay_rate,
            staircase)
        return lr_schedule

