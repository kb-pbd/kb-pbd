import os
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, TensorBoard

class GraspRecognitionTrainer:
    def __init__(self,model,model_path,model_name,train_data,train_label,batch_size=16,epochs=800, validation_split=0.25,shuffle= True,Flag_summary=True,callbacks=None):
        self.model = model
        self.model_path = model_path # path in which the model file will be stored.
        self.model_name = model_name # the name of model
        self.train_data = train_data
        self.train_label = train_label
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.Flag_summary = Flag_summary
        self.callbacks= callbacks


    def train(self):
        #print('this step')
        # Error this sentence
        history = self.model.fit(x=self.train_data,y=self.train_label,batch_size=self.batch_size,
        epochs=self.epochs, validation_split=self.validation_split,shuffle=self.shuffle,callbacks=self.callbacks)
        os.chdir(self.model_path)
        self.model.save(self.model_name)
        
        if self.Flag_summary==True:
            self.model.summary()
        return history
    
    def visualization(self,history):
        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1,len(loss_values)+1)
        plt.subplot(2,1,1)
        plt.plot(epochs, loss_values, 'b',label='Training Loss')
        plt.plot(epochs, val_loss_values, 'r',label='Validation Loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        #plt.savefig('loss.png')
        acc_values = history_dict['accuracy']
        val_acc_values = history_dict['val_accuracy']
        plt.subplot(2,1,2)
        plt.plot(epochs, acc_values, 'b',label='Training Acc')
        plt.plot(epochs, val_acc_values, 'r',label='Validation Acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accurary')
        plt.legend()
        plt.show()
