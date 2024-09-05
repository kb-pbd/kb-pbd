import keras
import keras.layers as layers
import tensorflow as tf
import tensorflow_addons as tfa
from config import *
from dataloader.lstm_loader import GraspRecognitionDataLoader
from model.lstm_model import GraspRecognitionModel
from trainer.lstm_trainer import GraspRecognitionTrainer
from evaluation import test_accuracy
import logging

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    data_loader=GraspRecognitionDataLoader(DATASET_PATH,class_num,test_size)
    data_loader.load_data()
    #data_loader.trim_data()
    #data_loader.subtract_wrist()
    train_data,test_data,train_label,test_label = data_loader.train_test_shuffle()
    # print(train_data.shape) 
    # (276, 50, 63)
    logger.info('Create the model.')
    lstm = GraspRecognitionModel(model_input_length,initial_learning_rate=0.001,decay_steps=100,decay_rate=0.9,staircase=True)

    logger.info('Create the trainer.')
    
    trainer = GraspRecognitionTrainer(lstm.model,model_path,model_name,train_data,train_label,batch_size=16,epochs=300, validation_split=0.25,shuffle= True,Flag_summary=True,callbacks=None)
    
    logger.info('Start training the model.')
    
    history = trainer.train()
    trainer.visualization(history)

    test_accuracy(test_data,test_label,trainer.model,class_num)



if __name__ == '__main__':
    main()
