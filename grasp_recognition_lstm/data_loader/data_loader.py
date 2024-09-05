import numpy as np
import sys
import os
from config import * 
import collections

parent_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(parent_dir)

sys.path.append(grandparent_dir)

from config import *

class GraspRecognitionDataLoader():
    def __init__(self, data_path,class_num,test_size):
        self.data_path = data_path
        self.class_num =int(class_num)
        self.test_size = float(test_size)
        self.data = []
        self.label = []
   
    def load_data(self):
        # sort the label num of grasp type
        grasp_type_dict = collections.defaultdict(int)
        for grasp_type_repo in os.listdir(self.data_path):
            grasp_type_repo_path = os.path.join(self.data_path, grasp_type_repo)
            for txt in os.listdir(grasp_type_repo_path):
                txt_path = os.path.join(grasp_type_repo_path,txt)
                if txt_path.endswith("label.txt"):
                    single_grasp_type_label = np.loadtxt(txt_path).tolist()[0]
                    grasp_type_dict[grasp_type_repo] = int(single_grasp_type_label)
        grasp_type_list_sorted = sorted(grasp_type_dict.items(), key = lambda kv:(kv[1], kv[0]))

        for grasp_type_repo in os.listdir(self.data_path):
            grasp_type_repo_path = os.path.join(self.data_path, grasp_type_repo)
            for txt in os.listdir(grasp_type_repo_path):
                txt_path = os.path.join(grasp_type_repo_path,txt)
                if txt_path.endswith("data.txt"):
                    single_grasp_type_data = np.loadtxt(txt_path).tolist()
                    if len(single_grasp_type_data) % model_input_length != 0:
                        print("The length of a single data sequence is not a multiple of 50")
                    trim_count = len(single_grasp_type_data) // model_input_length
                    label_tmp = [0] * self.class_num
                    for i, (grasp_type_name,_) in enumerate(grasp_type_list_sorted):
                        if grasp_type_name == grasp_type_repo:
                            label_tmp[int(i)] = 1
                            break
                    for i in range(trim_count):
                        single_sequence = single_grasp_type_data[i*model_input_length:(i+1)*model_input_length]
                        self.data.append(single_sequence)
                        self.label.append(label_tmp)
    
    def train_test_shuffle(self):
        # transform self.data to np array format.
        video_num = len(self.data)
        tmp_data = np.zeros((video_num,model_input_length,63))
        tmp_label = np.zeros((video_num,class_num))
        for i in range(video_num):
            for j in range(model_input_length):
                tmp_data[i,j,:] = self.data[i][j][:]
            tmp_label[i,:] =self.label[i][:]
        self.data = tmp_data
        self.label = tmp_label
        print(self.data.shape)
        len_train_data = int(self.data.shape[0]*(1-self.test_size))
        shuffler = np.random.permutation(self.data.shape[0])
        data_shuffle = self.data[shuffler,:,:]
        label_shuffle = self.label[shuffler,:]

        train_data = data_shuffle[0:len_train_data,:,:]
        test_data = data_shuffle[len_train_data:,:,:]

        train_label = label_shuffle[0:len_train_data,:]
        test_label = label_shuffle[len_train_data:,:]

        return train_data,test_data,train_label,test_label