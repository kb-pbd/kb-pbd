from config import *
import numpy as np
import os
import sys

class DataParser():    
    def read_raw_data_lable_timestamp(self):
        for rroot, ddirs, ffiles in os.walk(self.raw_data_path):
            for name in ffiles: 
                if name.endswith("data.txt"):
                    self.raw_data = np.loadtxt(rroot+'/'+name)
                elif name.endswith("label.txt"):
                    self.raw_label = np.loadtxt(rroot+'/'+name)
                elif name.endswith("timestamp.txt"):
                    self.timestamp = np.loadtxt(rroot+'/'+name)
    
    def open_folder(self):
        folder = os.path.exists(self.parsed_data_path)
        if not folder:
            os.mkdir(self.parsed_data_path)
        os.chdir(self.parsed_data_path)
    
    def write_single_frame(self, i, parsed_data_file, parsed_label_file):
        for j in range(self.raw_data.shape[1]):
            parsed_data_file.write(f'{self.raw_data[i,j]} ')
        parsed_data_file.write('\n')
        parsed_label_file.write(f'{int(self.raw_label[i])} ')
        parsed_label_file.write('\n')

    def parse_by_time_interval(self):
        sequence_count = 0
        #raw_data.shape = [n, 63]
        for i in range(self.raw_data.shape[0]):
            if i == 0:
                parsed_data_file = open('{}_data.txt'.format(sequence_count),'a')
                parsed_label_file = open('{}_label.txt'.format(sequence_count),'a')
            else:
                #write into new file when timestamp interval > parse_param
                if (self.timestamp[i]-self.timestamp[i-1]) > self.parse_param:
                    sequence_count += 1
                    parsed_data_file = open('{}_data.txt'.format(sequence_count),'a')
                    parsed_label_file = open('{}_label.txt'.format(sequence_count),'a')
                
            self.write_single_frame(i, parsed_data_file, parsed_label_file)
        
        parsed_data_file.close()
        parsed_label_file.close()
        print('Raw data parsed into', sequence_count)
    
    def parse_by_frame_count(self):
        sequence_count = 0
        #raw_data.shape = [n, 63]
        for i in range(self.raw_data.shape[0]):
            if i == 0:
                parsed_data_file = open('{}_data.txt'.format(sequence_count),'a')
                parsed_label_file = open('{}_label.txt'.format(sequence_count),'a')
                frame_count = 1
            else:
                #write into new file when single sequence has frames > parse_param
                if frame_count > (self.parse_param-1):
                    sequence_count += 1
                    parsed_data_file = open('{}_data.txt'.format(sequence_count),'a')
                    parsed_label_file = open('{}_label.txt'.format(sequence_count),'a')
                    frame_count = 0

                frame_count += 1
            
            self.write_single_frame(i, parsed_data_file, parsed_label_file)
        
        parsed_data_file.close()
        parsed_label_file.close()
        print('Raw data parsed into', sequence_count)
    
    def parse_write_data(self):
        self.open_folder()
        if self.parse_method == 'time_interval':
            self.parse_by_time_interval()
        elif 'frame_count':
            self.parse_by_frame_count()

    def __init__(self, raw_data_path, parse_method, parse_param, parsed_data_path):
        self.raw_data_path = raw_data_path
        self.parse_param = float(parse_param)
        self.parsed_data_path = parsed_data_path
        self.parse_method = parse_method

        self.raw_data = {}
        self.raw_label = {}
        self.timestamp = {}
    
        print('parsing dataset in', raw_data_path, 'with method', parse_method, 'with param', parse_param)
        print('storing data into', parsed_data_path)

        self.read_raw_data_lable_timestamp()

if __name__ == '__main__':

    raw_data_path = sys.argv[1]
    parse_method = sys.argv[2]
    parse_param  = sys.argv[3]
    parsed_data_path  = sys.argv[4]

    parser = DataParser(raw_data_path, parse_method, parse_param, parsed_data_path)
    parser.parse_write_data()