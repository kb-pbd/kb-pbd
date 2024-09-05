'''configuration parameters such as batch size, and the number of epochs
'''
import os
#---------------------------------------------------------------------
# Data Processing
# Dataset Format Alignment
#---------------------------------------------------------------------
# dataset parameter
DATASET_PATH = '../dataset'
# note that how long it takes from one video to the next video.
time_interval = 5
model_input_length = 50 # the length of sequences per action
hidden_layer_num = 8
class_num = 5 # four classes of label: not holding,grasping, releasing and holding
test_size = 0.1 # Proportion of dataset, which is used as test-dataset


#---------------------------------------------------------------------
# Data Training
#---------------------------------------------------------------------
# path, which is used to store the model
model_path = ''
# name of stored model 
model_name = 'grasp_recognition_lstm.h5'


#---------------------------------------------------------------------
# RUN
#---------------------------------------------------------------------
# temporary array utilizied for storing the coordinates, which will be once putted into
# trunc_frame_array, then this array will be tranported into input_array as the input of model

# temporary array------>trunc_frame_array------>input_array

# temporary array : (num_tmp_video_coord,63)
# trunc_frame_array(length of truncated frames): (1,num_trunc_frame,63)
# input_array: (1,parse_param,63)

# For instance:
'''
The input format of the model shall be (1,parse_param,63), whereas we could fill only some
pair array. For example, parse_param based on the process of dataset-alignment is 50, we could
only let 'number of truncated frames' fill with this input array. In this case,number of truncated 
frames is 30,but optional. Now (1,30,63) of input_array is with coordinates daten. For the rest 
(1,50-30,63) we could set the value 0 up, which will be regarded as Masked  in lstm model. 
Noted: The lstm model will not take This '0' Part into consideration.

As for temporary array of tmp_video_coord, its function is to store the coordinates of hand
with length of specific frames. For example in this case 'num_tmp_video_coord' is 20 (optional
but no more than 'num_trunc_frame_array'), the temporary array will replace the latter part of 
'trunc_frame_array', after 20 frame-data is in temporary array collected.
'''

camera_type = "realsense"
num_tmp_video_coord = 20
length_timeserie_frames = 30

threshold_time_serie_buffer_size = 5

processing_method = 'pre_2samelabel'

flag_print_processing_time = True
flag_show_landmarks = True
