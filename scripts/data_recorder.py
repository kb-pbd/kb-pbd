#!/usr/bin/env python3
import rospy
from grasp_recognition.msg import normalized_hands_landmarks
from grasp_recognition.srv import DataRecorderControl
import sys
import os
import time

class DataRecorderNode:
    def __init__(self):
        #ROS subscriber
        self.sub_hands_landmarks = rospy.Subscriber('/keypoints', normalized_hands_landmarks, self.record_data)

        self.normalized_hand_landmarks = None

        #Part: ros param
        rospy.set_param('/data_recorder_status', False)
        self.data_recorder_status = rospy.get_param('/data_recorder_status')

        #Part: ros service
        self.data_recorder_control_srv = rospy.Service('data_recorder_control', DataRecorderControl, self.start_stop_recorder)

        self.count_data_num = 0

    def make_dir(self,dataset_folder_path,label_name):
        self.data_folder_path = os.path.join(dataset_folder_path,label_name)

        if not os.path.exists(self.data_folder_path):
            self.data_folder_path = os.path.join(dataset_folder_path,label_name)
            os.mkdir(self.data_folder_path)
    
    def stop_recorder(self):
        rospy.set_param('/data_recorder_status', False)
        self.file_data.close
        self.file_label.close
        self.file_timestamp.close
        rospy.loginfo('data recorder stoped, files closed')

        self.count_data_num = 0

    def start_stop_recorder(self, srv_args):
        
        if srv_args.start_stop_recorder:
            self.make_dir(srv_args.dataset_folder_path, srv_args.label_name)
            
            self.file_data_path = os.path.join(self.data_folder_path,"data.txt")
            self.file_label_path= os.path.join(self.data_folder_path,"label.txt")
            self.file_timestamp_path = os.path.join(self.data_folder_path,"timestamp.txt")

            self.file_data = open(self.file_data_path,"a")
            self.file_label = open(self.file_label_path,"a")
            self.file_timestamp = open(self.file_timestamp_path,"a")

            self.label_index = srv_args.label_index
            rospy.set_param('/data_recorder_status', srv_args.start_stop_recorder)
            rospy.loginfo('data recorder started, files opened')
        
        else:
            #keep files open until service called with stop
            self.stop_recorder()

        return True
    
    def record_data(self, msg_hands_landmarks):

        self.data_recorder_status = rospy.get_param('/data_recorder_status')

        if self.data_recorder_status:

            time_stamp = int(time.time())
            
            flag_hands_existence = False
            self.normalized_hand_landmarks = None

            #priority use right hand landmarks
            if msg_hands_landmarks.normalized_hand_landmarks_right is not None:
                self.normalized_hand_landmarks = list(msg_hands_landmarks.normalized_hand_landmarks_right)
                flag_hands_existence = True
                rospy.loginfo('right hand landmarks found')
            #in case normalized_hand_landmarks_left has value, means that either left hand accidently recorded or right hand was mis-disdinguished 
            if msg_hands_landmarks.normalized_hand_landmarks_left != (0.0,):
                rospy.logwarn("WRONG HAND INSIDE! Please record again")
                

            if flag_hands_existence:
                if not all(element == 0 for element in self.normalized_hand_landmarks):
                    for i in range(len(self.normalized_hand_landmarks)):
                        self.file_data.write(f'{self.normalized_hand_landmarks[i]} ')
                    self.file_data.write("\n")
                    self.file_label.write(f'{self.label_index}')
                    self.file_label.write('\n')
                    self.file_timestamp.write(f'{time_stamp}')
                    self.file_timestamp.write('\n')
                    self.count_data_num += 1
                    rospy.loginfo('frame recorded')
   
            else:
                print("No hands input in this frame")
            
            if self.count_data_num > 300:
                rospy.logerr("number of data > 300 !!")
                self.stop_recorder() 
                
if __name__ == '__main__':
    rospy.init_node('data_recorder_node')
    data_recorder_node = DataRecorderNode()
    rospy.spin()