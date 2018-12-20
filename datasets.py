import os
import cv2
import numpy as np
from get_di import get_di

test_chute_list = [
    'chute13', 'chute14', 'chute15', 'chute16', 'chute17'
]

datasets_path = '/data3/dingqianggang/Big_Data/local/datasets/Multi_Camera_Fall_Detection'
dynamic_image_path = '/data3/dingqianggang/Big_Data/local/datasets/Dynamic_Image'
label_csv_path = '/data3/dingqianggang/Big_Data/local/datasets/label.csv'
label = {}

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    frames_num = int(cap.get(7))
    
    video_array = np.zeros((frames_num, height, width, 3), dtype='float32')
    while True:
        _, b = cap.read()
        if cap.get(1) == frames_num:
            break
            
        video_array[int(cap.get(1))] = b
        
    return video_array
    
def read_label_csv(label_csv_path):
    fr = open(label_csv_path, 'r')
    for line in fr.readlines():
        line = line.strip()
        chute_name = line.split(',')[0].split('/')[0]
        start_time = int(line.split(',')[1])
        if (chute_name in label.keys()) == True:
            if (start_time in label) == True:
                continue
            label[chute_name].append(start_time)
        else:
            label[chute_name] = [start_time]
        
if __name__ == '__main__':
    read_label_csv(label_csv_path)
    print(label)
    chute_list = os.listdir(datasets_path)
    for chute_name in chute_list:
        chute_path = os.path.join(datasets_path, chute_name)
        video_list = os.listdir(chute_path)
        for video_file_name in video_list:
            video_path = os.path.join(chute_path, video_file_name)
            cam_name = video_file_name.split('-')[0]
            from_time = int(video_file_name.split('-')[1].strip('s.avi'))
            to_time = int(video_file_name.split('-')[2].strip('s.avi'))
            video_array = read_video(video_path)
            di = get_di(video_array)
            
            if (chute_name in label.keys()) == True:
                for start_time in label[chute_name]:
                    if start_time >= from_time and start_time < to_time:
                        label_name = 'fall'
                    else:
                        label_name = 'other'
            else:
                label_name = 'other'
            
            if (chute_name in test_chute_list) == False:
                to_path = os.path.join(dynamic_image_path, cam_name, 'train', label_name)
            else:
                print(chute_name)
                to_path = os.path.join(dynamic_image_path, cam_name, 'test', label_name)
            
            if os.path.exists(to_path) == False:
                os.makedirs(to_path)
                
            di_file_name = chute_name + '_' + video_file_name.strip('.avi') + '.png'
            di_path = os.path.join(to_path, di_file_name)
            cv2.imwrite(di_path, di)
            print(di_path)
    