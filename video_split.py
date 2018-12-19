import cv2
import os


def get_fragment(cap, start, end, filename):
    writer = cv2.VideoWriter(filename, 0, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int(cap.get(5)),
                             (int(cap.get(3)), int(cap.get(4))))
    while cap.get(1) < cap.get(7):
        _, img = cap.read()
        if start <= cap.get(0) / 1000 < end:
            writer.write(img)
    writer.release()


def handle_avi(avi_path, to_path, window_size=3, stride=1):
    cap = cv2.VideoCapture(avi_path)
    num_seconds = cap.get(7) / cap.get(5)
    num_subavis = int((num_seconds - window_size) / stride) + 1
    cap.release()
    for i in range(num_subavis):
        start = int(i * stride)
        end = int(i * stride + window_size)
        cap = cv2.VideoCapture(avi_path)
        file_name = avi_path.split('/')[-1].replace('.avi','')
        folder_name = avi_path.split('/')[-2]
        folder_path = os.path.join(to_path, folder_name)
        out_path = os.path.join(folder_path, file_name + "-{0}s-{1}s.avi".format(start, end))
        if os.path.exists(folder_path) ==  False:
            os.makedirs(folder_path)
        print(out_path, folder_path)
        get_fragment(cap, start, end, out_path)
        cap.release()

if __name__ == '__main__':
    datasets_path = '/data2/Fall_Detection/Multi_Camera_Fall_Detection'
    to_path = '/data3/dingqianggang/Big_Data/local/datasets/Multi_Camera_Fall_Detection'
    chute_list = os.listdir(datasets_path)
    for chute_name in chute_list:
        chute_path = os.path.join(datasets_path, chute_name)
        if os.path.isdir(chute_path) == False:
            continue
        avi_list = os.listdir(chute_path)
        for avi_file_name in avi_list:
            avi_path = os.path.join(chute_path, avi_file_name)
            handle_avi(avi_path, to_path)
        
    
    
   # handle_avi('/data2/Fall_Detection/Multi_Camera_Fall_Detection/chute01/cam1.avi', '/data3/dingqianggang/Big_Data/local/datasets/Multi_Camera_Fall_Detection')