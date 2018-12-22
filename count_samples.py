import os

datasets_path = '/data3/dingqianggang/Big_Data/local/datasets/Multi_Camera_Fall_Detection'
dynamic_image_path = '/data3/dingqianggang/Big_Data/local/datasets/Dynamic_Image'

def check_samples(fw, datasets_path):
    for root, dirs, files in os.walk(datasets_path):
        for name in files:
            fw.writelines(os.path.join(root, name) + '\n')
        # for name in dirs:
            # fw.writeline(os.path.join(root, name) + '\n')
        
fw1 = open('check_cam1.txt', 'w')
fw2 = open('check_cam7.txt', 'w')
for root, dirs, files in os.walk(datasets_path):
    cam1_train_path = os.path.join(dynamic_image_path, 'cam1', 'test')
    cam8_train_path = os.path.join(dynamic_image_path, 'cam7', 'test')
    check_samples(fw1, cam1_train_path)
    check_samples(fw2, cam8_train_path)

fr1 = open('check_cam1.txt', 'r')
fr2 = open('check_cam7.txt', 'r')

fr1_list = fr1.readlines()
fr2_list = fr2.readlines()

for i in range(len(fr1_list)):
    line1 = fr1_list[i]
    line2 = fr2_list[i]
    line2 = line2.replace('cam7', 'cam1')
    if line1 != line2:
        print(line1,line2)