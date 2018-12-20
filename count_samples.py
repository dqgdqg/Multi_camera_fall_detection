import os

datasets_path = '/data3/dingqianggang/Big_Data/local/datasets/Multi_Camera_Fall_Detection'

for root, dirs, files in os.walk(datasets_path):
    for name in files:
        print(os.path.join(root, name))
    for name in dirs:
        print(os.path.join(root, name))