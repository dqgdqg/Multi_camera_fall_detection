import os
import sys

datasets_path = '/data3/dingqianggang/Big_Data/local/datasets/Dynamic_Image'

for root, dirs, files in os.walk(datasets_path):
    for name in files:
        file_path = os.path.join(root, name)
        second = name.split('-')[1]
        if name[0:7] == sys.argv[1] and second == sys.argv[2]:
            os.remove(file_path)
            print(file_path)