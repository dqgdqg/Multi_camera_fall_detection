import os
from MainInfer import MainInfer
import time


# clean
os.system("rm FeatureMap_*")
for i in range(4):
    os.system("ssh thumm0%d 'rm -f /home/dsjxtjc/2018211026/FeatureMap_*' "%(i+2))

machinelist=['2018211026@thumm0%d:/home/dsjxtjc/2018211026'%(i+2) for i in range(4)]

for i in range(4):
    os.system("scp -q SubInfer_%d_1.py %s &" % (i+2, machinelist[i]))
    os.system("scp -q SubInfer_%d_2.py %s &" % (i+2, machinelist[i]))

for i in range(4):
    os.system( "ssh thumm0%d 'python SubInfer_%d_1.py' &" % (i+2, i+2) ) 
    os.system( "ssh thumm0%d 'python SubInfer_%d_2.py' &" % (i+2, i+2) ) 

# lost test
# time.sleep(2)
# os.system("rm FeatureMap_1.txt")

flag = 1
test_time = 1
while flag:
    iter_n = 100
    MapList = [0 for i in range(8)]
    while iter_n:
        for num in range(8):
            if MapList[num]==0:
                tmp = os.system("test -e %s" % ('FeatureMap_%d_%d')%(num+1, test_time) ) 
                if tmp == 0:
                    MapList[num] = 1 # exist
        print(MapList)
        if sum(MapList) == 8:
            break
        iter_n -= 1
    if sum(MapList) == 0:
        print('no input in %ds'%(test_time) )
        break
    MainInfer(MapList)
    time.sleep(1)
    test_time += 1
    

