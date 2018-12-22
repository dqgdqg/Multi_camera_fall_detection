import os
import time

n = 1
while n<6:
    item = 'FeatureMap_5_%d'%(n)

    with open(item, 'w') as f:
        f.write('i am %s\n'%(item))

    os.system( "scp %s 2018211026@thumm01:/home/dsjxtjc/2018211026/" % (item) )
    time.sleep(1)
    n += 1