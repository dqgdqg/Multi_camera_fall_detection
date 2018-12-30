import socket
import threading
import time
import sys
import struct
import os
import numpy as np

from my_models import *
import paramiko

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense

def inference_master(model_master, arr, pos):
    print("Input is ready")

    xs = []
    for i in range(8):
        if arr[i] == 1:
            x = recv_dict['featuremap_{0}_{1}'.format(i, pos)]
        else:
            x = np.zeros((1, 56, 56, 64))
        xs.append(x)
    result = model_master.predict(xs)
    # print(result[0])

    if np.argmax(result[0]) == 1:
        print("{}s other".format(pos))
    else:
        print("{}s fall".format(pos))

def distribute_task():
    host_list = ['219.223.189.131', '219.223.190.150', '219.223.190.154', '219.223.190.155']

    # clean
    os.system("rm -f output_cache/*")
    for i in range(4):
        os.system("ssh %s 'source ~/jump/jumprc ; rm -f output_cache/* ; rm -f ~/jump/inference_slave.py' "%(host_list[i]))
        os.system("scp inference_slave.py %s:~/jump" % (host_list[i]))
        os.system("ssh %s 'source ~/jump/jumprc ; cp ~/jump/inference_slave.py ./'" % (host_list[i]))

    for i in range(4):
        os.system( "ssh %s 'source ~/jump/jumprc ; nohup python inference_slave.py %d' > /dev/null &" % (host_list[i], i*2) )
        os.system( "ssh %s 'source ~/jump/jumprc ; nohup python inference_slave.py %d' > /dev/null &" % (host_list[i], i*2+1) )

    print("Task are already distributed!")

def Predictor():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    input_master = [Input(shape=(56, 56, 64)) for _ in range(8)]
    y_master = ResNet18_top(x=input_master)
    y_master = GlobalAveragePooling2D()(y_master)
    output_master = Dense(2, activation='softmax')(y_master)
    model_master = Model(input_master, output_master)
    model_master.load_weights('./saved_model/model_master.h')
    
    print("Scanner start!...")
    time_start = time.time()
    for pos in range(10000):
        MapList = [0 for i in range(8)]
        # while time.time() - time_start < (pos + 1):
        while True:
            time.sleep(0.5)
            # print(len(recv_dict))
            for num in range(8):
                if ('featuremap_%d_%d' % (num, pos) in recv_dict.keys()) == True:
                    MapList[num] = 1
            # print(MapList)
            if sum(MapList) == 8:
                break
        if sum(MapList) == 0:
            print('no input in %ds'%(pos) )
            break
        inference_master(model_master, MapList, pos)

def socket_service():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # designated ip and port *********************************
        s.bind(('219.223.190.251', 22348))
        # ********************************************************
        s.listen()
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    print('Waiting connection...')

    # ****************
    # By zhao
    distribute_task()
    # ****************

    # scanner & predictor *********************************************
    # By zhao
    predictor = threading.Thread(target=Predictor)
    predictor.start()
    # *****************************************************************

    while 1:
        time.sleep(0.1)
        conn, addr = s.accept()
        # create a thread for per connection
        t = threading.Thread(target=deal_data, args=(conn, addr))
        t.start()
        

def recv_into(arr, source):
    view = memoryview(arr).cast('B')
    while len(view):
        nrecv = source.recv_into(view)
        view = view[nrecv:]


def deal_data(conn, addr):
    global mutex
    print('Accept new connection from {0}'.format(addr))
    conn.send('Hi, Welcome to the server!'.encode('utf-8'))
    while 1:
        fileinfo_size = struct.calcsize("128sl")
        buf = conn.recv(fileinfo_size)
        if buf:
            filename, filesize = struct.unpack("128sl",buf)
            filename = filename.decode('utf-8')
            fn = filename.strip('\00')
            if fn == 'exit':
                break
            # file storage path******************************************
            new_filename = fn
            # ***********************************************************
            # print('file new name is {0}, filesize is {1}'.format(new_filename,filesize))

            # *********************************************************
            data_recv = np.zeros(shape=(1,56,56,64),dtype=np.float32)
            # *********************************************************
            
            # after received , save as file
            # np.save(new_filename,data_recv)
            if mutex.acquire():
                recv_into(data_recv,conn)
                recv_dict[new_filename] = data_recv
                mutex.release()
            
    print('Connection from {} is closed!'.format(addr))
    conn.close()

if __name__ == '__main__':
    mutex = threading.Lock()
    recv_dict = {}
    socket_service()
