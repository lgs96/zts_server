import os
import sys
import socket
import threading
from io import BytesIO
import time
import re
import numpy as np
import logging
import pickle
import struct

from numpy.core.fromnumeric import std

def CreateLogger(name, fmt="[%(asctime)s]%(name)s<%(levelname)s>%(message)s",
              terminator='\n', level = logging.INFO):
    logger = logging.getLogger(name)
    
    # Check handler exists
    if len(logger.handlers) > 0:
        return logger
    
    cHandle = logging.StreamHandler()
    cHandle.terminator = terminator
    cHandle.setLevel(level)
    cHandle.setFormatter(logging.Formatter(fmt=fmt))
    logger.addHandler(cHandle)
    logger.setLevel(level)
    return logger  

def info (logger, *args):
    logger.info("".join(str(item) for item in args))
    
def debug (logger, *args):
    logger.debug("".join(str(item) for item in args))

logger = CreateLogger('NEW', terminator = '\n')
rlogger = CreateLogger('REPLACE', terminator = '\r')

def start_tcpdump(sname, iname, hname, port):
    cmd = 'tcpdump -ttt -w data/trace_%s.pcap -i %s dst %s and dst port %d &' % (sname, iname, hname, port)
    # cmd = str(cmd)
    # print(cmd)
    # cmd = 'tcpdump -ttt -w data/trace_%s.pcap &' % ('hello')
    sudoPassword = 'glaakstp'
    #print('echo %s|sudo -S %s' % (sudoPassword, cmd))
    #p = os.system('sudo -S %s' % (cmd))

def stop_tcpdump():
    os.system("kill -9 `ps -aux | grep tcpdump | awk '{print $2}'`")
    
def process_uplink(c, addr, tx_start, object_size, object_interval, object_number):
    
    recv_size = 0
    start = time.time()
    device_first = -1
    object_first = -1
    object_last = -1
    
    # statistics
    number_of_objects = 0
    first_to_last_list = []
    object_latency_list = []
    first_object_latency = 0
    mean_object_latency = 0
    std_object_latency = 0
    
    info(logger,'====================')
    
    while True:
        data = c.recv(8192)
        #print('uplink data: ', ('done' in str(data[0:4])))
        recv_size = recv_size + len(data)
        info(rlogger,'process_uplink ', (recv_size),' uploaded objects: ',  number_of_objects)
        #print('data: ',data)
        
        if ('o_last' in str(data)):
            # Notify completion
            number_of_objects += 1
            c.send(bytes('o_recv', encoding = 'ascii'))
            # If received one is the first object
            if object_last == -1:
                device_first = tx_start
            
            object_last = time.time()
            first_to_last = round((object_last-object_first)*1e3)
            upload_time = round(object_last*1e3-device_first)
            first_to_last_list.append(first_to_last)
            object_latency_list.append(upload_time)
            
            debug(logger, 'Object is uploaded, time difference between first to last byte is %dms and upload completion time is %dms',first_to_last,upload_time)
            #print(object_first, object_last)
                
                
        if ('o_first' in str(data)):
            idx = str(data).index('o_first')
            object_first= time.time()
            device_first = data.decode('ascii').strip('(').strip('$').strip('\x00')
            device_first = (int)(re.sub(r'[^0-9]', '', device_first))
            logger.debug('Object upload is started')     
        if ('u_done' in str(data)):
            logger.debug("All objects are uploaded")
            break
        #if not data:
        #    break
    end = time.time()
    
    start_index = 0
    if number_of_objects != 1:
        recv_size = recv_size*(number_of_objects-1)/number_of_objects
        start_index = 1
    info(rlogger,'\n')
    info(logger,'Disconnected', addr, '\n')
    
    c.close()
    #stop_tcpdump()
    object_size_arr = np.ones(len(object_latency_list[:]))*((int)(object_size))*8*1024
    throughput_arr = object_size_arr/(np.array(object_latency_list[:])*1e3)
    
    init_latency_arr = np.array(object_latency_list[:]) - np.array(first_to_last_list[:])
    
    mean_throughput = np.mean(throughput_arr[start_index:])
    std_throughput = np.std(throughput_arr[start_index:])
    
    mean_init_latency = np.mean(init_latency_arr[start_index:])
    std_init_latency = np.std(init_latency_arr[start_index:])
    
    ## save the log
    key = str(object_size)+'_'+str(object_interval)
    val = {'object num': number_of_objects, 'first tx time': object_latency_list[0], 'first init time': round(object_latency_list[0] - first_to_last_list[0],2) ,'mean throughput': mean_throughput, 
           'std throughput': std_throughput, 'mean upload time': round(np.mean(object_latency_list[start_index:]),2),
            'std upload time': round(np.std(object_latency_list[start_index:]),2), 'mean initial latency': mean_init_latency, 'std initial latency': std_init_latency}
    
    try:
        with open('uplink_data/uplink.pickle', 'rb') as fr:
            loaded_uplink = pickle.load(fr)
    except:
        loaded_uplink = {}
        
    loaded_uplink[key] = val 
    '''
    if key in loaded_uplink:
            loaded_uplink[key].append(val)
    else:
        loaded_uplink[key] = []
        loaded_uplink[key].append(val)
    '''
    with open('uplink_data/uplink.pickle', 'wb') as fw:
        pickle.dump(loaded_uplink, fw)
    
    info(logger,'====================')
    info(logger,'Total received size: ', round(recv_size/(1024*1024),2), 'Mbytes ', ' first throughput: ', throughput_arr[0], 'ms  mean throughput: ', round((recv_size*8)/((np.sum(object_latency_list[start_index:])*1e3)),2), 'Mbps')
    info(logger, 'First throughput: ', throughput_arr[0], 'Mbps First init latency: ', init_latency_arr[0], 'ms')
    info(logger, 'Mean/std throughput:', mean_throughput,' ',  std_throughput, 'Mbps Mean/std initial latency: ', mean_init_latency, ' ',std_init_latency)

def process_downlink(c, addr, object_size, object_interval, object_number):
    send_size = 0    
    start = time.time()
        
    for i in range(object_number):
        first_bytes_msg = "o_first_"+str(time.time())
        c.send(bytes(first_bytes_msg, encoding = "ascii"))
        for j in range(object_size):
            try:
                c.send(bytes(1024))
                send_size = send_size + 1024
                debug(rlogger,'tx_downlink ', (send_size),' transmitted objects: ',  i)
            except:
                logging.error(logger, "Exception while downlink transmission")
                break
        last_bytes_msg = "o_last"
        c.send(bytes(last_bytes_msg, encoding = "ascii"))
        
        # Listen object recption
        while True:
            data = c.recv(5000)
            if 'o_recv' in str(data):
                info(rlogger, "Object " ,i ," is received from client")
                break
        
        time.sleep(object_interval/1000)
    end = time.time()
    
    ## save the log (get downlink log from the mobile device)
    
    while True: 
        data = c.recv(4096)
        if 'd_fin' in str(data):
            info(logger, "Received finished message from mobile device")
            break
    
    c.close()
    
    data_str = data.decode('ascii').strip('(').strip('$').strip('\x00')
    data_str_list = data_str.split('_')
    
    first_tx_time = round((float)(data_str_list[2]),2) 
    mean_tx_time = round((float)(data_str_list[3]),2) 
    std_tx_time = round((float)(data_str_list[4]),2) 
    first_init_time = round((float)(data_str_list[5]),2) 
    mean_init_time = round((float)(data_str_list[6]),2) 
    std_init_time = round((float)(data_str_list[7]),2) 
    mean_throughput = round((float)(data_str_list[8]),2) 
    std_throughput = round((float)(data_str_list[9]),2) 

    key = str(object_size)+'_'+str(object_interval)
    val = {'object num': max(object_number-1, 1), 'first tx time': first_tx_time, 'first init time': first_init_time ,'mean throughput': mean_throughput, 
           'std throughput': std_throughput, 'mean upload time': mean_tx_time,
            'std upload time': std_tx_time, 'mean initial latency': mean_init_time, 'std initial latency': std_init_time}
    
    try:
        with open('downlink_data/downlink.pickle', 'rb') as fr:
            loaded_downlink = pickle.load(fr)
    except:
        loaded_downlink = {}
        
    loaded_downlink[key] = val
    
    with open('downlink_data/downlink.pickle', 'wb') as fw:
        pickle.dump(loaded_downlink, fw)

    info(logger,'Total send size: ', send_size, 'bytes',  ' first download time: ', first_tx_time, 'ms  mean throughput: ', mean_throughput, 'Mbps')
    info(logger,'Download time mean: ', mean_tx_time ,'ms, std: ', std_tx_time,
                ' Mean initial latency: ', mean_init_time, 'ms')
    info(logger,'Disconnected', addr)
    #stop_tcpdump()       

    
def receive_data (c, addr):
    print('Start to receive data ',  addr)
    while True:
        #receive_data(c)
        count = 0
        while True:
            try:
                data = c.recv(4)
                length = int.from_bytes(data, "little")
                print("data length", length)
                while len(data) < length:
                    temp = c.recv(4096)
                    data += temp
                    #print('Temp data: ', temp)
                count += 1
                print("Recv frame ", count, len(data))
            except:
                print("error ",  data)


if __name__ == '__main__':

    if len(sys.argv) != 3:
        logger.error('Input parameter is not valid')
        logger.error('example) python3 server.py <interface:wlan0>')
        exit()
    else:
        iname = sys.argv[1]
        pname = sys.argv[2]

    s = socket.socket()
    port = (int)(pname)
    s.bind(('', port))
    #s.setblocking(True)
    s.listen(10)
    logger.info('Start Server with port %d' % (port))
    try:
        while True:
            c, addr = s.accept()  
            print('Connected by ', addr)
            th = threading.Thread (target = receive_data, args =(c, addr))
            th.start()
    except:
        print('Exception before connection')
    #data = ""

    
        '''
        data = b""
        count = 0
        while True:
            while len (data) <  payload_size:
                try: 
                    data += c.recv(4096)
                    logger.info ("Data:  %s ", str(len(data)))
                except Exception as ex:
                    logger.info ("Exception: %s", ex)
            
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]

            logger.info ("Header info:  %s , %s ", str(payload_size), str(msg_size))

            while len(data) < msg_size:
                data += c.recv(4096)
            logger.info ("Frame received:  %s, %s", str(count) , str(len(data)))
        '''