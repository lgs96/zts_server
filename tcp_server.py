import argparse
import time
from pathlib import Path
import csv
import cv2
#import torch
import sys
#import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
import io
import socket
import struct
import pickle
import zlib
'''
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.plots import plot_one_box
from nvjpeg import NvJpeg
'''
import cv2
import signal
import os
import time

#from nvjpeg import NvJpeg
import struct
from queue import Queue
from threading import Thread
import threading
import pickle
import datetime
#import multiprocessing
import sys
#import ntplib
from timeit import default_timer as timer

# timeServer = 'time.bora.net'
# ntp_server = ntplib.NTPClient()
# # while (True):
# response = ntp_server.request(timeServer, version=3)

# ntp_offset = response.offset



# CLIENT='192.168.1.10'
CLIENT='192.168.1.3'
HOST='0.0.0.0'
PORT = 8889
PORT2 = 8889
cnt = 0
decode_time=0
recv_time=0
t1=time.time()
el = 0
load_model=0
client_connected=0
receive_data=0
receive_time=0
send_data_size=0
inference_time=0
lock = threading.Lock()
draw=False
record = True

trans_times = []

net = 'Stress'
video = 'beach'
bud = '1333'
win = '20'

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



def recv_queue(recv_q):
    global recv_start_f
    global draw
    print('[Thread] recv_queue() start')
    print('Socket.socket')
    count=1
    total_receive_time=0
    global client_connected
    global inference_time
    global receive_data
    global receive_time
    global recv_start_history
    global send_data_size
    global conn #kmbin added
    global trans_times

    recv_start_history = []

    s=socket.socket()
    print('Socket created')

    s.bind((HOST,PORT))
    # s.setblocking(True)
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')
    conn,addr=s.accept()
    print("connected", addr)
    temp_data = b""
    while len(temp_data) < struct.calcsize(">Q"):
        print(len(temp_data))
        temp_data += conn.recv(8192)
    
    t1 = time.time()*1000.0
    t0 = struct.unpack(">Q", temp_data[:])[0]
    print(f"[Time Sync] t0: {t0}")
    t2 = time.time()*1000.0
    
    conn.sendall(struct.pack(">f", t1))
    conn.sendall(struct.pack(">f", t2))

    temp_data = b""
    while len(temp_data) < struct.calcsize(">Q"):
        temp_data += conn.recv(4096)
    
    t3 = struct.unpack(">Q", temp_data[:])[0]
    print(f"[Time Sync] t3: {t3}")
    ntp_offset = ((t1 - t0) + (t2 - t3))/2
    print(f"[Time Sync] Time Offset: {ntp_offset}")
    
    conn.sendall((1000).to_bytes(4, byteorder='big'))
    client_connected=1

    data = b""
    payload_size = struct.calcsize(">L")

    videofile= "_".join([net,video,"bw",bud,win])
    # videofile='5g_beach'
    filename='csvdata/evaluation/cellular/'+str(videofile)+'.csv'
    if os.path.isfile(filename):
        os.remove(filename)
    
    recv_start_f=open(filename, 'w', encoding='utf-8', newline='')
    wr = csv.writer(recv_start_f)
    wr.writerow(['trans_time','data_size','bw'])
    
    while(True):
            
        num=0
        while len(data) < payload_size+struct.calcsize(">Q"):
            try:
                data += conn.recv(4096)
            except Exception as ex:
                print("exception : ", ex)
                exit_event.set()
                os._exit(0)
        ## 여기서 첫 packet 수신 후, Context-swtiching overhead 많이 발생    
        # lock.acquire()
        
        # kmbin added. write recv timestamp
        r_start_time = time.time()*1000.0

        if count%300==1:
            recv_start=time.thread_time()

        packed_msg_size = data[:payload_size]        
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        r_time = data[payload_size:payload_size+struct.calcsize(">Q")]
        r_time = struct.unpack(">Q", r_time)[0]

        # queing = r_start_time - r_time - ntp_offset
        
        data = data[payload_size+struct.calcsize(">Q"):]
        c=0
        # print(f"msg_size: {msg_size}, r_time: {r_time}")
        while len(data) < msg_size:
            # t00=time.time()
            data += conn.recv(4096)
            # t01=time.time()
            # print('recv packet : {:.2f}'.format((t01-t00)*1000))
        frame_data = data[:msg_size]
        data = data[msg_size:]
        
        # if ntp_offset > 0:
        trans_time = time.time()*1000.0 - r_time - ntp_offset
        # if trans_time < 10:
        
        # if trans_time < 1:
        #     trans_time = queing
        
        trans_times.append(trans_time)
        bw = (msg_size*8/1024/1024)/((trans_time)/1000)
        # print(f"[{count}] Queing: {queing:.1f}ms Trans: {trans_time:.1f}ms \t {(msg_size/1024):.1f}Kbytes \t Estimated BW: {bw:.1f}Mbps")
        print(f"[{count}] Trans: {trans_time:.1f}ms \t {(msg_size/1024):.1f}Kbytes \t Estimated BW: {bw:.1f}Mbps")
        wr.writerow([str(trans_time), str(msg_size), str(bw)])
        
        try:
            # frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
            frame_np = np.frombuffer(frame_data, dtype=np.uint8)
            frame = np.expand_dims(frame_np, axis=1)

            recv_q.put(frame)
            receive_time = time.thread_time()
            # print(recv_start)
            # total_receive_time=receive_time
            # lock.release()
            receive_data=sys.getsizeof(frame_data)/1024
            # print("Infer time ={:.1f} ms, RecV data : {:.1f} KB, Send data = :{:.1f} Bytes, infer_q={}, send_q={} done!".format(inference_time*1000,receive_data,send_data_size, recv_q.qsize(), pred_q.qsize()), end="\r")
        except Exception as ex:
                print("exception : ", ex)
        if exit_event.is_set():
            s.close()
            f.close()
            os._exit(0)
            break
        if count%300==0:
            total_receive_time=time.thread_time()-recv_start
            # print("Average recv time for 300 frames = {:.3f} ms".format(total_receive_time*1000/300))
            
        if pred_q.qsize()>40:
            time.sleep(2)
        if pred_q.qsize()>80:
            time.sleep(2)
        if pred_q.qsize()>100:
            time.sleep(2)    
        count=count+1
        
    

def infer_q(recv_q, pred_q):
    print("inference")
    cnt=0
    #nj=NvJpeg()
    t1=time.time()
    q_get=0
    decode_time=0
    total_infer_time=0
    global load_model
    global client_connected
    global draw
    #global receive_data
    global acc_f
    global conn #kmbin added.
    global trans_times


    t=0
    idx = 1
    fps=0
    load_model=0
    #print(q.qsize())
    #print("show", q.qsize())
    # Initialize
    weights, save_txt = opt.weights, opt.save_txt
    imgsz=3840
    set_logging()
    device = select_device(opt.device)
    
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load model
    print("loadl model")
    model = attempt_load(weights, map_location=device)  # load FP32 model ?????????
    load_model=1
    print("load model done")
    stride = int(model.stride.max())  # model stride
    print("stride={}".format(stride))
    
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    # print("imgsz={}".format(imgsz))
    #img_size=imgsz
    
    #dataset = LoadImages(""source"", img_size=imgsz, stride=stride)
    
    if half:
        model.half()  # to FP16
    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()


    while client_connected==0:
        time.sleep(1)
        print("Connecting client socket ..")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # #client_socket.settimeout(300)
    # client_socket.connect((CLIENT, 8888))
    print("Listening()")
    client_socket.bind((HOST, PORT2))
    client_socket.listen()
    conn2, addr2 = client_socket.accept()
    print("Connected to client ", addr2)
    # #client_socket.settimeout(None)
    connection = client_socket.makefile('wb')
    if draw==True:
        cv2.namedWindow('server', cv2.WINDOW_NORMAL)
    
 
    videofile= "_".join([net,video,"acc",bud,win])
    # videofile='5g_beach'
    filename='csvdata/evaluation/cellular/'+str(videofile)+'.csv'
    acc_f=open(filename, 'w', encoding='utf-8', newline='')
    wr = csv.writer(acc_f)
    wr.writerow(['frame_cnt', 'cls','conf', 'x1','y1', 'x2', 'y2'])
    
    while(True):
        if recv_q :
            lock.acquire()    
            img=recv_q.get()
            framename = 'sample2/frame_1.npy'

            im0 = np.load(framename)
            # t1 = time.time() 
            t1 = timer()
            #frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
            frame = nj.decode(img)
            # '''show frame'''
            # cv2.imshow('server', frame)
            # if cv2.waitKey(1) & 0XFF == ord('q'):
            #    break

            #t1=time.thread_time()
            img_size=frame.shape[1]
            # Padded resize
            imgsz = check_img_size(img_size, s=stride)
            frame = letterbox(frame, imgsz, stride=stride)[0]

            ### numpy to tensor
            img = frame.transpose(2,0,1)
            img=torch.from_numpy(img).to(device)
            #print(img.shape)

            ### Input data
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            ### Inference
            
            pred = model(img, augment=opt.augment)[0]

            ### Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            #t3 = time_synchronized()
            
            ## Get CPP
            
            
            th=0.5
            detected_num=0
            candi_num=0
            b1=130
            b2=373*326
            p_total=0            
            for i, det in enumerate(pred):
                if record == True:
                    gn1 = torch.tensor((im0.shape))[[1, 0, 1, 0]]
                gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]
                if len(det):
                    if record == True:
                        det1=det.clone().detach()
                        det1[:, :4] = scale_coords(img.shape[2:], det1[:, :4], im0.shape).round()

                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                    if record == True:
                        for *xyxy, conf, cls in reversed(det1):
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn1).view(-1).tolist()
                            x1,y1,x2,y2,conf1,cls1= int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), float(conf), int(cls)
                            # label = f'{names[int(cls)]} {conf:.2f}'
                            # if conf > th:
                            wr.writerow([cnt,cls1,conf1,x1,y1,x2,y2])

                    for *xyxy, conf, cls in reversed(det):
                        x1,y1,x2,y2,conf1,cls1= int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), float(conf), int(cls)
                        obj_height = y2-y1
                        obj_width = x2-x1
                        obj_size=obj_height*obj_width
                        label = f'{names[int(cls)]} {conf:.2f}'
                        if conf > th:
                            if draw==True:
                                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
                            detected_num+=1
                        else :
                            if draw==True:
                                plot_one_box(xyxy, frame, label='candi', color=[0,0,0], line_thickness=2)
                            candi_num+=1
                            if obj_size<=b1 :
                                p=1
                            elif obj_size<b2 :
                                # p=max((b1/(obj_height*obj_width))**2, 0.5)
                                p=(b1/(obj_height*obj_width))**0.5
                            else :
                                p=-1
                            # print(p)
                            p_total += p
                    # alpha = max(0,p_total/(detected_num+candi_num))
                    if candi_num==0:
                        alpha=0
                    else :
                        alpha = max(0,p_total/(candi_num))
                    # print(alpha)
                    if draw==True:
                        str_fps = '[+'+str(cnt)+'] '+' fps : '+str(fps)+ ' D : '+str(detected_num)+' C : '+str(candi_num)+' a : '+str(alpha)
                        cv2.putText(frame,str(str_fps),(10,150),cv2.FONT_HERSHEY_SIMPLEX, 1, (2,2,240), 2)
                        cv2.imshow('server', frame)
                        if cv2.waitKey(1) & 0XFF == ord('q'):
                            break

            if cnt%10==0:
                fps=int(10/(time.time()-t))
                t=time.time()




            alpha_list=[alpha]

            ### MEMCPY -> tensor(GPU) to numpy(CPU)
            # pred1=pred[0].cpu().numpy()
            
            
            pred1=pred[0].cpu()
            # alpha_list.extend([pred1])
            lock.release()
            # t2 = time.time()
            t2 = timer()

            #t2 = time.thread_time()
            inference_time=t2-t1
            total_infer_time=total_infer_time+inference_time
            
            #data_string = pickle.dumps(pred1)
            data_string = pickle.dumps(alpha)
            send_data_size = sys.getsizeof(data_string)
            send_len=len(data_string)
            
            conn2.sendall(struct.pack(">f", alpha))
            conn2.sendall(struct.pack(">f", trans_times[cnt]))
            # conn2.sendall(data_string)
            
            # conn.sendall(struct.pack(">L", send_len)+data_string)
            #Process detections
            # print(f"[{cnt}] Inf time: {inference_time*1000.0}")
            cnt=cnt+1 
            

            if cnt%300==0:
                # print("Average inference time for 300 frames = {:.3f} ms".format((total_infer_time)*1000/300))
                total_infer_time=0

        else :
            time.sleep(0.01)
        if exit_event.is_set():
            # client_socket.close()
            f.close()
            break
    t_end=time.time()
    # f.close()


        
        
def signal_handler(signum, frame):
    global recv_start_f
    global acc_f
    print("signal_handler : exit_event.set()")
    print("signal_handler : exit_event.set()!!")
    recv_start_f.close()
    acc_f.close()
    exit_event.set()
    sys.exit(0)

if __name__ == "__main__": 
    print('main()')
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=3840, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    #check_requirements()

    # q = Queue(maxsize=300)
    recv_q = Queue(maxsize=2)
    pred_q = Queue(maxsize=2)
    exit_event = threading.Event()
    signal.signal(signal.SIGINT, signal_handler)
    p1 = Thread(target=recv_queue, args=(recv_q,))
    # p2 = Thread(target=decode_q, args=(recv_q,q,)) 
    #p3 = Thread(target=infer_q, args=(recv_q, pred_q,)) 
    # p4 = Thread(target=send_qsize, args = (pred_q,))

    p1.daemon=True
    # p2.daemon=True
    #p3.daemon=True
    # p4.daemon=True
    #p3.start() 
    
    
    p1.start() 
            
    p1.join()
    
    print("thread join") 
    
