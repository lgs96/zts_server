from aiohttp import web
from starlette.responses import PlainTextResponse, Response, JSONResponse
from django.http import HttpResponse

from bdq import *
from utils import *

import time
import json
import os
import threading

os.environ.get("DJANGO_SETTINGS_MODULE")

last_milli = 0
fps = 0
counter = 0
th = 0
recv_size = 0
rl_agent = agent_runner()

async def handle(request):
    name = request.match_info.get('name', "Anonymous")
    text = "Reset, " + name
    print(text)
    rl_agent.agent.end = False

    return web.Response(text=text)

async def rl_handler(request):
    global fps
    global th

    data = await request.json()
    data['network_fps'] = fps
    data['bitrate'] = th

    actions = rl_agent.run_train(data)

    response = {}
    response['res'] = actions[0]
    response['bitrate'] = actions[1]
    response['clock'] = actions[2]

    data = {"res": int(actions[0]), "bitrate": int(actions[1]), "clock":  int(actions[2])}

    #print(data)

    return web.json_response(data)

async def post_handler(request):
    global last_milli
    global fps
    global th
    global counter
    global recv_size

    await request.post()

    counter += 1
    recv_size += int(request.headers.get("content-length"))
    '''
    content_upload_time = current_milli_time()
    object_size = int(request.headers.get("content-length"))

    if (content_upload_time - last_milli > 2000):
        time_gap = content_upload_time - last_milli 
        last_milli = content_upload_time
        fps = round(counter/((time_gap)/1000),2)
        th = round(recv_size*8/2e6,2)
        counter = 0
        recv_size = 0
    else:
        counter += 1
        recv_size += object_size
    '''

    media_type = request.headers.get("content-type")
    content_upload_time = current_milli_time()
    #print('Object size: ', object_size/1e6, fps, th)

    data = {"fps": int(fps), "th": round(float(th),2)}

    #print(data)

    response = web.json_response(data)

    return response

def calculate_fps():
    global counter
    global recv_size
    global fps
    global th

    sampling_sec = 2

    print('Start performance measurement thread')

    while(1):
        time.sleep(sampling_sec)
        fps = round(counter/sampling_sec,2)
        th = round(recv_size*8/2e6,2)
        counter = 0
        recv_size = 0

def current_milli_time():
    return int(round(time.time()*1000))

app = web.Application()
app.add_routes([web.get('/reset', handle),
                web.post('/rl', rl_handler),
                web.post('/video', post_handler)])

if __name__ == '__main__':
    t1 = threading.Thread(target = calculate_fps)
    t1.start()
    web.run_app(app, port = 8888)