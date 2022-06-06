from aiohttp import web
#import multipart as mp
#from multipart import tob
import time
from starlette.responses import PlainTextResponse, Response, JSONResponse
from django.http import HttpResponse
import json
import os
os.environ.get("DJANGO_SETTINGS_MODULE")

last_milli = 0
fps = 0
counter = 0
th = 0
recv_size = 0

async def handle(request):
    name = request.match_info.get('name', "Anonymous")
    text = "Hello, " + name
    print(text)
    return web.Response(text=text)

async def post_handler(request):
    global last_milli
    global fps
    global th
    global counter
    global recv_size

    await request.post()

    content_upload_time = current_milli_time()
    object_size = int(request.headers.get("content-length"))

    if (content_upload_time - last_milli > 1000):
        time_gap = content_upload_time - last_milli 
        last_milli = content_upload_time
        fps = round(counter/((time_gap)/1000),2)
        th = round(recv_size*8/1e6,2)
        counter = 0
        recv_size = 0
    else:
        counter += 1
        recv_size += object_size

    #print(request.post(), request.headers)
    media_type = request.headers.get("content-type")
    #print('Content bytes: ', len(content)/1024 'KB')
    #image = Image.open(io.BytesIO(content))
    #print('Time shift: ', time_shift)
    content_upload_time = current_milli_time()
    #print('Inference finished time: ', inference_finished_time, datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    print('Object size: ', object_size/1e6, fps, th)

    data = {"fps": int(fps), "th": round(float(th),2)}

    response = web.json_response(data)

    return response

def current_milli_time():
    return int(round(time.time()*1000))

app = web.Application()
app.add_routes([web.get('/', handle),
                web.post('/video', post_handler)])

if __name__ == '__main__':
    web.run_app(app, port = 3030)