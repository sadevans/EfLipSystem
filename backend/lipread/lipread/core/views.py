
# Create your views here.
from threading import Thread, Lock
from queue import Queue
import multiprocessing
from PIL import Image
import numpy as np
import subprocess
import threading
import base64
import json
import time
import cv2
import sys
import os
import io

from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.csrf import ensure_csrf_cookie

from .forms import VideoUploadForm

# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from pipelines.pipelines import InferencePipeline
from pipelines.utils.utils import save2vid

import logging
logging.basicConfig(
                    format = '%(asctime)s [%(name)-20s] %(levelname)-8s %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger('views.py')
logger.setLevel(logging.DEBUG)

# from pipelines.pipelines import InferencePipeline
BUFFER_SIZE = 4*30

def index(request):
    form = VideoUploadForm()
    return render(request, 'web_app.html', {'form': form})


def upload_video(request):
    video_url = None
    filename = None
    if request.method == 'POST':
        video = request.FILES['video']
        fs = FileSystemStorage()
        filename = fs.save(video.name, video)
        video_bytes = video.file.read()
        encoded_video = base64.b64encode(video_bytes).decode('ascii')
        video_url = 'data:%s;base64,%s' % ('video/mp4', encoded_video)
        return JsonResponse({'video_url': video_url, 'filename': filename})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)


def process_video(request):
    out_video_url = None
    if request.method == 'GET':
        filename = request.GET.get('filename')
        fs = FileSystemStorage()
        uploaded_file_path = fs.path(filename)
        # config_file = ""
        pipeline = InferencePipeline()
        name_out_file = os.path.join(settings.MEDIA_ROOT,filename[:-4]+'_out.mp4')
        
        transcript = pipeline.process_input_file(filename=uploaded_file_path, save=True, save_file_name=name_out_file)
        print(transcript)
        out_data_bytes = open(name_out_file, "rb").read()
        os.remove(uploaded_file_path)
        # os.remove(name_out_file)
        encoded_video_out = base64.b64encode(out_data_bytes).decode('ascii')
        out_video_url = 'data:%s;base64,%s' % ('video/mp4', encoded_video_out)
        return JsonResponse({'out_video_url': out_video_url})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)


# @ensure_csrf_cookie
def get_video(request):
    mp4_filename = None
    if request.method == 'POST' and 'video' in request.FILES:
        video_file = request.FILES['video']
        fps = float(request.POST.get('frameRate'))
        print('FPS: ', fps, type(fps))
        file_extension = os.path.splitext(video_file.name)[1]
        filename = f'{video_file.name[:-len(file_extension)]}_{int(time.time())}'
        webm_filename = filename + file_extension
        mp4_filename = filename + '.mp4'
        webm_path = os.path.join(settings.MEDIA_ROOT, webm_filename)
        mp4_path = os.path.join(settings.MEDIA_ROOT, mp4_filename)
        with open(webm_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)
        try:
            command = ['ffmpeg', '-i', f'{webm_path}', '-r', f'{int(fps)}', '-c:v', 'libx264', '-crf', '23', '-c:a', 'aac', '-b:a', '128k', '-f', 'mp4', f'{mp4_path}']
            subprocess.run(command, check=True)
            os.remove(webm_path)
            return JsonResponse({'message': 'Video uploaded and saved successfully!', 'filename': mp4_filename})
        except Exception as e:
            logging.error(f"Error occured {e.message} {e.args}")
            mp4_filename = None
            return JsonResponse({'message': 'Video uploaded but cannot saved successfully!', 'filename': mp4_filename})
    else:
        return JsonResponse({'message': 'Invalid request method', 'filename': mp4_filename})
    

@ensure_csrf_cookie
def receive_frame(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image')

            if image_data:
                header, encoded = image_data.split(",", 1)
                data = base64.b64decode(encoded)
                np_array = np.frombuffer(data, np.uint8) 
                frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                print(type(frame))
                return JsonResponse({'success': True, 'message': 'Image uploaded successfully'})
            else:
                return JsonResponse({'success': False, 'message': 'Image data not found in request'})

        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})
    else:
        return JsonResponse({'success': False, 'message': 'Invalid request method'})


class StreamHandler:

    def __init__(self):
        self.frames_buffer = [] # Global list to store frames
        self.frames_to_process= []
        # config_file = "/home/sadevans/space/personal/LRSystem/config/LRS3_V_WER19.1.ini"
        self.pipeline = InferencePipeline()
        self.fps = 30
        # self.frames_buffer = Queue()  # Queue for gathering incoming frames
        # self.processing_lock = Lock()
        self.processing_semaphore = threading.Semaphore(1) 
        self.stop_processing = False
        self.process_atleast_once = False

    # @ensure_csrf_cookie
    def receive_frame(self, request):
        if request.method == 'POST':
            try:
                data = json.loads(request.body)
                image_data = data.get('image')
                self.fps = float(str(data.get('fps')))
                if image_data:
                    print('here')
                    header, encoded = image_data.split(",", 1)
                    data = base64.b64decode(encoded)
                    np_array = np.frombuffer(data, np.uint8) 
                    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                    self.frames_buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    # if not self.stop_processing:
                    if len(self.frames_buffer) == BUFFER_SIZE and not self.stop_processing:
                        print('IN THIS IF')
                        self.processing_semaphore.acquire()
                        self.frames_to_process = self.frames_buffer.copy()
                        self.frames_buffer = []
                        thread = threading.Thread(target=self.process_frames)
                        thread.start()
                    
                    # if len(self.frames_buffer) > 0 and self.stop_processing:
                    #     print('NOW THIS IF')
                    #     self.process_remaining_frames()

                    return JsonResponse({'success': True, 'message': 'Image uploaded successfully'})
                    # else:
                    #     self.process_remaining_frames()
                    #     return JsonResponse({'success': True, 'message': 'Stopped processing'})
                else:
                    return JsonResponse({'success': False, 'message': 'Image data not found in request'})

            except Exception as e:
                return JsonResponse({'success': False, 'message': str(e)})
        else:
            return JsonResponse({'success': False, 'message': 'Invalid request method'})
        

    def process_frames(self):
        print('HERE IN PROCESS FRAMES')
        print(len(self.frames_to_process))
        transcript, output_frames = self.pipeline.process_video_frames(video_frames=np.array(self.frames_to_process), fps=self.fps)
        print('TRANSCRIPT: ', transcript)
        self.processing_semaphore.release()
        self.process_atleast_once = True
        print(len(self.frames_buffer))
        thread = threading.Thread(target=self.give_frames, args=([output_frames]))
        thread.start()
        # print()
        # if len(self.frames_buffer) > 0 and self.stop_processing:
        # if self.stop_processing:
            # self.process_remaining_frames()


    def give_frames(self, out_frames):
        for frame in out_frames:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            print(type(frame))
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

    def process_remaining_frames(self):
        print('HERE IN PROCESS REMAINING FRAMES')
        if len(self.frames_buffer)>0:
            self.processing_semaphore.acquire()
            self.frames_to_process = self.frames_buffer.copy()
            self.frames_buffer = []
            thread = threading.Thread(target=self.process_frames)
            thread.start()
            self.process_atleast_once = False
            self.stop_processing = False
            # self.frames_to_process = []


    def change_stream_flag(self, request):
        print('IN CHANGE STREAM FLAG')
        if request.method == 'POST':
            if not bool(int(request.body)):
                self.stop_processing = True
                if len(self.frames_buffer) > 0 and self.process_atleast_once:
                    print('ONCE PROCESSED FRAMES< WAIT')
                    self.processing_semaphore.acquire()
                    self.process_remaining_frames()
                elif not self.process_atleast_once:
                    print('NOT PROCESSED FRAMES< FIRST PROCESSING')
                    print(len(self.frames_buffer))
                    self.process_remaining_frames()
            return JsonResponse({'success': True, 'message': 'Stopped processing'})

            # return JsonResponse({'success': True, 'message': 'Stream flag is changed succesfully'})
            
        else:
            return JsonResponse({'success': False, 'message': 'Invalid request method'})
        
        