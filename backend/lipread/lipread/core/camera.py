from imutils.video import VideoStream
import imutils
import cv2,os,urllib.request
import numpy as np
import pyaudio
import wave
import struct
from django.conf import settings


class VideoCamera(object):
	def __init__(self):
		self.rate = 44100
		self.bits_per_sample = 16
		self.audio_format = pyaudio.paInt16
		self.chunk = 1024
		self.channels = 2

		self.record_flag = False
		self.frames = []
		self.waveforms = []

		self.wav_file = wave.open("recorded_vid_out.wav", 'wb')
		self.wav_file.setnchannels(self.channels)
		self.wav_file.setsampwidth(self.bits_per_sample // 8)
		self.wav_file.setframerate(self.rate)

		self.video = cv2.VideoCapture(0)
		self.audio = pyaudio.PyAudio()
		self.audio_stream = self.audio.open(format=self.audio_format,
                                            channels=self.channels,
                                            rate=self.rate,
                                            input=True,
                                            frames_per_buffer=self.chunk)

		self.fps = self.video.get(cv2.CAP_PROP_FPS)
		print("FPS: ", self.fps)
		

	# def callback(self, in_data, frame_count, time_info, status):
	# 	self.wav_file.writeframes(in_data)
	# 	return None, pyaudio.paContinue

	# def __del__(self):
	# 	self.video.release()
	# 	self.audio_stream.stop_stream()
	# 	self.audio_stream.close()
	# 	self.audio.terminate()
	# 	# self.wav_file.close()
	# 	cv2.destroyAllWindows()

	def get_frame(self):
		succes, frame = self.video.read()

		if succes and self.record_flag: 
			print('here')
			self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		audio_data = self.audio_stream.read(self.chunk)
		
		if self.record_flag:
			# self.wav_file.writeframes(audio_data)
			self.waveforms.append(audio_data)
			# print(struct.unpack('>f', audio_data)[0])
		ret, jpeg = cv2.imencode('.jpg', frame)
		return jpeg.tobytes()


# class IPWebCam(object):
# 	def __init__(self, url):
# 		self.url = url

# 	def __del__(self):
# 		cv2.destroyAllWindows()

# 	def get_frame(self):
# 		imgResp = urllib.request.urlopen(self.url)
# 		imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
# 		img= cv2.imdecode(imgNp,-1)
# 		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
# 		# so we must encode it into JPEG in order to correctly display the
# 		# video stream
# 		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 		faces_detected = face_detection_webcam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
# 		for (x, y, w, h) in faces_detected:
# 			cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
# 		resize = cv2.resize(img, (640, 480), interpolation = cv2.INTER_LINEAR) 
# 		frame_flip = cv2.flip(resize,1)
# 		ret, jpeg = cv2.imencode('.jpg', frame_flip)
# 		return jpeg.tobytes()

		
class LiveWebCam(object):
	def __init__(self, url):
		self.url = cv2.VideoCapture(url)
		self.frames = []

	def __del__(self):
		cv2.destroyAllWindows()

	def get_frame(self, record=False):
		success, imgNp = self.url.read()
		if success and record: self.frames.append(imgNp)
		# resize = cv2.resize(imgNp, (640, 480), interpolation = cv2.INTER_LINEAR) 
		# ret, jpeg = cv2.imencode('.jpg', resize)
		# return jpeg.tobytes()