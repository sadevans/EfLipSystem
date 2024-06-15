import os
import cv2
import torch
import torchvision
import numpy as np

class AVSRDataLoader:
    def __init__(self, modality="video", detector="mediapipe", subset = "train", transform = True, convert_gray=True):
        self.modality = modality
        self.detector_type = detector
        self.transform = transform
        self.subset = subset
        
        if self.detector_type == "mediapipe":
            from  detectors.mediapipe.processing import VideoHandler
            from  transforms import VideoTransform
            from detectors.mediapipe.detector import Detector
            self.video_transform = VideoTransform(subset=self.subset)
            self.video_handler = VideoHandler(convert_gray=convert_gray)
            self.detctor = Detector()

    
    def __call__(self, filename):
        video_frames = None, None, None
        video_frames = torchvision.io.read_video(filename, pts_unit='sec')[0].numpy()
        landmarks = self.detector.detect(video_frames)

        self.process_data(video_frames, landmarks)


    def process_data(self, video_frames=None, landmarks=None,):
        if video_frames is None:
            return "Video data is None"
        video_data = self.video_handler(video_frames, landmarks)
        print(video_data.shape)
        video_data = torch.tensor(video_data)
        video_data = video_data.permute(0, 3, 1, 2)
        print('CURRENT SHAPE: ', video_data.shape)
        if self.subset == "test": self.color_roi = video_data.clone()
        if self.transform:
            video_data = self.video_transform(video_data)
            print('NEW SHAPE: ', video_data.shape)
        return video_data
