import torch
import os
import numpy as np
import mediapipe as mp
import torchvision


class Detector:
    def __init__(self, mode="offline"):
        self.face_detection = mp.solutions.face_detection
        if mode == "stream": self.detector = self.face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0)
        elif mode == "offline": self.detector = self.face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)

    def __call__(self, video_frames):
        if isinstance(video_frames, str):
            video_frames = torchvision.io.read_video(video_frames, pts_unit='sec')[0].numpy()

        landmarks, bboxes = self.detect(video_frames, self.detector)
        return landmarks, bboxes
    

    def detect(self, video_frames):
        landmarks = []
        bboxes = []
        for frame in video_frames:
            if frame is not None:
                detected_faces = self.detector.process(frame)
                if not detected_faces.detections:
                    landmarks.append(None)
                    bboxes.append(None)
                    continue
                key_points = []
                for idx, faces in enumerate(detected_faces.detections):
                    # max_id, max_size = 0, 0
                    bboxC = faces.location_data.relative_bounding_box
                    frame_height, frame_width, frame_depth = frame.shape
                    bbox =  int(bboxC.xmin * frame_width), \
                            int(bboxC.ymin * frame_height), \
                            int(bboxC.width * frame_width), \
                            int(bboxC.height * frame_height)
                    # bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                    # if bbox_size > max_size:
                    #     max_id, max_size = idx, bbox_size
                    transformed_relative_keypints = [
                        [int(faces.location_data.relative_keypoints[self.face_detection.FaceKeyPoint(i).value].x * frame_width),
                        int(faces.location_data.relative_keypoints[self.face_detection.FaceKeyPoint(i).value].y * frame_height)] 
                        for i in range(4)]
                    
                    key_points.append(transformed_relative_keypints)
                landmarks.append(np.array(key_points[0]))
                bboxes.append(bbox)

            else:
                landmarks.append(None)
                bboxes.append(None)
        return landmarks, bboxes