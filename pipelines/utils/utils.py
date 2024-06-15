import os
import cv2
import numpy as np
import torchvision
import torchaudio
from typing import Dict, Any, Optional
from torch import Tensor
from scipy.io.wavfile import read


def overlay_roi(video_frames, roi_frames, bboxes, fps=25, text=None, font_scale=0.5, color=(255, 255, 255), thickness=2):

    if text is not None:
        duration = len(video_frames) // fps
        letter_duration = duration / len(text)
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    video_height, video_width, video_depth = video_frames.shape[1:]
    roi_height, roi_width, roi_depth = roi_frames.shape[1:]
    k = video_width/video_height
    offset_y = video_height//(int(k*10))
    offset_x = video_width//10

    i=0
    current_letter_index = 0
    for vid_frame, roi_frame, bbox in zip(video_frames, roi_frames, bboxes):
        
        frame_time = i / fps
        if text is not None:
            if frame_time >= current_letter_index * letter_duration and text is not None:
                current_text = text[:current_letter_index + 1]
                current_letter_index += 1
        
        roi_frame = cv2.resize(roi_frame, (int(roi_width*k), int(roi_height*k))) 
        vid_frame[offset_x:offset_x + int(roi_width*k), offset_y:offset_y + int(roi_height*k)] = roi_frame
        cv2.rectangle(vid_frame, (offset_y, offset_x), (offset_y + int(roi_height*k), offset_x + int(roi_width*k)), (255, 0, 255), 2)
        if bbox is not None: cv2.rectangle(vid_frame, (bbox[0], bbox[1]), \
                                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 255), 2)
        i += 1

        if text is not None:
            text_x = offset_x
            if text_x + text_size[0] > video_width:
                text_x = video_width - text_size[0]
            cv2.putText(vid_frame, current_text, (text_x, int(video_height - text_size[1] - offset_x)), \
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return video_frames


def save2vid(filename, save_file_name, output_video, frames_per_second):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        waveforms, sample_rate = torchaudio.load(filename, normalize=True)
        audio_array = np.ascontiguousarray(waveforms.numpy())
    except:
        audio_array = None
    if audio_array is not None:
        torchvision.io.write_video(
            filename=save_file_name,
            video_array=output_video,
            audio_array=audio_array,
            fps=frames_per_second,
            audio_fps=sample_rate,
            audio_codec='aac'
        )
    else:
        torchvision.io.write_video(
            filename=save_file_name,
            video_array=output_video,
            fps=frames_per_second
        )
