import os
import torch
import torchvision
import torchaudio
import numpy as np
# import matplotlib.pyplot as plt
import sys
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pipelines.data.dataloaders import VSRDataLoader
from pipelines.detectors.mediapipe.detector import Detector
from pipelines.utils.utils import overlay_roi, save2vid
from model.EfLipReading.model.model_module import ModelModule
import hydra
current_file_directory = os.path.abspath(__file__)


class InferencePipeline(torch.nn.Module):
    def __init__(self, detector="mediapipe", face_track=False, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(InferencePipeline, self).__init__()
        self.modality="video"
        dir = '/'.join(current_file_directory.split('/')[:-2])
        with open(f"{dir}/model/EfLipReading/configs/config.yaml", 'r') as file:
            cfg = OmegaConf.load(file)
        self.dataloader = VSRDataLoader(subset="test", convert_gray=False)
        self.modelmodule = ModelModule(cfg, mode="infer")

        cfg.pretrained_model_path = f'{dir}/model_zoo/epoch=8.ckpt'
        self.modelmodule.load_state_dict(torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage)["state_dict"])

        self.detector = Detector()


    def split_into_batches(self,data, batch_size):
        """
        Split the input data into batches of a specified size.
        
        Args:
            data (torch.Tensor): The input data to be split.
            batch_size (int): The desired batch size.
        
        Returns:
            list: A list of batches, where each batch is a torch.Tensor.
        """
        # Get the shape of the input data
        batch_dim, *other_dims = data.shape
        
        # Calculate the number of batches
        num_batches = batch_dim // batch_size
        
        # Use torch.as_strided to create the batches
        batches = torch.as_strided(data, size=(num_batches, batch_size, *other_dims),
                                stride=(batch_size, *[d for d in other_dims]))
        
        # Convert the batches to a list
        batches = list(batches)
        
        return batches
    

    def load_data(self, filename: str):
        video_frames, fps, audio_frames, sample_rate = None, None, None, None
        if self.modality in ['video', 'audiovisual']:
            video_frames, fps = self.load_video(filename)
            audio_frames, sample_rate = self.load_audio(filename)
        print("SHAPE VIDEO FRAMES: ", video_frames.shape)
        return video_frames, fps, audio_frames, sample_rate


    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate
    

    def load_video(self, data_filename):
        video, _, info = torchvision.io.read_video(data_filename, pts_unit='sec')
        print(info['video_fps'])
        return video.numpy(), info['video_fps']


    def process_input_file(self, filename:str, save: bool = True, save_file_name: str = './out.mp4'):
        video_frames, fps, audio_frames, sample_rate = self.load_data(filename)
        landmarks, bboxes = None, None
        if self.modality in ['video', 'audiovisual']:
            landmarks, bboxes = self.detector.detect(video_frames)
        data = self.dataloader.process_data(video_frames=video_frames, landmarks=landmarks)
        transcript = self.modelmodule(data)
        
        if landmarks is not None:
            print(video_frames.shape, self.dataloader.color_roi.shape)
            output_video = overlay_roi(video_frames, self.dataloader.color_roi.permute(0,2,3,1).numpy(), bboxes, fps=fps, text=transcript)
            # output_video = overlay_roi(video_frames, self.dataloader.color_roi.numpy(), bboxes)

            if save: save2vid(filename, save_file_name, output_video, frames_per_second=fps)

        return transcript
    

    def process_video_frames(self, video_frames: np.ndarray, fps: float):
        landmarks, bboxes = None, None
        if self.modality in ['video', 'audiovisual']:
            landmarks, bboxes = self.detector.detect(video_frames)
        data = self.dataloader.process_data(video_frames=video_frames, landmarks=landmarks)
        transcript = self.model.infer(data)
        if landmarks is not None:
            print(video_frames.shape, self.dataloader.color_roi.shape)
            output_video = overlay_roi(video_frames, self.dataloader.color_roi.permute(0,2,3,1).numpy(), bboxes)
        return transcript, output_video