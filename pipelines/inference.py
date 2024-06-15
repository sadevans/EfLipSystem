import torch
import hydra
from pipelines.pipelines import InferencePipeline


# @hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
# def inference(cfg):
#     device = torch.device(f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available() and cfg.gpu_idx >= 0 else "cpu")
#     cfg.detector = "mediapipe"
#     pipeline = InferencePipeline(cfg.config_filename, device=device, detector=cfg.detector, face_track=True)
#     output = pipeline.process_input_file(cfg.data_filename, save=True, cfg.save_filename)
#     print(f"hyp: {output}")


# if __name__ == '__main__':
#     inference()