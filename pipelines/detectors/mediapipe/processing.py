import cv2
import os
import numpy as np
from skimage import transform as tf


class VideoHandler:
    def __init__(self, mean_face_path="20words_mean_face.npy", crop_width=96, crop_height=96,
                 start_idx=3, stop_idx=4, window_margin=12, convert_gray=False):
        self.reference = np.load(os.path.join(os.path.dirname(__file__), mean_face_path))

        self.crop_width = crop_width
        self.crop_height = crop_height
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.window_margin = window_margin
        self.convert_gray = convert_gray


    def __call__(self, video_frames, landmarks):
        # должна работать и для полного видео и для реал-тайма (когда один фрейм)
        interpolated_landmarks = self.interpolate(landmarks)
        if interpolated_landmarks is None:
            return None
        cropped_data = self.crop_roi(video_frames, interpolated_landmarks) # возвращает либо эрэй эрэев, либо просто эрэй (если один фрейм)
        
        return cropped_data
    

    def crop_roi(self, video_frames, landmarks):
        cropped_data = []
        for idx, frame in enumerate(video_frames):
            transformed_frame, transformed_landmarks = self.affine_transform(frame, landmarks[idx], self.reference, \
                                                                             grayscale=self.convert_gray)
            roi = self.cut_roi_patch(transformed_frame, transformed_landmarks[self.start_idx : self.stop_idx], \
                                self.crop_height // 2, self.crop_width // 2)
            cropped_data.append(roi)

        return np.array(cropped_data)



    def interpolate(self, landmarks):
        valid_landmarks = [idx for idx, landmark in enumerate(landmarks) if landmark is not None]

        if not valid_landmarks:
            return None
        
        missed_landmarks = []
        for idx in range(1, len(valid_landmarks)):
            if valid_landmarks[idx] - valid_landmarks[idx-1] > 1:
                missed_landmarks.append((valid_landmarks[idx-1], valid_landmarks[idx]))
        landmarks = self.linear_interpolation(landmarks, missed_landmarks)
        del missed_landmarks

        valid_landmarks = [idx for idx, landmark in enumerate(landmarks) if landmark is not None]
        if valid_landmarks:
            landmarks[: valid_landmarks[0]] = [landmarks[valid_landmarks[0]]] * valid_landmarks[0]
            landmarks[valid_landmarks[-1] :] = [landmarks[valid_landmarks[-1]]] * (len(landmarks) - valid_landmarks[-1])

        return landmarks


    def cut_roi_patch(self, frame, landmarks, height, width, threshold=5):
        center_x, center_y = np.mean(landmarks, axis=0)
        if abs(center_y - frame.shape[0] / 2) > height + threshold:
            raise OverflowError("wrong height value: too big")
        if abs(center_x - frame.shape[1] / 2) > width + threshold:
            raise OverflowError("wrong width value: too big")
        y_min = int(round(np.clip(center_y - height, 0, frame.shape[0])))
        y_max = int(round(np.clip(center_y + height, 0, frame.shape[0])))
        x_min = int(round(np.clip(center_x - width, 0, frame.shape[1])))
        x_max = int(round(np.clip(center_x + width, 0, frame.shape[1])))
        cutted_img = np.copy(frame[y_min:y_max, x_min:x_max])
        return cutted_img


    def linear_interpolation(self, landmarks, missed_landmarks):
        for pair in missed_landmarks:
            start_idx, stop_idx = pair
            start_lm, stop_lm = landmarks[start_idx], landmarks[stop_idx]
            delta = stop_lm - start_lm
            for idx in range(1, stop_idx - start_idx):
                landmarks[start_idx + idx] = (start_lm + idx / float(stop_idx - start_idx) * delta)

        # del start_idx, stop_idx, start_lm, stop_lm, delta
        return landmarks
    

    def affine_transform(self, frame, landmarks, reference, grayscale=False, target_size=(256, 256), reference_size=(256, 256), \
                         stable_points=(0, 1, 2, 3), interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0,):
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        stable_reference = self.get_stable_reference(reference, reference_size, target_size)
        transform = cv2.estimateAffinePartial2D(np.vstack([landmarks[x] for x in stable_points]), stable_reference, method=cv2.LMEDS)[0]
        # transform = self.estimate_affine_transform(landmarks, stable_points, stable_reference)
        transformed_frame, transformed_landmarks = self.apply_affine_transform(frame, landmarks, transform, target_size, interpolation, \
                                                                               border_mode, border_value)

        return transformed_frame, transformed_landmarks
    

    def get_stable_reference(self, reference, reference_size, target_size):
        # R EYE, L EYE, NOSE, MOUTH
        stable_reference = np.vstack(
            [
                np.mean(reference[36:42], axis=0),
                np.mean(reference[42:48], axis=0),
                np.mean(reference[31:36], axis=0),
                np.mean(reference[48:68], axis=0),
            ]
        )
        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0
        return stable_reference
    

    def apply_affine_transform(self, frame, landmarks, transform, target_size, interpolation, border_mode, border_value):
        transformed_frame = cv2.warpAffine(frame, transform, dsize=(target_size[0], target_size[1]), flags=interpolation,\
                                           borderMode=border_mode,borderValue=border_value)
        transformed_landmarks = (np.matmul(landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose())
        return transformed_frame, transformed_landmarks