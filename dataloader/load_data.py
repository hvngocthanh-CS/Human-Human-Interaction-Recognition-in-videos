import torch
from torch import nn

import torch.nn.functional as F
import pysrt
import re
import pickle
import math

import cv2
import os
import numpy as np
import pandas as pd

from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import torchaudio
import random
import torch.optim as optim

from sklearn.metrics import classification_report

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor, ASTForAudioClassification


def read_data(video_dir, text_dir, audio_dir, output_file):
    data = {}

    # Duyệt qua tất cả các file trong thư mục video
    for file in os.listdir(video_dir):
        if file.endswith('.mp4'):
            base_name = os.path.splitext(file)[0]  # Lấy tên file không có đuôi
            video_path = os.path.join(video_dir, file)
            text_path = os.path.join(text_dir, f"{base_name}.srt")
            audio_path = os.path.join(audio_dir, f"{base_name}.mp3")

            # Kiểm tra xem file .srt và .mp3 có tồn tại không
            if os.path.exists(text_path) and os.path.exists(audio_path):
                data[base_name] = {"video": video_path, "text": text_path, "audio": audio_path}
                print(data[base_name])
            else:
                print(f"Warning: Missing text or audio file for {file}")

    # Lưu dữ liệu bằng pickle
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    print(f"Data saved to {output_file}")

    return data

class MultiModalDataset(Dataset):
    def __init__(self, data_pickle, transform=None, target_size=(128, 128), num_frames=512):
        self.data_pickle = data_pickle
        self.transform = transform
        self.target_size = target_size
        self.num_frames = num_frames

        # Load dữ liệu từ pickle
        self.data = self._load_pickle(data_pickle)

        # Lưu danh sách các keys (tên file) để truy cập dữ liệu
        self.keys = list(self.data.keys())

    def _load_pickle(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def _read_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = self._resize_frame(frame)  # Resize frame về kích thước cố định
            frames.append(frame)
        cap.release()

        # Lấy mẫu số lượng frame cố định
        frames = self._sample_frames(frames)

        # Chuyển đổi frames thành tensor
        frames_array = np.array(frames, dtype=np.float32)
        frames_array /= 255.0  # Chuẩn hóa giá trị về [0, 1]

        # Chuyển đổi từ (F, H, W, C) sang (C, F, H, W)
        video_tensor = torch.from_numpy(frames_array).permute(3, 0, 1, 2)  # (C, F, H, W)
        return video_tensor

    def _resize_frame(self, frame):
        return cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)

    def _sample_frames(self, frames):
        num_total_frames = len(frames)

        if num_total_frames >= self.num_frames:
            # Phân bổ đều các nhóm frame
            indices = np.linspace(0, num_total_frames - 1, self.num_frames, dtype=int)
            sampled_frames = [frames[idx] for idx in indices]
        else:
            # Nếu số frame ít hơn, lấy ngẫu nhiên các frame để đủ số lượng
            extra_frames = np.random.choice(num_total_frames, self.num_frames - num_total_frames, replace=True)
            sampled_frames = frames + [frames[idx] for idx in extra_frames]
        return sampled_frames

    def _clean_text(self, text):
        text = re.sub(r"<.*?>", "", text)  # Xóa thẻ HTML
        text = re.sub(r"\([^\)]+\)", "", text)  # Xóa văn bản trong dấu ()
        text = re.sub(r"\[[^\]]+\]", "", text)  # Xóa văn bản trong dấu []
        text = re.sub(r"\s+", " ", text).strip()  # Xóa khoảng trắng dư thừa
        return text

    def _read_srt(self, srt_path):
        subs = pysrt.open(srt_path, encoding='utf-8')
        dialogues = []
        for sub in subs:
            dialogue = self._clean_text(sub.text)
            dialogues.append(dialogue)
        return dialogues

    def pad_texts(self, texts, pad_token="<PAD>"):
        max_length = max(len(text) for text in texts)  # Độ dài lớn nhất
        padded_texts = []
        for text in texts:
            # Thêm padding token nếu câu ngắn hơn max_length
            padded = text + [pad_token] * (max_length - len(text))
            padded_texts.append(padded[:max_length])  # Cắt nếu vượt quá max_length
        return padded_texts

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        video_path = self.data[key]["video"]
        text_path = self.data[key]["text"]
        audio_path = self.data[key]["audio"]

        video_tensor = self._read_video(video_path)

        text = self._read_srt(text_path)
        text = self.pad_texts([text])[0]

        # Áp dụng transform (nếu có)
        if self.transform:
            video_tensor = self.transform(video_tensor)

        return (video_tensor, video_path), (text_path, text), audio_path
    
def custom_collate_fn(batch):
    """
    Hàm collate cho DataLoader để kết hợp các mẫu vào batch (không có labels).
    """
    # Tách các phần tử từ batch
    video_tensors = []
    video_paths = []
    text_paths = []
    texts = []
    audio_paths = []
    
    for item in batch:
        (video_tensor, video_path), (text_path, text), audio_path = item
        video_tensors.append(video_tensor)
        video_paths.append(video_path)
        text_paths.append(text_path)
        texts.append(text)
        audio_paths.append(audio_path)

    # Kết hợp video tensor thành batch
    video_tensors = torch.stack(video_tensors)  # (batch_size, C, F, H, W)

    max_text_len = max(len(t) for t in texts)
    padded_texts = [
        t + ["<PAD>"] * (max_text_len - len(t)) for t in texts
    ]

    # Trả về dictionary
    return {
        "videos": video_tensors,
        "video_paths": video_paths,
        "text_paths": text_paths,
        "texts": padded_texts,
        "audio_paths": audio_paths,
    }


