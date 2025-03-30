import torch
from torch import nn

import torch.nn.functional as F
import re
import pickle
import math

import cv2
import os
import numpy as np
import pandas as pd

from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import transforms
import torchaudio
import random
import torch.optim as optim

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

from sklearn.metrics import classification_report

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor, ASTForAudioClassification
from models.ViViT import Video_ViT

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces = True)  # Load tokenizer của BERT
bert_model = AutoModel.from_pretrained("bert-base-uncased")  # Load mô hình BERT

bert_model.eval()

feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
AST_model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

AST_model.eval()

ViViT_model = Video_ViT(
    image_size = 128,          # image size
    frames = 512,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = 2,      # frame patch size
    num_classes = 7,
    dim = 1024,
    spatial_depth = 6,         # depth of the spatial transformer
    temporal_depth = 6,        # depth of the temporal transformer
    heads = 8,
    mlp_dim = 2048,
    variant = 'factorized_self_attention', # or 'factorized_encoder'
)

def extract_audio_features(audio_paths):
    features = []
    for audio_path in audio_paths:
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert stereo to mono nếu cần

        # Resample về 16kHz nếu không đúng
        if sample_rate != 16000:
            resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resample_transform(waveform)

        # Chuyển đổi waveform thành input cho AST model
        inputs = feature_extractor(waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

        # Trích xuất đặc trưng mà không cần thực hiện phân loại
        with torch.no_grad():
            outputs = AST_model.audio_spectrogram_transformer(**inputs)
            feature = outputs.last_hidden_state

        pooled_features = torch.mean(feature, dim=1)
        features.append(pooled_features.squeeze(0))
    return torch.stack(features)

def process_texts(texts, tokenizer, device, max_length=512):
    # Chuẩn hóa văn bản: loại bỏ ký tự đặc biệt
    texts = [re.sub(r'[^\w\s.,!?-]', '', str(text)) for text in texts]

    # Token hóa văn bản
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    return input_ids, attention_mask

def extract_features(data_loader, visual_model, text_model, tokenizer, device, output_path):
    visual_features_list = []
    text_features_list = []
    audio_features_list = []

    labels_list = []

    video_paths_list = []
    text_paths_list = []
    audio_paths_list = []

    visual_model.to(device)
    text_model.to(device)

    for i, batch in enumerate(data_loader):
        videos = batch['videos'].float().to(device)
        texts = batch['texts']
        labels = batch['labels']

        video_paths = batch['video_paths']
        text_paths = batch['text_paths']
        audio_paths = batch['audio_paths']


        with torch.no_grad():
            visual_features = visual_model.extract_features(videos)

        input_ids, attention_mask = process_texts(texts, tokenizer, device)
        with torch.no_grad():
            outputs = text_model(input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Đặc trưng từ CLS token


        audio_embeddings = extract_audio_features(audio_paths)

        # Lưu các đặc trưng và labels vào danh sách
        visual_features_list.extend(visual_features.tolist())
        text_features_list.extend(cls_embeddings.tolist())
        audio_features_list.extend(audio_embeddings.tolist())
        labels_list.extend(labels)

        video_paths_list.extend(video_paths)
        text_paths_list.extend(text_paths)
        audio_paths_list.extend(audio_paths)

        # In ra thông báo mỗi 50 batch
        if (i + 1) % 50 == 0:
            print(f"Batch {i + 1}/{len(data_loader)} completed.")

    features_data = {
        'video_paths': video_paths_list,
        'visual_features': visual_features_list,
        'text_paths': text_paths_list,
        'text_features': text_features_list,
        'audio_paths': audio_paths_list,
        'audio_features': audio_features_list,
        'labels': labels_list,
    }


    with open(output_path, "wb") as f:
        pickle.dump(features_data, f)

    print(f"Features saved to {output_path}")