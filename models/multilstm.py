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

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.4):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, attn_mask=None, key_padding_mask=None):
        attn_output, _ = self.multihead_attn(
            x, context, context, 
            attn_mask=attn_mask, 
            key_padding_mask=key_padding_mask
        )
        attn_output = self.dropout(attn_output)
        return self.norm(attn_output + x)

class MultimodalLSTM(nn.Module):
    def __init__(self, hidden_dim=512, num_labels=7, dropout_rate=0.3):
        super(MultimodalLSTM, self).__init__()

        # LSTM layers (Bidirectional → output size = hidden_dim * 2)
        self.visual_lstm = nn.LSTM(input_size=1024, hidden_size=hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.text_lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.audio_lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, num_layers=2, batch_first=True, bidirectional=True)

        # Cross-Attention (adjusted to hidden_dim * 2 due to bidirectional LSTM)
        self.cross_attn_vt = CrossAttention(hidden_dim * 2)  
        self.cross_attn_va = CrossAttention(hidden_dim * 2)  
        self.cross_attn_ta = CrossAttention(hidden_dim * 2) 

        # Fusion MLP
        self.fusion_fc1 = nn.Sequential(
            nn.Linear(hidden_dim * 6, 1024),  # hidden_dim * 2 per modality → 6 modalities
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(dropout_rate)
        )

        self.fusion_fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout_rate)
        )

        self.fusion_fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout_rate)
        )

        self.output_layer = nn.Linear(256, num_labels)

    def forward(self, visual_input, text_input, audio_input):
        # Normalize inputs (prevent vanishing/exploding activations)
        #visual_input = (visual_input - visual_input.mean(dim=1, keepdim=True)) / (visual_input.std(dim=1, keepdim=True) + 1e-6)
        #text_input = (text_input - text_input.mean(dim=1, keepdim=True)) / (text_input.std(dim=1, keepdim=True) + 1e-6)
        #audio_input = (audio_input - audio_input.mean(dim=1, keepdim=True)) / (audio_input.std(dim=1, keepdim=True) + 1e-6)

        # LSTM Encoding
        visual_feat, _ = self.visual_lstm(visual_input)  
        text_feat, _ = self.text_lstm(text_input)    
        audio_feat, _ = self.audio_lstm(audio_input)

        # Cross-Attention (Parallel Processing)
        visual_feat = self.cross_attn_vt(visual_feat, text_feat)
        visual_feat = self.cross_attn_va(visual_feat, audio_feat)

        text_feat = self.cross_attn_ta(text_feat, audio_feat)
        text_feat = self.cross_attn_vt(text_feat, visual_feat)

        audio_feat = self.cross_attn_va(audio_feat, visual_feat)
        audio_feat = self.cross_attn_ta(audio_feat, text_feat)

        # Global average pooling (reduce temporal dimension)
        visual_feat = torch.mean(visual_feat, dim=1)  
        text_feat = torch.mean(text_feat, dim=1)  
        audio_feat = torch.mean(audio_feat, dim=1)  

        # Concatenation of all modalities
        fusion_out = torch.cat((visual_feat, text_feat, audio_feat), dim=-1)  # (batch_size, hidden_dim * 6)

        # MLP Fusion
        fusion_out = self.fusion_fc1(fusion_out)
        fusion_out = self.fusion_fc2(fusion_out)
        fusion_out = self.fusion_fc3(fusion_out)

        # Output layer
        output = self.output_layer(fusion_out)  # (batch_size, num_labels)

        return output