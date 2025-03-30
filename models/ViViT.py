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

# helpers
def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.3):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
    
class FactorizedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        b, f, n, _ = x.shape
        for spatial_attn, temporal_attn, ff in self.layers:
            x = rearrange(x, 'b f n d -> (b f) n d')
            x = spatial_attn(x) + x
            x = rearrange(x, '(b f) n d -> (b n) f d', b=b, f=f)
            x = temporal_attn(x) + x
            x = ff(x) + x
            x = rearrange(x, '(b n) f d -> b f n d', b=b, n=n)

        return self.norm(x)
    
class TubeletEmbedding(nn.Module):
    def __init__(self, channels, tubelet_size, patch_size, dim):
        super().__init__()
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        patch_dim = channels * tubelet_size * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            # Điều chỉnh Rearrange để có đủ số chiều
            Rearrange('b c (f tf) (h ph) (w pw) -> b f (h w) (tf ph pw c)',
                      tf=tubelet_size, ph=patch_size, pw=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, video):
        return self.to_patch_embedding(video)
    
class Video_ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        dim,
        spatial_depth,
        temporal_depth,
        heads,
        mlp_dim,
        pool = 'cls',
        channels = 3,
        dim_head = 64,
        dropout = 0.3,
        emb_dropout = 0.3,
        variant = 'factorized_self_attention',):

        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'
        assert variant in ('factorized_encoder', 'factorized_self_attention'), f'variant = {variant} is not implemented'

        num_image_patches = (image_height // patch_height) * (image_width // patch_width)
        num_frame_patches = (frames // frame_patch_size)

        #patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.global_average_pool = pool == 'mean'

        #Tubelet or Uniform sampling

        self.to_patch_embedding = TubeletEmbedding(
            channels=channels,
            tubelet_size=frame_patch_size,
            patch_size=patch_height,
            dim=dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_image_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None

        if variant == 'factorized_encoder':
            self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None
            self.spatial_transformer = Transformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout)
            self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, dropout)
        elif variant == 'factorized_self_attention':
            assert spatial_depth == temporal_depth, 'Spatial and temporal depth must be the same for factorized self-attention'
            self.factorized_transformer = FactorizedTransformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        #self.mlp_head = nn.Linear(dim, num_classes)
        self.variant = variant

    def extract_features(self, video):
        x = self.to_patch_embedding(video)
        b, f, n, _ = x.shape
        x = x + self.pos_embedding[:, :f, :n]

        if exists(self.spatial_cls_token):
            spatial_cls_tokens = repeat(self.spatial_cls_token, '1 1 d -> b f 1 d', b=b, f=f)
            x = torch.cat((spatial_cls_tokens, x), dim=2)

        x = self.dropout(x)

        if self.variant == 'factorized_encoder':
            x = rearrange(x, 'b f n d -> (b f) n d')
            x = self.spatial_transformer(x)
            x = rearrange(x, '(b f) n d -> b f n d', b=b)
            x = x[:, :, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b f d', 'mean')
            if exists(self.temporal_cls_token):
                temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b=b)
                x = torch.cat((temporal_cls_tokens, x), dim=1)
            x = self.temporal_transformer(x)
            x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')

        elif self.variant == 'factorized_self_attention':
            x = self.factorized_transformer(x)
            x = x[:, 0, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b d', 'mean')

        return x

    def forward(self, video):
        x = self.extract_features(video)
        return x  # Trả về đặc trưng video mà không qua lớp phân loại