import os
import torch
import pickle
import numpy as np
import streamlit as st
from torch.utils.data import DataLoader, TensorDataset
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg
from transformers import AutoTokenizer, AutoModel
from models.multilstm import MultimodalLSTM
from dataloader.load_data import MultiModalDataset
from models.ViViT import Video_ViT
from utils.utils import process_video, merge_shots_into_clips, convert_mp4_to_mp3_folder, transcribe_videos
import re
import torchaudio
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor, ASTForAudioClassification
import shutil   
import random
import base64

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces = True)  # Load tokenizer của BERT
bert_model = AutoModel.from_pretrained("bert-base-uncased")  # Load mô hình BERT

bert_model.eval()

feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
AST_model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

AST_model.eval()

def padding(features, max_length, pad_value=0):
    """
    Hàm padding cho tensor, hỗ trợ cả 1D và 2D tensor.
    """
    if len(features.shape) == 1:
        # Padding cho tensor 1D
        padded_features = torch.nn.functional.pad(
            features, (0, max_length - features.size(0)), value=pad_value
        )
    elif len(features.shape) == 2:
        # Padding cho tensor 2D
        padded_features = torch.nn.functional.pad(
            features, (0, 0, 0, max_length - features.size(0)), value=pad_value
        )
    else:
        raise ValueError(f"Unsupported tensor shape: {features.shape}")
    return padded_features

def pad_scene_level(scene_data):
    """
    Hàm padding tất cả các scene để đảm bảo có cùng độ dài sequence.
    """
    scene_names = list(scene_data.keys())
    # Tính max_length từ visual_features
    max_length = max(scene_data[scene_name]['visual_features'].size(0) for scene_name in scene_names)
    for scene_name in scene_names:
        try:
            visual_features = scene_data[scene_name]['visual_features']
            text_features = scene_data[scene_name]['text_features']
            audio_features = scene_data[scene_name]['audio_features']
            # Padding cho từng loại dữ liệu
            scene_data[scene_name]['visual_features'] = padding(visual_features, max_length, pad_value=0)
            scene_data[scene_name]['text_features'] = padding(text_features, max_length, pad_value=0)
            scene_data[scene_name]['audio_features'] = padding(audio_features, max_length, pad_value=0)
        except KeyError as e:
            raise KeyError(f"Missing key in scene data: {e}")
    return scene_data

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

def group_features(video_paths, visual_features, text_paths, text_features, audio_paths, audio_features):
    scene_feats = {}

    for i, video_path in enumerate(video_paths):
        # Extract scene name from video path
        scene_name = "merged_clip"
        if scene_name not in scene_feats:
            scene_feats[scene_name] = {
                'visual_features': [],
                'text_features': [],
                'audio_features': [],
                'video_paths': [],
                'text_paths': [],
                'audio_paths': [],
            }
        print(visual_features.shape)
        scene_feats[scene_name]['visual_features'].append(visual_features[i].tolist())
        scene_feats[scene_name]['text_features'].append(text_features[i].tolist())
        scene_feats[scene_name]['audio_features'].append(audio_features[i].tolist())
        scene_feats[scene_name]['video_paths'].append(video_paths[i])
        scene_feats[scene_name]['text_paths'].append(text_paths[i])
        scene_feats[scene_name]['audio_paths'].append(audio_paths[i])

    # Convert lists to tensors
    for scene_name in scene_feats:
        scene_feats[scene_name]['visual_features'] = torch.tensor(scene_feats[scene_name]['visual_features'])
        scene_feats[scene_name]['text_features'] = torch.tensor(scene_feats[scene_name]['text_features'])
        scene_feats[scene_name]['audio_features'] = torch.tensor(scene_feats[scene_name]['audio_features'])

    return scene_feats

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

    video_paths_list = []
    text_paths_list = []
    audio_paths_list = []

    visual_model.to(device)
    text_model.to(device)

    # Lặp qua các batch
    for batch_idx, batch in enumerate(data_loader):
        videos = batch['videos'].float().to(device)
        texts = batch['texts']

        video_paths = batch['video_paths']
        text_paths = batch['text_paths']
        audio_paths = batch['audio_paths']

        print(f"Batch {batch_idx + 1}:")
        print(f"  Video Tensors Shape: {videos.shape}")  # (batch_size, C, F, H, W)
        print(f"  Video Paths: {video_paths}")
        print(f"  Text Paths: {text_paths}")
        print(f"  Texts: {texts}")
        print(f"  Audio Paths: {audio_paths}")

        with torch.no_grad():
            visual_features = visual_model.extract_features(videos)

        input_ids, attention_mask = process_texts(texts, tokenizer, device)
        with torch.no_grad():
            outputs = text_model(input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Đặc trưng từ CLS token

        audio_embeddings = extract_audio_features(audio_paths)

        # Lưu các đặc trưng vào danh sách
        visual_features_list.extend(visual_features.tolist())
        text_features_list.extend(cls_embeddings.tolist())
        audio_features_list.extend(audio_embeddings.tolist())

        video_paths_list.extend(video_paths)
        text_paths_list.extend(text_paths)
        audio_paths_list.extend(audio_paths)

    features_data = {
        'video_paths': video_paths_list,
        'visual_features': visual_features_list,
        'text_paths': text_paths_list,
        'text_features': text_features_list,
        'audio_paths': audio_paths_list,
        'audio_features': audio_features_list,
    }

    with open(output_path, "wb") as f:
        pickle.dump(features_data, f)

    print(f"Features saved to {output_path}")

# Define utility functions
def load_features(input_path):
    with open(input_path, "rb") as f:
        features_data = pickle.load(f)
    return (
        features_data['video_paths'],
        torch.tensor(features_data['visual_features']),
        features_data['text_paths'],
        torch.tensor(features_data['text_features']),
        features_data['audio_paths'],
        torch.tensor(features_data['audio_features']),
    )

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

def map_predictions(predictions, id2interaction):
    return [
        [id2interaction[idx] for idx, value in enumerate(pred) if value == 1]
        for pred in predictions
    ]

# Streamlit UI with improved styling
st.title("Interaction Recognition in video using Multimodal Approach")
st.markdown("### Thanh Huynh - Ha Nguyen")
# Step 1: Upload video
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "webm", "mkv", "avi"])

if uploaded_file:
    # Save the uploaded file locally
    input_path = f"./{uploaded_file.name}"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())
    # Detect the file type from the uploaded file name
    with open(input_path, "rb") as video_file:
        video_data = video_file.read()
        encoded_video = base64.b64encode(video_data).decode()
    file_extension = uploaded_file.name.split('.')[-1].lower()
    mime_types = {
        "mp4": "video/mp4",
        "webm": "video/webm",
        "mkv": "video/x-matroska",
        "avi": "video/x-msvideo",
    }

    if file_extension in mime_types:
        mime_type = mime_types[file_extension]
        # Embed the video with autoplay using HTML
        st.markdown(f"""
            <video controls autoplay loop width="100%">
                <source src="data:{mime_type};base64,{encoded_video}" type="{mime_type}">
                Your browser does not support the video tag.
            </video>
        """, unsafe_allow_html=True)
    else:
        st.error("Unsupported video format. Please upload a valid video file (MP4, WEBM, MKV, AVI).")
    # Create temporary directories for intermediate files
    temp_dirs = {
        "shots": "./temp_shots",
        "clips": "./temp_clips",
        "audio": "./temp_audio",
        "srt": "./temp_srt",
    }
    temp_files = {
        "data_pickle": "./temp_data.pkl",
        "features_pickle": "./temp_features.pkl",
    }
    for dir_path in temp_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Create columns for process and results
    process_col, result_col = st.columns([1, 1])  # Equal column width

    with process_col:
        st.markdown("## **Process**")
    
    with result_col:
        st.markdown("## **Results**")

    try:
        # Step 2: Process video into shots
        with process_col:
            st.info("Segmenting scene into shots ...")
        num_shots = process_video(input_path, temp_dirs["shots"])
        with result_col:
            st.markdown(f"""
                <div style="font-size:18px; font-weight:bold; text-align:center;">
                    Shots Extracted: {len(os.listdir(temp_dirs['shots']))}
                </div>
            """, unsafe_allow_html=True)

        # Step 3: Merge shots into clips
        with process_col:
            st.info("Processing shots and merging into clips ...")
        num_clips = merge_shots_into_clips(temp_dirs["shots"], temp_dirs["clips"])
        with result_col:
            st.markdown(f"""
                <div style="font-size:18px; font-weight:bold; text-align:center;">
                    Clips Created: {len(os.listdir(temp_dirs["clips"]))}
                </div>
            """, unsafe_allow_html=True)

        # Step 4: Convert MP4 to MP3
        with process_col:
            st.info("Extracting audio data from clips ...")
        convert_mp4_to_mp3_folder(temp_dirs["clips"], temp_dirs["audio"])
        with result_col:
            st.markdown("""
                <div style="font-size:18px; font-weight:bold; text-align:center;">
                    Audio Extracted
                </div>
            """, unsafe_allow_html=True)

        # Step 5: Transcribe video to SRT
        with process_col:
            st.info("Extracting character dialogue data from clips ...")
        transcribe_videos(temp_dirs["clips"], temp_dirs["srt"])
        with result_col:
            st.markdown("""
                <div style="font-size:18px; font-weight:bold; text-align:center;">
                    Transcription Completed
                </div>
            """, unsafe_allow_html=True)

        # Step 6: Read and save data paths
        with process_col:
            st.info("Preparing data for feature extraction ...")
        data = read_data(temp_dirs["clips"], temp_dirs["srt"], temp_dirs["audio"], temp_files["data_pickle"])
        with result_col:
            st.markdown(f"""
                <div style="font-size:18px; font-weight:bold; text-align:center;">
                    Data Paths Saved: {len(data)} Clips
                </div>
            """, unsafe_allow_html=True)

        # Step 7: Initialize models
        with process_col:
            st.info("Loading models ViViT, BERT, AST ...")

        # Load and initialize models
        ViViT_model = Video_ViT(
            image_size=128,
            frames=512,
            image_patch_size=16,
            frame_patch_size=2,
            num_classes=7,
            dim=1024,
            spatial_depth=6,
            temporal_depth=6,
            heads=8,
            mlp_dim=2048,
            variant="factorized_self_attention",
        )
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        bert_model = AutoModel.from_pretrained("bert-base-uncased").eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ViViT_model.to(device)
        bert_model.to(device)

        # Step 8: Extract features
        with process_col:
            st.info("Extracting multi-modal features ...")
        dataset = MultiModalDataset(data_pickle=temp_files["data_pickle"], target_size=(128, 128), num_frames=64)
        data_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)
        extract_features(data_loader, ViViT_model, bert_model, tokenizer, device, temp_files["features_pickle"])
        with result_col:
            st.markdown("""
                <div style="font-size:18px; font-weight:bold; text-align:center;">
                    Features Extracted and Saved
                </div>
            """, unsafe_allow_html=True)

        # Step 9: Load extracted features
        with process_col:
            st.info("Loading extracted features ...")
        features = load_features(temp_files["features_pickle"])
        test_feats_padded = pad_scene_level(group_features(*features))
        with result_col:
            st.markdown("""
                <div style="font-size:18px; font-weight:bold; text-align:center;">
                    Features Processed for Prediction
                </div>
            """, unsafe_allow_html=True)

        # Prepare features for prediction
        visual_features, text_features, audio_features = [], [], []
        for scene in test_feats_padded.values():
            visual_features.append(scene["visual_features"])
            text_features.append(scene["text_features"])
            audio_features.append(scene["audio_features"])

        visual_features = torch.stack(visual_features)
        text_features = torch.stack(text_features)
        audio_features = torch.stack(audio_features)

        # Step 10: Predict interactions
        with process_col:
            st.info("Predicting interactions in a scene ...")
        classifier_model = MultimodalLSTM(hidden_dim=512, num_labels=7)
        classifier_model.load_state_dict(torch.load("weights/weights_2.pth", map_location=device))
        classifier_model.to(device).eval()

        test_loader = DataLoader(
            TensorDataset(visual_features, text_features, audio_features), batch_size=16, shuffle=False
        )
        test_preds = []
        with torch.no_grad():
            for visual_data, text_data, audio_data in test_loader:
                outputs = classifier_model(visual_data.to(device), text_data.to(device), audio_data.to(device))
                test_preds.append(torch.sigmoid(outputs).cpu().numpy())

        # Step 11: Map predictions to interactions
        id2interaction = {
            0: "asks",
            1: "gives to",
            2: "talks to",
            3: "walks with",
            4: "watches",
            5: "yells at",
            6: "no_interaction",
        }
        test_preds_scene = (np.concatenate(test_preds, axis=0) >= 0.6).astype(int)
        predicted_interactions = [id2interaction[idx] for idx, val in enumerate(test_preds_scene.reshape(-1)) if val == 1]
        interaction_probs = [(id2interaction[idx], prob) for idx, prob in enumerate(test_preds[0].reshape(-1))]
        sorted_interactions = sorted(interaction_probs, key=lambda x: x[1], reverse=True)
        
        if "no_interaction" in predicted_interactions:
            predicted_interactions.remove("no_interaction")
        with result_col:
            st.markdown("""
                <div style="font-size:20px; font-weight:bold; text-align:center; color:green;">
                    Recognition Completed !!!
                </div>
            """, unsafe_allow_html=True)
            for interaction, prob in sorted_interactions:
                st.write(f"{interaction}: {prob:.2f}")
            st.markdown(f"""
                <div style="font-size:18px; font-weight:bold; text-align:center;">
                    Interactions in this scene: {predicted_interactions}
                </div>
            """, unsafe_allow_html=True)
            

    except Exception as e:
        with result_col:
            st.error(f"An error occurred: {e}")
    finally:
        # Cleanup temporary files
        for dir_path in temp_dirs.values():
            shutil.rmtree(dir_path, ignore_errors=True)
        for file_path in temp_files.values():
            if os.path.exists(file_path):
                os.remove(file_path)