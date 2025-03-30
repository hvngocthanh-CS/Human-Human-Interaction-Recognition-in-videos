import os
from tqdm import tqdm
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg
from moviepy.editor import VideoFileClip, concatenate_videoclips
from pydub import AudioSegment
from groq import Groq

def process_video(input_path, output_dir):
    """
    Processes a single video file, detects scenes, and splits the video into scenes.

    Args:
        input_path (str): Path to the input video file.
        output_dir (str): Path to the directory where processed scenes will be saved.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Detect scenes and split the video
    scene_list = detect(input_path, AdaptiveDetector())
    split_video_ffmpeg(input_path, scene_list, output_dir)

def merge_shots_into_clips(folder_path, output_folder):
    # Get a sorted list of video files in the folder
    video_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))])

    if not video_files:
        print("No video files found in the folder.")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process shots in groups of 3
    for i in range(0, len(video_files) - 2, 1):
        # Get the current group of shots (3 or fewer)
        group = video_files[i:i+3]

        # Load the video clips
        clips = [VideoFileClip(os.path.join(folder_path, file)) for file in group]

        # Concatenate the clips
        merged_clip = concatenate_videoclips(clips, method="compose")

        # Define the output file name
        output_file = os.path.join(output_folder, f"merged_clip_{i+1}.mp4")

        # Write the merged clip to the output file
        merged_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

        # Close the clips to free up memory
        for clip in clips:
            clip.close()

        merged_clip.close()

    print("Merging completed! Clips are saved in:", output_folder)

def convert_mp4_to_mp3_folder(input_folder, output_folder):
    """
    Convert all MP4 files in the input folder to MP3 format and save them in the output folder.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".mp4"):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_name = os.path.splitext(file_name)[0] + ".mp3"
            output_file_path = os.path.join(output_folder, output_file_name)

            # Convert MP4 to MP3
            try:
                audio = AudioSegment.from_file(input_file_path, format="mp4")
                audio.export(output_file_path, format="mp3")
                print(f"Converted: {file_name} -> {output_file_name}")
            except Exception as e:
                print(f"Error converting {file_name}: {e}")

def transcribe_with_groq(file_path):
    """
    Use Groq API to transcribe speech to text in English.
    """
    client = Groq(api_key='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')
    filename = os.path.basename(file_path)

    with open(file_path, "rb") as file:
        result = client.audio.transcriptions.create(
            file=(filename, file.read()),
            model="whisper-large-v3",
            language='en',
            temperature=0.2,
            response_format="verbose_json"  # Get detailed transcription with timestamps
        )
    # Convert result to dictionary using model_dump() (Pydantic V2)
    transcription_data = result.model_dump()
    return transcription_data

def save_to_srt(transcription_result, output_path):
    """
    Convert the transcription result to a .srt file.
    """
    with open(output_path, "w", encoding="utf-8") as srt_file:
        for index, segment in enumerate(transcription_result["segments"], start=1):
            start_time = convert_to_srt_time(segment["start"])
            end_time = convert_to_srt_time(segment["end"])
            text = segment["text"]
            
            # Write segment to SRT file
            srt_file.write(f"{index}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{text.strip()}\n\n")

def convert_to_srt_time(seconds):
    """
    Convert seconds to SRT time format (hh:mm:ss,mmm).
    """
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    minutes = seconds // 60
    hours = minutes // 60
    minutes = minutes % 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def transcribe_videos(folder_path, output_folder):
    """
    Processes video clips from a folder, transcribes each clip, and saves the transcripts to a single output file.

    Args:
        folder_path (str): Path to the folder containing the video clips.
        output_file (str): Path to the output file where transcripts will be saved.

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)
    for scene_name in tqdm(os.listdir(folder_path)):
        scene_path = os.path.join(folder_path, scene_name)
        if not scene_path.lower().endswith(('.mp4', '.mkv', '.avi')):
            continue
        temp = scene_name.split('.')[0]
        output_name = f'{temp}.srt'
        output_path = os.path.join(output_folder, output_name)
        transcription_result = transcribe_with_groq(scene_path)
        save_to_srt(transcription_result, output_path)
        print(f"Transcription for {scene_name} saved to output file.")

