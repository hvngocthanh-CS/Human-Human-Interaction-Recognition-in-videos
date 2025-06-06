#  Collaborative Multimodal Learning for Human-Human Interaction Recognition in Videos
This is the official code for our ICCCI paper.

## Getting Started
### Installation
Git clone our repository, creating a python environment and activate it via the following command:
```
git clone github.com/hvngocthanh-CS/Human-Human-Interaction-Recognition-in-videos
cd Human-Human-Interaction-Recognition-in-videos
conda create -n HHIR python=3.8
conda activate HHIR
pip install -r requirements.txt
```
### Dataset
The full dataset is available from the corresponding author upon reasonable request.

## Training
The source code and training scripts are available [here](https://github.com/hvngocthanh-CS/Human-Human-Interaction-Recognition-in-videos/tree/main/src). Model training and experiments are conducted on Kaggle.

## Testing

The testing scripts and evaluation (including F1 score calculation) are included in the same [source code repository](https://github.com/hvngocthanh-CS/Human-Human-Interaction-Recognition-in-videos/tree/main/src).  
All testing and result analysis were also performed on Kaggle.

## Demo
![](https://github.com/hvngocthanh-CS/Human-Human-Interaction-Recognition-in-videos/raw/main/asset/img2.png)

A demo video is available [here](https://drive.google.com/file/d/1AEByp5rkUe1wJWEViZE9I-vDpCbe18rt/view?usp=sharing), showing the model's predicted interaction labels for a single scene (corresponding to one video).
All actions recognized in the scene are displayed in the demo.

Additional sample videos for testing can be found [here](https://github.com/hvngocthanh-CS/Human-Human-Interaction-Recognition-in-videos/tree/main/scenes).

### Demo Overview

The demo video includes the following steps:
1. **Video Shot Segmentation:** The input video is segmented into multiple shots.
2. **Shot Merging:** Consecutive shots are merged to form clips, with every 3 consecutive shots grouped into a single clip (the number of shots per clip can be adjusted [here](https://github.com/hvngocthanh-CS/Human-Human-Interaction-Recognition-in-videos/blob/main/utils/utils.py#L26C56-L60C5)).
3. **Speech and Audio Extraction:** For each clip, Whisper-v3 is used to generate transcripts, and Pydub is used to extract audio features.
4. **Feature Extraction:** The resulting clips are processed by ViViT, BERT, and AST models to extract multimodal features.
5. **Action Prediction:** The extracted features are fed into the final model to predict interaction labels for each scene.
### How to run
#### Step 1: Training
Train the model to obtain the weights, then create a `./weights` directory and place the trained `.pth` file into this folder.

#### Step 2: Running the Demo
Update the environment variable at [this line](https://github.com/hvngocthanh-CS/Human-Human-Interaction-Recognition-in-videos/blob/main/utils/utils.py#L93) in `utils.py` to enable transcript extraction using Whisper.

To launch the demo, run:
```bash
streamlit run app.py
```
## License
This code is licensed under the MIT License. See [here](https://opensource.org/licenses/MIT) for details.
