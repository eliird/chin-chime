import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from chin_chime.face_landmark_extractor import(
    save_landmarks, get_landmarks, process_video_all_frames
)
import os

videos_path = "/media/cv/Extreme Pro1/MERR/mer2023train/train"
csv_path = "/media/cv/Extreme Pro1/MERR/mer2023train/train-label.csv"

assert os.path.exists(csv_path)


def make_vid_path(filename, base_path):
    return os.path.join(base_path, filename + '.avi')

def load_csv(path, base_path):
    df = pd.read_csv(path)
    df['path'] = df['name'].apply(lambda x: make_vid_path(x, base_path))
    return df


if __name__ == "__main__":
    df = load_csv(csv_path, videos_path)
    df.head()
    
    # create the necessary directories
    saved_dataset_path = './dataset_augmented'
    os.makedirs(saved_dataset_path, exist_ok=True)
    for emotion in df['discrete'].unique().tolist():
        os.makedirs(os.path.join(saved_dataset_path, emotion), exist_ok=True)
        emotion_path = os.path.join(saved_dataset_path, emotion)
        os.makedirs(os.path.join(emotion_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(emotion_path, 'landmarks'), exist_ok=True)
        os.makedirs(os.path.join(emotion_path, 'landmarks_temporal'), exist_ok=True)
        
        
    failed_count = 0

    for i, row in df.iterrows():
        path = row['path']
        emotion = row['discrete']
        filename = row['name']
        try:
            process_video_all_frames(path, emotion, saved_dataset_path, num_imgs=10)
        except:
            failed_count += 1
            print(filename, failed_count)
        
        if i % 100 == 0:
            print("Completed: ", i*100/df.shape[0], "%")