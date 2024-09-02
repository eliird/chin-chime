import torch
from .model import MLPLandmark
from .face_landmark_extractor import (
    get_mp_detecion, draw_landmarks_on_image, get_landmarks
)
from .constants import (
    LOWER_FACE_LANDMARKS, 
    LANDMARK_MODEL_PATH,
    LANDMARK_LOWER_MODEL_PATH,
    DEVICE, ID2EMOTION, EMOTION2ID, EMOTIONS
    )
from typing import Tuple
from numpy import ndarray


INP_LANDMARKS = 63
DIM_LANDMARK = 3
EMOTION_CLASSES = 6
HIDDEN_LAYERS = [1024, 1024, 1024]

model = MLPLandmark(
    inp_landmarks=INP_LANDMARKS,
    dim_per_landmark=DIM_LANDMARK,
    layers=HIDDEN_LAYERS,
    out_dim=EMOTION_CLASSES
)

model.load_state_dict(torch.load(LANDMARK_LOWER_MODEL_PATH))
model.to(DEVICE)


def run_emotion_detction(image:ndarray) -> dict:
    landmarks = torch.tensor(get_landmarks(image), dtype=torch.float32)
    lower_landmarks = landmarks[LOWER_FACE_LANDMARKS]
    lower_landmarks = lower_landmarks.unsqueeze(0).to(DEVICE)
    
    logits = model(lower_landmarks)[0]
    probs = torch.softmax(logits, dim=0).detach().cpu().tolist()
    emotion_prob = {
        e:p for e, p in zip(EMOTIONS, probs)
    }
    return emotion_prob
    
    
    
