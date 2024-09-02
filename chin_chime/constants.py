# mediapipe model path
MODEL_PATH =\
    '/home/cv/workspace/eliird/chin-chime/model_weights/face_landmarker.task'

LANDMARK_MODEL_PATH =\
    '/home/cv/workspace/eliird/chin-chime/model_weights/landmark_emotion.py'
    

LANDMARK_LOWER_MODEL_PATH =\
    '/home/cv/workspace/eliird/chin-chime/model_weights/landmark_lower_emotion.pt'
    
DEVICE = 'cuda'

MIN_CONFIDENCE = 0.5


LOWER_FACE_LANDMARKS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    61, 62, 63, 64, 65, 66, 67, 68, 75, 76, 77, 78, 79, 80, 81, 82, 87, 88, 89,
    90, 91, 127, 128, 129, 130, 131, 132, 134, 135, 136, 147, 148, 152, 191, 
    192, 193, 194, 195, 291, 317, 318, 324
]

EMOTION2ID = {
    'angry': 0, 
    'happy': 1,
    'neutral': 2,
    'sad': 3,
    'surprise': 4,
    'worried': 5
    }

ID2EMOTION = {
    0: 'angry',
    1: 'happy',
    2: 'neutral',
    3: 'sad',
    4: 'surprise',
    5: 'worried'
} 

EMOTIONS = [
    'angry',
    'happy',
    'neutral',
    'sad',
    'surprise',
    'worried'
]