import os
from chin_chime.face_landmark_extractor import get_landmarks, save_landmarks
import numpy as np


def test_landmarks():
    file_path = '/home/cv/workspace/eliird/chin-chime/test_face.jpg'
    save_folder = '/home/cv/workspace/eliird/chin-chime/dummy_folder/'

    landmarks = get_landmarks(file_path)
    save_landmarks(file_path, save_folder)

    file_name = file_path.split('/')[-1].split('.')[0]
    save_path = os.path.join(save_folder, file_name + '.npy')

    saved_landmarks = np.load(save_path)
    assert np.array_equal(landmarks, saved_landmarks)