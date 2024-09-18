## Chin Chime

This library uses the lower face of a person to detect the emotion.


# Model

The model is a simple MLP that takes the 63 landmarks for the lower face to detect the emotion. This model is 3 hidden layers of size 1024. Input layer is 63 x 3 for the x, y, z value of each of the 63 landmarks and outputs the 6 classes of emotion `Happy, Angry, Sad, Neutral, Surprised and Worried`.

## MER Dataset
The model trained on the `MERR` dataset. The training code can be found in the `merr.ipynb` notebook.
There are 3 models trained on this dataset.
 - All landmark model
 - Lower landmark model
 - Temporal landmark model

For landmarks we use the mediapipe to compute the landmarks. There are 478 landmarks in total. 
For lower face model we use the 63 landmarks below the nose. Check the notebook for the details of which landmarks are used.

## IEMOCAP Dataset
We tried computing the landmarks using the mediapipe model so we used the landmarks provided in the dataset.
Check the `iemocap_extract_landmark.ipynb` file to see how we extract landmarks and stored it like vision dataset in npy files.
There are 52 landmarks provided in the dataset and only 18 for the lower face which we use for emtoion detction.
We train two models all landmakrs and half face model using. Check the `iemocap.ipynb` for the details.
We also train an audio + lower face landmark model. We use audio for 2 seconds before the existing frame. Check the `iemocap_with_audio.ipynb` notebook for details.
For audio feature extraction we use pretrained wav2vec2 model.

## FER Dataset
Similar to MER we create two models after extracting landmarks using the mediapipe.
Check the `fer.ipynb` for training and resutls.

# Installation
```
git clone https://github.com/eliird/chin-chime
pip install -e .

# edit the chin-chime/consants.py to update the path of the model weights
 >MODEL_PATH # mediapipe model path can be found in `model_weights/face_landmarker`
 
 >LANDMARK_LOWER_MODEL_PATH # path of the trained model

```

# Using the model
`mediapipe_landmark.ipynb` demonstrates how you can use the model to make predictions from an image.
Or run the `run.py` file after installing to use the model with your camera.

## Points to be careful about. 

The confidence of the face detection is set to 0.2 for the mediapipe library. So please ensure that the image you give to the model has only one face in it and the distance from the camera should not be that much. If distance is larger you can give a zoomed version of face and it seems to work.




