## Chin Chime

This library uses the lower face of a person to detect the emotion.


# Model

The model is a simple MLP that takes the 63 landmarks for the lower face to detect the emotion. This model is 3 hidden layers of size 1024. Input layer is 63 x 3 for the x, y, z value of each of the 63 landmarks and outputs the 6 classes of emotion `Happy, Angry, Sad, Neutral, Surprised and Worried`.

The model is trained on the `MERR` dataset. The training code can be found in the `demo.ipynb` notebook.

# Using the model

`mediapipe_landmark.ipynb` demonstrates how you can use the model to make predictions from an image.

## Points to be careful about. 

The confidence of the face detection is set to 0.2 for the mediapipe library. So please ensure that the image you give to the model has only one face in it and the distance from the camera should not be that much. If distance is larger you can give a zoomed version of face and it seems to work.


# Installation
```
git clone https://github.com/eliird/chin-chime
pip install -e .

# edit the chin-chime/consants.py to update the path of the model weights
 >MODEL_PATH # mediapipe model path can be found in `model_weights/face_landmarker`
 
 >LANDMARK_LOWER_MODEL_PATH # path of the trained model

```

