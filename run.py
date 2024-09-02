import cv2
from chin_chime.face_landmark_extractor import (
    get_mp_detecion, draw_landmarks_on_image, plot_face_blendshapes_bar_graph
)
from chin_chime.realtime_detection import run_emotion_detction

# Window name in which image is displayed
window_name = 'Image'

# font
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (50, 50)

# fontScale
fontScale = 1
 
# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2
 
 
def main():
    cap = cv2.VideoCapture(0) #  webcam
    
    skipped_frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Skipping frame, {}".format(skipped_frame_count))
            skipped_frame_count += 1
        
        emotion = run_emotion_detction(frame)
        
        if emotion is not None:
            # Position for the text
            x, y = frame.shape[1] - 250, 20  # Starting position on the top right

            # Write each line on the frame
            for i, (emotion, probs) in enumerate(emotion.items()):
                cv2.putText(
                    frame, 
                    f'{emotion}: {probs}', 
                    (x, y + i * 30), 
                    font, 
                    fontScale=fontScale, 
                    color=color, 
                    thickness=thickness
                )
            
        cv2.imshow("Camera Feed", frame)
        
        if cv2.waitKey(5) == ord('q'):
            break
        
    pass

if __name__ == '__main__':
    main()