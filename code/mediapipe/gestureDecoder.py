from sklearn.metrics import accuracy_score  # Accuracy metrics
import pickle
import pyvirtualcam
from pyvirtualcam import PixelFormat
from PIL import Image
import webcamPNG as png
import cv2  # Import opencv
import mediapipe as mp  # Import mediapipe
import numpy as np
import pandas as pd
import time

# open the sample PNG (question.png)
sampleImg = png.loadIMG('.\\effect\\questionSmall.png')

# mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_holistic = mp.solutions.holistic  # Mediapipe Solutions

# Load Model
with open('.//code//mediapipe//body_language.pkl', 'rb') as f:
    model = pickle.load(f)

### Video Capture for Window
cap = cv2.VideoCapture(0)
# ### Video Capture for Mac
# cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)
prevTime = 0
readData = True

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = cap.get(cv2.CAP_PROP_FPS)
# cap = cv2.flip(originalCap, 1)


# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    with pyvirtualcam.Camera(width, height, fps_in, fmt=PixelFormat.BGR, print_fps=fps_in) as cam:
        print(f'Virtual cam started: {cam.device} ({cam.width}x{cam.height} @ {cam.fps}fps)')

        while cap.isOpened():
            ret, frame = cap.read()

            # Start point: Timer for make dataframe
            if readData:
                beginTime = time.time()
                readData = False
            
            ### Start point: Timer for Debugging
            initialTime = time.time()

            
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            image.flags.writeable = False        
            
            # Make Detectionsq
            results = holistic.process(image)
                    
            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            ### First point (Debugging)
            firstTime = time.time() - initialTime
            cv2.putText(image, f'1: {round(firstTime,3)}', (20,150), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 1)

            
            # Export coordinates
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                
                # Concate rows
                row = pose_row+face_row
                
                ### Second point (Debugging)
                secondTime = time.time() - initialTime
                cv2.putText(image, f'2: {round(secondTime,3)}', (20,200), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 1)
                
    #             # Make Detections
    #             X = pd.DataFrame([row])
                
    #             # Make Prediction
    #             body_language_class = model.predict(X)[0]
    #             body_language_prob = model.predict_proba(X)[0]
                
                # 100ms (Making Dataframe & Predict)
                duration = time.time() - beginTime
                if duration > 0.5:
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    readData = True
                
                ### Third point (Debugging)
                thirdTime = time.time() - initialTime
                cv2.putText(image, f'3: {round(thirdTime,3)}', (20,250), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 1)

                # Add png image
                if body_language_class == 'Question':
                    # show png file of question mark
                    pilim = Image.fromarray(image)
                    pilim.paste(sampleImg, box=(0, 500), mask=sampleImg)
                    image = np.array(pilim)
                
                ### Fourth point (Debugging)
                fourthTime = time.time() - initialTime
                cv2.putText(image, f'4: {round(fourthTime,3)}', (20,300), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 1)
                

                # Get status box
                cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                
                # Display Class
                cv2.putText(image, 'CLASS'
                            , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0]
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(image, 'PROB'
                            , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                            , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
            
            except:
                pass
            
            # Calculate FPS
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            cv2.putText(image, f'FPS: {round(fps,3)}', (20,100), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 1)
            
            # send video to virtualCam
            cam.send(image)
            cam.sleep_until_next_frame()

            ### Fifth point (Debugging)
            fifthTime = time.time() - initialTime
            cv2.putText(image, f'5: {round(fifthTime,3)}', (20,350), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 1)

            # show video with pop-up screen
            cv2.imshow('Interactive Zoom Demo', image)


            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()