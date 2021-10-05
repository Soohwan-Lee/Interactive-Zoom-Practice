import pickle
import pyvirtualcam
from pyvirtualcam import PixelFormat
from PIL import Image
import webcamPNG as png
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import os.path

# Load Pose_Hand_Images with Array
pose_hand_imgs = []
path = ".//effect//final_pose_hand//"
fileType = '.png'
fileList = os.listdir(path)
fileList.sort()
for i in range(0, 12):
    pose_hand_imgs.append(Image.open(path+str(i)+fileType))

# Load Face_Images with Array
face_imgs = []
path = ".//effect//final_face//"
fileType = '.png'
fileList = os.listdir(path)
fileList.sort()
for i in range(0, 3):
    face_imgs.append(Image.open(path+str(i)+fileType))

# Set the directory path
path_rawData = ".//code//mediapipe//rawData"
path_model = ".//code//mediapipe//model"

# Set the model name
pose_hand_model = "//210916pose_hand.pkl"
face_model = "//210916face.pkl"

# Load Model
with open(path_model + pose_hand_model, 'rb') as f:
    pose_hand_model = pickle.load(f)
with open(path_model + face_model, 'rb') as f:
    face_model = pickle.load(f)

# Set the coords number
num_pose_coords = 22
num_right_hand_coords = 20
num_left_hand_coords = 20
num_pose_hand_coords = num_pose_coords + num_right_hand_coords + num_left_hand_coords
num_face_coords = 467

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_holistic = mp.solutions.holistic  # Mediapipe Solutions

# Video Capture for Window
cap = cv2.VideoCapture(0)
# ### Video Capture for Mac
# cap = cv2.VideoCapture(1)

# Set the Frame Size
cap.set(3, 1280)
cap.set(4, 720)

# Variable for calculating FPS
prevTime = 0

# Boolean variables for Reading Data
read_pose_hand = True
read_face = True

# Boolean variables for Existing of Right & Left Hands
right_hand = True
left_hand = True

# Set default threshold value
threshold_pose_hand = 0.6
threshold_face = 0.8

# Time interval of Reading Data for pose_hand & face
timeInterval = 0.2

# Initialize global variable for class
pose_hand_class = 0
face_class = 0

# Get Frame's width & height, fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = cap.get(cv2.CAP_PROP_FPS)

# Get average data in real-time
def cumulativeAverage(prevAvgArray, newArray, listLength):
    if listLength > 0:
        oldWeight = (listLength - 1) / listLength
        newWeight = 1 / listLength
        avg = (prevAvgArray * oldWeight) + (newArray * newWeight)
    return avg

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    with pyvirtualcam.Camera(width, height, fps_in, fmt=PixelFormat.BGR, print_fps=fps_in) as cam:
        print(f'Virtual cam started: {cam.device} ({cam.width}x{cam.height} @ {cam.fps}fps)')

        while cap.isOpened():
            ret, frame = cap.read()

            # Start point: Timer for recieving pose_hand_data
            if read_pose_hand:
                beginTime_pose_hand = time.time()
                listLength_pose_hand = 0
                read_pose_hand = False

            # Start point: Timerr for recieving face_data
            if read_face:
                beginTime_face = beginTime_pose_hand + timeInterval
                listLength_face = 0
                read_face = False

            # Start point: Timer for Debugging
            initialTime = time.time()

            # Flip Image
            image = cv2.flip(frame, 1)
            image.flags.writeable = False

            # Make Detectionsq
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # --------------------------------------------------------------------------------------------- #
            # 1. Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                    )
            
            # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    )

            # 3. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                    )

            # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )
            # --------------------------------------------------------------------------------------------- #

            # Export Pose-Hand coordinates
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = np.array([[pose[i].x-pose[0].x, pose[i].y-pose[0].y] 
                            for i in range(1, num_pose_coords+1)]).flatten()
                # Extract "RIGHT" Hand lanmarks
                try:
                    righthand = results.left_hand_landmarks.landmark
                    righthand_row = np.array([[righthand[i].x - righthand[0].x, righthand[i].y - righthand[0].y]
                                for i in range(1,num_right_hand_coords+1)]).flatten()
                    right_hand = True
                except:
                    righthand_row = [0 for i in range(num_right_hand_coords*2)]
                    right_hand = False

                # Extract "LEFT" Hand lanmarks
                try:
                    lefthand = results.right_hand_landmarks.landmark
                    lefthand_row = np.array([[lefthand[i].x - lefthand[0].x, lefthand[i].y - lefthand[0].y]
                                                    for i in range(1,num_left_hand_coords+1)]).flatten()
                    left_hand = True
                except:
                    lefthand_row = [0 for i in range(num_left_hand_coords*2)]
                    left_hand = False

                # Concate rows
                pose_hand_row = np.concatenate((pose_row, righthand_row, lefthand_row), axis=None)
                
                # Average data
                listLength_pose_hand = listLength_pose_hand + 1
                if listLength_pose_hand == 1:
                    pose_hand_row_avg = pose_hand_row
                elif listLength_pose_hand > 1:
                    pose_hand_row_avg = cumulativeAverage(pose_hand_row_avg, pose_hand_row, listLength_pose_hand)
                    

                # Make prediction for "timeInterval" sec
                duration = time.time() - beginTime_pose_hand
                if right_hand == False and left_hand == False:
                    pose_hand_class = 0
                else:
                    if duration > timeInterval:
                        # print("pose_hand")
                        list(pose_hand_row_avg)
                        pose_hand_class = pose_hand_model.predict([pose_hand_row_avg])[0]
                        pose_hand_prob = pose_hand_model.predict_proba([pose_hand_row_avg])[0]
                        read_pose_hand = True
                        if float(pose_hand_prob[np.argmax(pose_hand_prob)]) < threshold_pose_hand:
                            pose_hand_class = 0

                # # Add png image
                # pilim = Image.fromarray(image)
                # pilim.paste(pose_hand_imgs[pose_hand_class], box=(0, 500), mask=pose_hand_imgs[pose_hand_class])
                # image = np.array(pilim)
                        
                # Get status box
                cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                
                # Display Class
                cv2.putText(image, 'CLASS'
                            , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(pose_hand_class)
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(image, 'PROB'
                            , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(pose_hand_prob[np.argmax(pose_hand_prob)],2))
                            , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            except:
                pass
            
            if pose_hand_class == 0:
                # Export Face coordinates
                try:
                    # Extract Face landmarks
                    face = results.face_landmarks.landmark
                    face_row = np.array([[face[i].x - face[0].x, face[i].y - face[0].y]
                                                        for i in range(1,num_face_coords+1)]).flatten()

                    # Average data
                    listLength_face = listLength_face + 1
                    if listLength_face == 1:
                        face_row_avg = face_row
                    elif listLength_face > 1:
                        face_row_avg = cumulativeAverage(face_row_avg, face_row, listLength_face)

                    # Make prediction for "timeInterval" sec
                    duration = time.time() - beginTime_face
                    if duration > timeInterval:
                        # print("face")
                        list(face_row_avg)
                        face_class = face_model.predict([face_row])[0]
                        face_prob = face_model.predict_proba([face_row])[0]
                        read_face = True
                        if float(face_prob[np.argmax(face_prob)]) < threshold_face:
                                face_class = 0
                    
                    # Add png image
                    pilim = Image.fromarray(image)
                    pilim.paste(face_imgs[face_class], box=(1080, 500), mask=face_imgs[face_class])
                    image = np.array(pilim)
                    
                    # Get status box
                    cv2.rectangle(image, (250,0), (500, 60), (245, 117, 16), -1)
                    
                    # Display Class
                    cv2.putText(image, 'CLASS'
                                , (345,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(face_class)
                                , (340,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Display Probability
                    cv2.putText(image, 'PROB'
                                , (265,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(face_prob[np.argmax(face_prob)],2))
                                , (260,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                except:
                    pass

            # Calculate FPS
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            cv2.putText(image, f'FPS: {round(fps,3)}', (20, 100),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

            # convert color BRG to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # send video to virtualCam
            cam.send(image)
            cam.sleep_until_next_frame()

            # show video with pop-up screen
            cv2.imshow('Interactive Zoom Demo', image)

            # Press 'q' to quit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()