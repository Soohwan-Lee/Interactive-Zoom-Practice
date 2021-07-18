#################################
## cuDNN 버전 문제가 있는 것 같음##
#################################

import tensorflow.keras
import numpy as np
import cv2
import webcamPNG as png
from PIL import Image
import pyvirtualcam
from pyvirtualcam import PixelFormat


def loadModel(model_filename):
    # 케라스 모델 가져오기
    model = tensorflow.keras.models.load_model(model_filename)
    return model


def captureCamera():
    # 카메라를 제어할 수 있는 객체
    cap = cv2.VideoCapture(0)

    pref_width = 1280
    pref_height = 720
    pref_fps_in = 30
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, pref_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, pref_height)
    cap.set(cv2.CAP_PROP_FPS, pref_fps_in)

    return cap


def preprocessing(frame):
    # 이미지 처리하기
    frame_fliped = cv2.flip(frame, 1)
    # 사이즈 조정 티쳐블 머신에서 사용한 이미지 사이즈로 변경해준다.
    size = (224, 224)
    frame_resized = cv2.resize(
        frame_fliped, size, interpolation=cv2.INTER_AREA)

    # 이미지 정규화
    # astype : 속성
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1

    # 이미지 차원 재조정 - 예측을 위해 reshape 해줍니다.
    # keras 모델에 공급할 올바른 모양의 배열 생성
    frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))
    # print(frame_reshaped)
    return frame_reshaped


def predict(model, frame):
    # 예측용 함수
    prediction = model.predict(frame)
    return prediction


def teachableMachine(capture):
    # video preprocessing, predict by model,
    with pyvirtualcam.Camera(width, height, fps_in, fmt=PixelFormat.BGR, print_fps=fps_in) as cam:
        print(
            f'Virtual cam started: {cam.device} ({cam.width}x{cam.height} @ {cam.fps}fps)')

        while True:
            ret, frame = capture.read()

            if cv2.waitKey(100) > 0:
                break

            preprocessed = preprocessing(frame)
            prediction = predict(model, preprocessed)
            frame_fliped = cv2.flip(frame, 1)

            fontScale = 2
            fontORG = (0, 450)
            fontColor = (0, 0, 256)  # B G R (A)
            fontThickness = 2

            if (prediction[0, 0] < prediction[0, 1]):
                # print('Question')
                cv2.putText(frame_fliped, 'Question', fontORG,
                            cv2.FONT_HERSHEY_PLAIN, fontScale, fontColor, fontThickness)

                # show png file of question mark
                pilim = Image.fromarray(frame_fliped)
                pilim.paste(img, box=(0, 20), mask=img)
                frame_fliped = np.array(pilim)

            else:
                cv2.putText(frame_fliped, 'Background', fontORG,
                            cv2.FONT_HERSHEY_PLAIN, fontScale, fontColor, fontThickness)
                # print('Background')

            # # Show webcam video through additional window
            # cv2.imshow("Interactive Zoom", frame_fliped)

            # Send webcam video through virtual cam
            cam.send(frame_fliped)
            # Wait until it's time for the next frame.
            cam.sleep_until_next_frame()


if __name__ == "__main__":
    model = loadModel('..\\model\\keras_model.h5')
    img = png.loadIMG('..\\effect\\questionSmall.png')
    capture = captureCamera()

    # Query final capture device values (may be different from preferred settings).
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = capture.get(cv2.CAP_PROP_FPS)
    print(f'Webcam capture started ({width}x{height} @ {fps_in}fps)')
    fps_out = 20

    teachableMachine(capture)
