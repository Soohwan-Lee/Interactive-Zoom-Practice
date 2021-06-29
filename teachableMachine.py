# import numpy as np
# import cv2

# # 영상 촬영 장치와 연결하기
# capture = cv2.VideoCapture(0)

# # 영상의 Width와 Height 크기조절
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


# while True:
#     if cv2.waitKey(10) > 0:
#         break

#     ret, frame = capture.read()
#     # 영상을 한 프레임씩 읽어온다.
#     # ret:프레임 제대로 읽었는지 확인,정상이면 True이 출력된다.
#     # frame: 읽은 프레임이 출력됨(이미지)

#     cv2.putText(frame,'test',(0,25), cv2.FONT_HERSHEY_PLAIN,30,(0,0,0))
#     #영상에 텍스트를 삽입한다.(넣을영상,넣을텍스트,텍스트위치(x,y),폰트명,폰트크기,색상(R,G,B))
#     cv2.imshow("camera test", frame)
#     #영상을 출력한다.


#################################
## cuDNN 버전 문제가 있는 것 같음##
#################################

import tensorflow.keras
import numpy as np
import cv2

# 모델 위치
model_filename = 'C:\\Users\\LeeSooHwan\\Desktop\\Interactive-Zoom-Practice\\keras_model.h5'

# 케라스 모델 가져오기
model = tensorflow.keras.models.load_model(model_filename)

# 카메라를 제어할 수 있는 객체
capture = cv2.VideoCapture(0)

# 카메라 길이 너비 조절
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# 이미지 처리하기


def preprocessing(frame):
    #frame_fliped = cv2.flip(frame, 1)
    # 사이즈 조정 티쳐블 머신에서 사용한 이미지 사이즈로 변경해준다.
    size = (224, 224)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    # 이미지 정규화
    # astype : 속성
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1

    # 이미지 차원 재조정 - 예측을 위해 reshape 해줍니다.
    # keras 모델에 공급할 올바른 모양의 배열 생성
    frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))
    # print(frame_reshaped)
    return frame_reshaped

# 예측용 함수


def predict(frame):
    prediction = model.predict(frame)
    return prediction


while True:
    ret, frame = capture.read()

    if cv2.waitKey(100) > 0:
        break

    preprocessed = preprocessing(frame)
    prediction = predict(preprocessed)

    if (prediction[0, 0] < prediction[0, 1]):
        # print('Question')
        cv2.putText(frame, 'Question', (0, 25),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))

    else:
        cv2.putText(frame, 'Background', (0, 25),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        # print('Background')

    cv2.imshow("Interactive Zoom", frame)
    # cv.imshow()
