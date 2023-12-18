import dlib
import cv2
import numpy as np
from scipy.spatial import distance 
import time

# 함수 정의
def compute_EAR(vec):
    a = distance.euclidean(vec[1], vec[5])
    b = distance.euclidean(vec[2], vec[4])
    c = distance.euclidean(vec[0], vec[3])
    ear = (a + b) / (2.0 * c)
    return ear

# 파일 경로 및 카메라 설정
predictor_path = "/home/raul/catkin_ws/src/sleep-detection/data"
camera_path = "/dev/video0"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

cap = cv2.VideoCapture(camera_path)
#추가 코드
start_time = None
sleeping = False

while True:
    ret, frame = cap.read()   # 확인 필요한 부분          -->  웹 캠에서 필요한 프레임 값을 가져오는 함수
    if not ret:               # ret은 부울 변수 값임 잘 읽었는지 아닌지 
        print("Failed to capture frame")     
        break

    dets = detector(frame, 1)                 
    vec = np.empty([68, 2], dtype=int)

    status = "Not Sleeping"

    for k, d in enumerate(dets):
        shape = predictor(frame, d)

        for b in range(68):
            vec[b][0] = shape.part(b).x
            vec[b][1] = shape.part(b).y

        right_ear = compute_EAR(vec[42:48])
        left_ear = compute_EAR(vec[36:42])

        if (right_ear + left_ear) / 2 < 0.2:
            if not sleeping:
                start_time = time.time()
                sleeping = True
            else:
                if time.time() - start_time > 3:  # 3초 동안 눈을 감았을 때
                    status = "Sleeping"
        else:
            sleeping = False

    print(status)

    if status == "Sleeping":
        frame = cv2.putText(frame, "Sleeping", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        frame = cv2.putText(frame, "Not Sleeping", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Webcam", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
