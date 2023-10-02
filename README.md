# 使用Mediapipe分析動作
MediaPipe 是 Google Research 所開發的多媒體機器學習模型應用框架，透過 MediaPipe，可以簡單地實現手部追蹤、人臉檢測或物體檢測等功能，這篇教學將會介紹如何使用 MediaPipe。

如果使用 Python 語言進行開發，MediaPipe 支援下列幾種辨識功能：

**MediaPipe Face Detection ( 人臉追蹤 ) 
MediaPipe Face Mesh ( 人臉網格 )
MediaPipe Hands ( 手掌偵測 )
MediaPipe Holistic ( 全身偵測 )
MediaPipe Pose ( 姿勢偵測 )
MediaPipe Objectron ( 物體偵測 )
MediaPipe Selfie Segmentation ( 人物去背 )**

本次教學將會著重在姿勢偵測、全身偵測及人臉網格。
## 安裝能夠執行Mediapipe的Python環境
打開網址：
https://www.anaconda.com/download
![](https://hackmd.io/_uploads/SkcluP0yp.png)
點選Download即會自動下載，安裝完成後打開.exe檔開始安裝。
安裝完成後，從電腦左下角的地方搜尋Jupyter Notebook並開啟。
![](https://hackmd.io/_uploads/HJUa_DC1T.png)
會跳出一個黑色視窗並開啟一個網頁
![](https://hackmd.io/_uploads/B1u8tv0yp.png)
![](https://hackmd.io/_uploads/SJWDYv0Ja.png)
若成功進入到網頁中，代表環境開啟成功。
### 開啟一個新的專案
![](https://hackmd.io/_uploads/H1B1cwCkT.png)
點選右邊的new後會有一個Pyhton3並點選。
![](https://hackmd.io/_uploads/Byi45v0ya.png)
若進入此畫面代表專案開啟成功。
## 獲取程式碼
前往網址：
> https://github.com/shiayongshen/Warm-Up_Exercise_Detection_System_based_on_LSTM_and_Mediapipe

並點選綠色code按鈕，選取 Download ZIP：
![](https://hackmd.io/_uploads/BkHBoe1e6.png)

下載完成後請解壓縮檔案至`C:\User\users`。
#### 注意：每台電腦User存放位置不同，若有問題請舉手提問。
## 測試Mediapipe是否能夠正常執行
請打開Jupyter notebook並進入到剛剛解壓縮的資料夾，名稱為`Warm-Up_Exercise_Detection_System_based_on_LSTM_and_Mediapipe-main`再進到`detect`資料夾中開啟一個新的專案，並先執行：
```
!pip install opencv-python
!pip install mediapipe
!pip install numpy
!pip install tensorflow
```
並等待執行完成，執行完成後接續執行：
```
import cv2
import mediapipe as mp
import numpy as np #載入套件
```
`import` 為Python 載入套件的基本語法，執行後Jupyter notebook會將所需套件載入至環境中。
執行完成後執行：
```
mp_drawing = mp.solutions.drawing_utils         # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles # mediapipe 繪圖樣式
mp_holistic = mp.solutions.holistic             # mediapipe 全身偵測方法
```
這三行為Mediapipe套件之執行方式，分別代表繪圖方法、繪圖樣式與全身偵測方式。
其中第三行能針對不同領域更換不同的變數。
```
#若要偵測姿勢則改成
mp_pose = mp.solutions.pose
#若要偵測手部則改成
mp_hands = mp.solutions.hands
```
接者要打開設攝影機使用mediapipe偵測：
```
cap = cv2.VideoCapture(0)#開啟攝影機
#需先確定裝備是否有內建攝影機，若有則不須變動，若沒有，請先插上攝影機後再執行

# mediapipe 啟用偵測全身
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()  #若無法開啟攝影機則退出
    while True: #可以打開攝影機
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break #若無法收到畫面則退出
        img = cv2.resize(img,(640,480))               #將攝影機畫面設定大小為(640,480)
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        results = holistic.process(img2)              # 開始偵測全身
        # 身體偵測，繪製身體骨架
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
        cv2.imshow('warmup', img) #視窗名稱
        if cv2.waitKey(5) == ord('q'):
            break       #按下 q 鍵停止
cap.release()           #釋放資源
cv2.destroyAllWindows() #刪除視窗
```
### 完整程式碼
```
import cv2
import mediapipe as mp
import numpy as np 

mp_drawing = mp.solutions.drawing_utils         
mp_drawing_styles = mp.solutions.drawing_styles 
mp_holistic = mp.solutions.holistic 

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()  
    while True: 
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break 
        img = cv2.resize(img,(640,480))               
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
        results = holistic.process(img2)              
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
        cv2.imshow('warmup', img) 
        if cv2.waitKey(5) == ord('q'):
            break       
cap.release()           
cv2.destroyAllWindows() 
```

## 如何提取骨架做角度運算
以使用Pose(姿勢預測)為例，則會顯示出33個點，其點Index如下圖：
![](https://hackmd.io/_uploads/HkTAoDY1T.png)
而每個點之座標格式如下：
```
x: 0.25341567397117615
y: 0.71121746301651
z: -0.03244325891137123
```
由上可知，Mediapipe能夠預測出三維的向量，代表Mediapipe有偵測出空間向量的能力。

### 程式碼講解

而要計算出骨架的角度，則需選出連接之三個點，並透過這三個點來做角度運算，其程式碼範例如下：
```
def get_knee_angle(landmarks):
    r_hip = get_landmark(landmarks, "RIGHT_HIP")
    l_hip = get_landmark(landmarks, "LEFT_HIP")

    r_knee = get_landmark(landmarks, "RIGHT_KNEE")
    l_knee = get_landmark(landmarks, "LEFT_KNEE")

    r_ankle = get_landmark(landmarks, "RIGHT_ANKLE")
    l_ankle = get_landmark(landmarks, "LEFT_ANKLE")

    r_angle = calc_angles(r_hip, r_knee, r_ankle)
    l_angle = calc_angles(l_hip, l_knee, l_ankle)

    return [r_angle, l_angle]
    
def get_landmark(landmarks, part_name):
    return [
        landmarks[mppose.PoseLandmark[part_name].value].x,
        landmarks[mppose.PoseLandmark[part_name].value].y,
        landmarks[mppose.PoseLandmark[part_name].value].z,
    ]
    
def calc_angles(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle
```
在`get_knee_angle`函式中，先呼叫了`get_landmark`函式，並傳回一個陣列中包含的該位置的x,y,z座標，並求出要運算之三點座標後，呼叫`calc_angles`函式來運算角度，在`calc_angles`函式中，使用`arctan2()`函式來進行運算，其得到之值為兩點之弧度，而求出兩個弧度後再作相減，再乘以`pi`，即可得到三點連線之角度。
 
## 透過計算角度判斷目前姿勢
目前mediapipe主流為透過骨架之角度來判斷人體目前的狀態動作，而本次將以小腿拉伸作為範例。
本次教學參考Youtube上小腿拉伸影片，並擷取骨架來使用，透過影片可以看到要達到小腿拉伸的正確動作必須要為其中一支腳為115度至140度，另外一隻腳為160度至180度，因此將此做為判斷標準，只要有達到螢幕上就會顯示綠燈，並開始倒數30秒。
### 程式碼講解
先載入下方套件：
```
import cv2
import numpy as np
import mediapipe as mp
import time
```
由於要偵測動作，因此設定mediapipe的pose功能，並先初始化參數：
```
mppose = mp.solutions.pose
pose = mppose.Pose()
h = 0                   #int
w = 0                   #int
status = False          #bool
countdown_seconds = 30  #int
start_time = 0          #int
```
定義所需函數，上面已講解過。
```
def calc_angles(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


def get_landmark(landmarks, part_name):
    return [
        landmarks[mppose.PoseLandmark[part_name].value].x,
        landmarks[mppose.PoseLandmark[part_name].value].y,
        landmarks[mppose.PoseLandmark[part_name].value].z,
    ]


def get_visibility(landmarks):
    if landmarks[mppose.PoseLandmark["RIGHT_HIP"].value].visibility < 0.8 or \
            landmarks[mppose.PoseLandmark["LEFT_HIP"].value].visibility < 0.8:
        return False
    else:
        return True
def get_knee_angle(landmarks):
    r_hip = get_landmark(landmarks, "RIGHT_HIP")
    l_hip = get_landmark(landmarks, "LEFT_HIP")

    r_knee = get_landmark(landmarks, "RIGHT_KNEE")
    l_knee = get_landmark(landmarks, "LEFT_KNEE")

    r_ankle = get_landmark(landmarks, "RIGHT_ANKLE")
    l_ankle = get_landmark(landmarks, "LEFT_ANKLE")

    r_angle = calc_angles(r_hip, r_knee, r_ankle)
    l_angle = calc_angles(l_hip, l_knee, l_ankle)

    return [r_angle, l_angle]
```
接續進入主程式，而將使用到opencv作為影像處理，其顯示內容為左上角為示範骨架，左下角顯示目前角度：
```
video1 = cv2.VideoCapture('output_video.mp4')                    #設定video1開啟物件路徑
video1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'XVID')) #指定讀取格式為XVID
video1.set(cv2.CAP_PROP_FPS, 30)                                 #設定幀率
video1.set(cv2.CAP_PROP_POS_FRAMES, 0)                           #設定物件初始化

video2 = cv2.VideoCapture(0)                                     #設定video2開啟物件路徑

if not video1.isOpened() or not video2.isOpened():
    print("無法打開影片")
    exit()             #若無法打開則退出

frame_width2 = 1280    #設定攝影機螢幕寬
frame_height2 = 960    #設定攝影機螢幕高
print(frame_height2)   #960
print(frame_width2)    #1280

scaled_width = frame_width2 // 6     #骨架影片寬為主螢幕之1/6
scaled_height = frame_height2 // 6   #骨架影片高為主螢幕之1/6

cv2.namedWindow('Combined Video', cv2.WINDOW_NORMAL)             #設定視窗名字
cv2.resizeWindow('Combined Video', frame_width2, frame_height2)  #設定視窗大小

while True:                               #持續偵測
    if not status:                        #若status為False
        start_time = time.time()          #設定開始時間
    ret1, frame1 = video1.read()          #讀取骨架影片
    ret2, frame2 = video2.read()          #讀取攝影機

    if not ret1:                                  #若骨架影片撥放完畢
        video1.set(cv2.CAP_PROP_POS_FRAMES, 0)    #初始化影片
        continue                                  #繼續執行

    if not ret2:                                  #若讀取不到攝影機畫面
        break                                     #跳出迴圈
    results = pose.process(frame2)                #攝影機讀取骨架結果
    if results.pose_landmarks is not None:        #若有讀取到骨架
        mp_drawing = mp.solutions.drawing_utils   #繪圖方式
        annotated_image = frame2.copy()           #複製攝影到的幀
        mp_drawing.draw_landmarks(                #於複製的畫面上作畫
        annotated_image, results.pose_landmarks, mppose.POSE_CONNECTIONS)
        knee_angles = get_knee_angle(results.pose_landmarks.landmark)          #計算角度                     
        if 115<knee_angles[0] < 140 and 160<knee_angles[1]< 180:               #符合標準
            cv2.putText(annotated_image, "Left: {:.1f}".format(knee_angles[0]), (10, 230)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated_image, "Right: {:.1f}".format(knee_angles[1]), (10, 260)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated_image, "Good! Keep going!", (10, 290)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            #在複製的畫面上印出文字，分別為左腳右腳之角度，並顯示綠色
            status= True     #令status為True
        elif 115<knee_angles[1] < 140 and  160<knee_angles[0]< 180:            #符合標準
            cv2.putText(annotated_image, "Left: {:.1f}".format(knee_angles[0]), (10, 230)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated_image, "Right: {:.1f}".format(knee_angles[1]), (10, 260)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA) 
            cv2.putText(annotated_image, "Good! Keep going!", (10, 290)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)               #在複製的畫面上印出文字，分別為左腳右腳之角度，並顯示綠色
            status= True    #令status為True
        else:                                                                  #不符合標準
            cv2.putText(annotated_image, "Left: {:.1f}".format(knee_angles[0]), (10, 230)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_image, "Right: {:.1f}".format(knee_angles[1]), (10, 260)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_image, "Squat down!", (10, 290)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA) 
            #在複製的畫面上印出文字，分別為左腳右腳之角度，並顯示紅色
            status= False    #令status為False            
        if status:                                                      #若status為True
            current_time = time.time()                                  #取得目前時間
            elapsed_time = int(current_time - start_time)               #目前時間與開始時間之時間差
            remaining_seconds = max(0, countdown_seconds - elapsed_time)#算出剩餘時間

            text = f" {remaining_seconds} second"
            cv2.putText(annotated_image, text, (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)                                                      #印出剩餘時間

            if remaining_seconds == 0:                                  #若剩餘時間歸0
                status = False                                          #令status為False

    frame1_resized = cv2.resize(frame1, (scaled_width, scaled_height))  #將骨架影片縮小至指定尺寸
    annotated_image[0:scaled_height, 0:scaled_width] = frame1_resized   #複製幀中放入骨架影片
    cv2.imshow('Combined Video', annotated_image)                       #將複製幀顯示出來
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break            #按q退出
video1.release()         #釋放資源
video2.release()         #釋放資源
cv2.destroyAllWindows()  #釋放畫面
```
### 完整程式碼
```
import cv2
import numpy as np
import mediapipe as mp
import time
mppose = mp.solutions.pose
pose = mppose.Pose()
h = 0
w = 0
status = False
countdown_seconds = 30
start_time = 0

def calc_angles(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


def get_landmark(landmarks, part_name):
    return [
        landmarks[mppose.PoseLandmark[part_name].value].x,
        landmarks[mppose.PoseLandmark[part_name].value].y,
        landmarks[mppose.PoseLandmark[part_name].value].z,
    ]


def get_visibility(landmarks):
    if landmarks[mppose.PoseLandmark["RIGHT_HIP"].value].visibility < 0.8 or \
            landmarks[mppose.PoseLandmark["LEFT_HIP"].value].visibility < 0.8:
        return False
    else:
        return True


def get_body_ratio(landmarks):
    r_body = abs(landmarks[mppose.PoseLandmark["RIGHT_SHOULDER"].value].y
                 - landmarks[mppose.PoseLandmark["RIGHT_HIP"].value].y)
    l_body = abs(landmarks[mppose.PoseLandmark["LEFT_SHOULDER"].value].y
                 - landmarks[mppose.PoseLandmark["LEFT_HIP"].value].y)
    avg_body = (r_body + l_body) / 2
    r_leg = abs(landmarks[mppose.PoseLandmark["RIGHT_HIP"].value].y
                - landmarks[mppose.PoseLandmark["RIGHT_ANKLE"].value].y)
    l_leg = abs(landmarks[mppose.PoseLandmark["LEFT_HIP"].value].y
                - landmarks[mppose.PoseLandmark["LEFT_ANKLE"].value].y)
    if r_leg > l_leg:
        return r_leg / avg_body
    else:
        return l_leg / avg_body


def get_knee_angle(landmarks):
    r_hip = get_landmark(landmarks, "RIGHT_HIP")
    l_hip = get_landmark(landmarks, "LEFT_HIP")

    r_knee = get_landmark(landmarks, "RIGHT_KNEE")
    l_knee = get_landmark(landmarks, "LEFT_KNEE")

    r_ankle = get_landmark(landmarks, "RIGHT_ANKLE")
    l_ankle = get_landmark(landmarks, "LEFT_ANKLE")

    r_angle = calc_angles(r_hip, r_knee, r_ankle)
    l_angle = calc_angles(l_hip, l_knee, l_ankle)


    return [r_angle, l_angle]

video1 = cv2.VideoCapture('output_video.mp4')  
video1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'XVID'))
video1.set(cv2.CAP_PROP_FPS, 30)
video1.set(cv2.CAP_PROP_POS_FRAMES, 0)


video2 = cv2.VideoCapture(0)  


if not video1.isOpened() or not video2.isOpened():
    print("無法打開影片")
    exit()

frame_width2 = 1280
frame_height2 = 960
print(frame_height2) #480
print(frame_width2) #640

scaled_width = frame_width2 // 6
scaled_height = frame_height2 // 6


cv2.namedWindow('Combined Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Combined Video', frame_width2, frame_height2)

while True:
    if not status:
        start_time = time.time()
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()

    if not ret1:
        video1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    if not ret2:
        break
    results = pose.process(frame2)
    if results.pose_landmarks is not None:
        mp_drawing = mp.solutions.drawing_utils
        annotated_image = frame2.copy()
        mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mppose.POSE_CONNECTIONS)
        knee_angles = get_knee_angle(results.pose_landmarks.landmark)
        body_ratio = get_body_ratio(results.pose_landmarks.landmark)
        avg_angle = (knee_angles[0] + knee_angles[1]) // 2
        if 115<knee_angles[0] < 140 and 160<knee_angles[1]< 180:
            cv2.putText(annotated_image, "Left: {:.1f}".format(knee_angles[0]), (10, 230)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated_image, "Right: {:.1f}".format(knee_angles[1]), (10, 260)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated_image, "Good! Keep going!", (10, 290)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            status= True
        elif 115<knee_angles[1] < 140 and  160<knee_angles[0]< 180:
            cv2.putText(annotated_image, "Left: {:.1f}".format(knee_angles[0]), (10, 230)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated_image, "Right: {:.1f}".format(knee_angles[1]), (10, 260)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA) 
            cv2.putText(annotated_image, "Good! Keep going!", (10, 290)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)            
            status= True   
        else:
            cv2.putText(annotated_image, "Left: {:.1f}".format(knee_angles[0]), (10, 230)
                            , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_image, "Right: {:.1f}".format(knee_angles[1]), (10, 260)
                            , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_image, "Squat down!", (10, 290)
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA) 
            status= False               
        if status:
            current_time = time.time()
            elapsed_time = int(current_time - start_time)
            remaining_seconds = max(0, countdown_seconds - elapsed_time)

            text = f" {remaining_seconds} second"
            cv2.putText(annotated_image, text, (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if remaining_seconds == 0:
                status = False


    frame1_resized = cv2.resize(frame1, (scaled_width, scaled_height))
    annotated_image[0:scaled_height, 0:scaled_width] = frame1_resized
    cv2.imshow('Combined Video', annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video1.release()
video2.release()
cv2.destroyAllWindows()
```

