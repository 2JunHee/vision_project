import cv2 #opcnv 라이브러리
import mediapipe as mp #mediapipe 라이브러리
import numpy as np #numpy 라이브러리리

def hands_detection(max_num_hands): #손 추척함수 정의
    global mp_hands, mp_drawing, hands #글로벌 변수 선언
    mp_hands=mp.solutions.hands #mediapiep의 손, 손가락 추척 솔루션
    mp_drawing=mp.solutions.drawing_utils #mediapipe의 landmark점을 그리는 유틸리터
    hands=mp_hands.Hands(
        max_num_hands = max_num_hands, min_detection_confidence=0.5, min_tracking_confidence=0.5
        #인식할 수 있는 손의 갯수, 최소 인식의 신뢰도, 최소 추적의 신뢰도
    )

def train_data(): #학습데이터 파일
    global file
    file = np.genfromtxt('C:/Users/82103\Desktop/vision_end/data_1/test_0.csv', delimiter=',') #.csv파일을 불러옴
    print(file.shape) #파일의 배열 형태
    
def move(event, x,  y, flags, param): #마우스 이벤트 함수 정의
    global data, file #글로벌 변수 생성
    if event == cv2.EVENT_MOUSEWHEEL: #만약 마우스 휠이 움직면
        file=np.vstack((file, data)) #파일의 배열과 data 배열 결합
        print(file.shape) #파일의 배열 형태

def detection_result(res): #landmark 각도 계산
    joint=np.zeros((21,3)) # 21*3 행결생성
    for j,lm in enumerate(res.landmark): #반복문
        joint[j]=[lm.x, lm.y, lm.z] # 21개의 랜드마크를 3개의 좌표로 저장
        #각 랜드마크의 x,y,z 좌표를 joint에 저장
    v1=joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # 각 joint의 번호 인덱스 저장
    v2=joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] #각 joint의 번호 인덱스 저장
    v=v2-v1 #각각의 벡데의 각도 계산
    v=v/np.linalg.norm(v,axis=1)[:,np.newaxis] #벡터 정규화
    #두 벡터의 내적값은 cos값
    angle=np.arccos(np.einsum('nt,nt->n', #arccos에 대입하여 두 벡터가 이루는 각 angle변수에 저장
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],  #15개의 각도
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))  #15개의 각도
            
    angle=np.degrees(angle) #angle은 라디안이기 때문에 degree로 변환

    return angle #각도 리턴
  
max_num_hands = 1 #인식할 수 있는 손의 갯수
gesture = {
    0:'ㄱ', 1:'ㄴ', 2:'ㄷ', 3:'ㄹ', 4:'ㅁ', 5:'ㅂ', 6:'ㅅ',
    7:'ㅇ', 8:'ㅈ', 9:'ㅊ', 10:'ㅋ', 11:'ㅌ', 12:'ㅍ', 13:'ㅎ',
    14:'ㅏ', 15:'ㅑ', 16:'ㅓ', 17:'ㅕ', 18:'ㅗ', 19:'ㅛ', 20:'ㅜ', 21:'ㅠ',
    22:'ㅡ', 23:'ㅣ', 24:'ㅐ', 25:'ㅒ', 26:'ㅔ', 27:'ㅖ', 28:'ㅢ', 29:'ㅚ', 30:'ㅟ'
} #31가지의 제스쳐, 제스쳐 데이터는 손가락 landmark의 각도와 각각의 라벨

hands_detection(max_num_hands) #손추적 함수 호출

train_data() # 함수 호출

cap = cv2.VideoCapture(0) #카메라 영상

cv2.namedWindow("train_window") #train_window 생성
cv2.setMouseCallback("train_window", move) #train_window에서 마우스 이벤트 함수

index=0 #변수생성

while cap.isOpened(): #cap이 열려있는동안 반복문
    ret, img = cap.read() #cap비디오의 한 프레임씩 읽기
    if not ret: #만약 비어있다면
        continue #코드 생략

    img=cv2.flip(img, 1) #영상의 좌우반전
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #bgr->rgb

    result=hands.process(img) #손 랜드마크 감지 결과 저장

    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #rgb->bgr

    if result.multi_hand_landmarks is not None: #result.multi_hand_landmark가 비어있지 않다면
        #multi_hand_landmark 는 손의 21개의 랜드마크 리스트
        for res in result.multi_hand_landmarks: #반복문
            angle_train = detection_result(res) #각도 계산 함수 호출
            data=np.array([angle_train], dtype=np.float32) #배열형태로 저장

            data=np.append(data,index) #index번째 클래스의 학습데이터 생성

            

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS) #손 관절의 landmark그리기

    cv2.imshow("train_window", img) #train_window창에 img 출력
    c=cv2.waitKey(1) #입력키 기다림
    if c == 27: #esc누르면
        break #종료
    elif c==ord('n'): #n누르면
        index+=1 #인덱스 증가
        print(index) #인덱스 출력
    elif c==ord('b'): #b누르면
        index-=1 #인덱스 감소

np.savetxt('C:/Users/82103\Desktop/vision_end/data_1/test_0.csv',file, delimiter=',') #파일에 저장

