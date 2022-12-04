import cv2 #opencv 라이브러리
import mediapipe as mp #mediapipe 라이브러리
import numpy as np #numpy 라이브러리
import copy #copy 라이브러리
import pygame as py #pygame 라이브러리
from hangul_utils import join_jamos #hangul_utils의 join_jamos
from PIL import ImageFont, ImageDraw, Image #PIL의 Imagefont, ImageDraw, Image
from sklearn.svm import SVC #sklearn의 SVC
from sklearn.neighbors import KNeighborsClassifier #sklearn의 KNeigborsClassifier
from sklearn.neural_network import MLPClassifier #sklearn의 MLPClassifier
from sklearn.preprocessing import StandardScaler #sklearn의 StandardScaler

def hand_tracking(max_num_hands): #손 추적 함수 정의
    mp_hands=mp.solutions.hands #mediapipe의 손, 손가락 추적 솔루션
    mp_drawing=mp.solutions.drawing_utils #mediapipe의 landmark점을 그리는 유틸리티
    hands=mp_hands.Hands( 
    max_num_hands=max_num_hands, #인식할 수 있는 손의 갯수
    min_detection_confidence=0.5, #최소 인식의 신뢰도
    min_tracking_confidence=0.5 #초소 추적의 신뢰도
    )
    return hands, mp_drawing, mp_hands #리턴

def train_data(): #학습데이터 호출 함수 정의 
    global angle, label #글로벌 변수 선언
    file=np.genfromtxt('C:/Users/82103\Desktop/vision_end/data_1/train_0.csv', delimiter=',') #train.csv파일 불러옴
    angle=file[:,:-1].astype(np.float32)
    label=file[:, -1].astype(np.float32)

def data_standardScaler(): #데이터 전처리 함수 정의
    sc=StandardScaler() # 표준화 함수
    sc.fit(angle) #학습
    angle_std=sc.transform(angle) #데이터 표준화
    return angle_std, sc #리턴턴

def data_train(angle_std): #데이터 학습 함수 정의
    svm=SVC(kernel='rbf', gamma='auto', max_iter=1000) #Classification에 사용되는 SVM모델, 학습 반복횟수 1000번
    svm.fit(angle_std, label) #svm 학습

    knn=KNeighborsClassifier(n_neighbors=10) #knn, 최근접 10개
    knn.fit(angle_std, label) #knn 학습

    # mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', \
    #                 solver='sgd', alpha=0.01, batch_size=32, \
    #                 learning_rate_init=0.1, max_iter=500)
    mlp=MLPClassifier(solver='sgd', random_state=0, hidden_layer_sizes=[200,200,200,200])
    #mlp, 경사하강법 sgd, 난수생성, 은닉유닛 200 은닉층 4
    mlp.fit(angle_std, label) #mlp 학습
    return svm, knn, mlp #리턴

def click(event, x, y, flags, param): #마우스이벤트 함수 정의
    global word1, word2, rect1, rect2, text_knn #글로벌 변수 선언
    rect1=py.Rect(350,0,150,75) #add 구역
    rect2=py.Rect(350,75,150,75) #reset 구역
    if event == cv2.EVENT_LBUTTONDOWN: #만약 왼쪽 버튼이 눌리면
        if rect1.collidepoint((x, y)): #만약 좌표가 포함되어 있다면
            word1=word1+text_knn #단어 저장
            word2=join_jamos(word1) #단어 조합
            
        elif rect2.collidepoint((x, y)): #좌표가 포함되어 있다면
            word1='' #초기화
            word2='' #초기화

def change(img, image_clone, hands): #영상 전처리 함수
    img=cv2.flip(img, 1) #영상 좌우반전
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #bgr->rgb
    result=hands.process(img) #손 랜드마크 감지
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR) #rgb->bgr
    imageT=copy.copy(image_clone) #text창 복사
    return result, imageT, img #리턴

def tracking_result(res, sc): #landmark 계산
    joint=np.zeros((21,3)) #21*3의 행렬생성
    for j,lm in enumerate(res.landmark): #반복문
        joint[j]=[lm.x, lm.y, lm.z] #21개의 랜드마크를 3개의 좌표로 저장
        #각 랜드마크의 x,y,z 좌표를 joint에 저장
    v1=joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] #각 joint의 번호 인덱스 저장
    v2=joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] #각 joint의 번호 인덱스 저장
    v=v2-v1 #각각의 벡터의 각도 계산
    v=v/np.linalg.norm(v,axis=1)[:, np.newaxis] #벡터 정규화
    #두 벡터의 값은 cos값
    angle=np.arccos(np.einsum('nt,nt->n', #arccos에 대입하여 두 벡터가 이루는 각을 angle변수에 저장
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],  #15개의 각도
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) #15개의 각도
    angle=np.degrees(angle) #angle은 라디안이기 때문에 degree로 저장
    data=np.array([angle], dtype=np.float32) #배열의 형태로 저장
    data_std=sc.transform(data) #데이터 표준화
    
    return data_std #리턴

def data_predic(knn, svm, mlp, data_std): #머신러닝 추론 함수 정의
    pred_knn=knn.predict(data_std) #knn 추론
    idx_knn=pred_knn[0] #추론 결과

    pred_svm=svm.predict(data_std) #svm 추론
    idx_svm=pred_svm[0] #추론 결과

    pred_mlp=mlp.predict(data_std)#mlp 추론
    idx_mlp=pred_mlp[0] #추론결과

    return idx_knn, idx_mlp, idx_svm #리턴

def text_window(imageT, font, word2): #text창 영상 변환
    imageT=Image.fromarray(imageT) #PIL 영상 변환
    draw=ImageDraw.Draw(imageT) #imageT영상에 드로우
    org1=(50,50) #좌표
    draw.text(org1,word2, font=font,fill=(0)) #문자출력
    org_add=(380,10) #좌표
    org_reset=(370,85) #좌표
    text_add='add' #add 텍스트
    text_reset='reset' #reset 텍스트
    draw.text(org_add,text_add, font=font, fill=(0)) #문자 출력
    draw.text(org_reset,text_reset, font=font, fill=(0)) #문자 출력
    imageT=np.array(imageT) #opencv영상 변환

    return imageT #리턴

def text_camera(res, img, idx_knn, idx_svm, idx_mlp):
    global text_knn
    img=Image.fromarray(img) #영상 변환
    draw=ImageDraw.Draw(img) #img 그리기
    org=(res.landmark[0].x, res.landmark[0].y) #좌표
    org_=(res.landmark[0].x, res.landmark[0].y+20) #좌표
    org__=(res.landmark[0].x, res.landmark[0].y+40) #좌표

    text_knn=gesture[idx_knn].upper() #추론 결과
    # text_svm=gesture[idx_svm].upper() #추론 결과
    # text_mlp=gesture[idx_mlp].upper() #추론 결과

    draw.text(org_,text_knn,font=font,fill=(0)) #문자열 출력
    # draw.text(org__,text_svm,font=font,fill=(0)) #문자열 출력
    # draw.text(org,text_mlp,font=font,fill=(0)) #문자열 출력
    img=np.array(img) #opencv영상 변환
    return img


max_num_hands = 1 #손의 갯수
gesture = { #
    0:'ㄱ', 1:'ㄴ', 2:'ㄷ', 3:'ㄹ', 4:'ㅁ', 5:'ㅂ', 6:'ㅅ',
    7:'ㅇ', 8:'ㅈ', 9:'ㅊ', 10:'ㅋ', 11:'ㅌ', 12:'ㅍ', 13:'ㅎ',
    14:'ㅏ', 15:'ㅑ', 16:'ㅓ', 17:'ㅕ', 18:'ㅗ', 19:'ㅛ', 20:'ㅜ', 21:'ㅠ',
    22:'ㅡ', 23:'ㅣ', 24:'ㅐ', 25:'ㅒ', 26:'ㅔ', 27:'ㅖ', 28:'ㅢ', 29:'ㅚ', 30:'ㅟ'
} #31가지의 제스쳐, 제스쳐 데이터는 손가락 landmark의 각도와 각각의 라벨

hands, mp_drawing, mp_hands = hand_tracking(max_num_hands) #손 추적함수 호출

train_data() #학습 데이터 호출

angle_std, sc = data_standardScaler() #학습데이터 전처리

svm, knn, mlp = data_train(angle_std) #머신러닝 학습

cap=cv2.VideoCapture(0) #카메라 영상

font=ImageFont.truetype("fonts/gulim.ttc", 50) #출력 문자의 폰트, 크기

imageT = np.ones((150, 500), dtype=np.uint8)*255 #배열 생성
cv2.line(imageT, (350,0),(350,150), (0), 3) #line 그리기함수
cv2.line(imageT, (350,75),(500,75), (0), 3) #line 그리기 함수

image_clone=copy.copy(imageT) #얕은 복사

cv2.namedWindow("imageT") #윈도우창 생성
cv2.setMouseCallback("imageT", click) #윈도우창에서 마우스 이벤트
word1='' #변수 초기화
word2='' #변수 초기화

while cap.isOpened(): #cap이 열려있는 동안 반복문
    ret,img=cap.read() #cap비디오의 한 프레임씩 읽기
    if not ret: #만약 비어있다면
        continue #아래 코드 생략

    result,imageT,img = change(img, image_clone, hands) #영상 전처리 함수 호출

    text_knn='' #변수 초기화

    if result.multi_hand_landmarks is not None: #result.multi_hand_landmark가 비어있지 않다면
        #multi_hand_landmark 는 손의 21개의 랜드마크 리스트
        cv2.setMouseCallback("imageT", click) #마우스 이벤트
        for res in result.multi_hand_landmarks: #반복문
            data_std = tracking_result(res, sc) #각도 계산 함수 호출

            idx_knn, idx_svm, idx_mlp = data_predic(knn, svm, mlp, data_std) #머신러닝 추론

            img = text_camera(res, img, idx_knn, idx_svm, idx_mlp)

            mp_drawing.draw_landmarks(img,res,mp_hands.HAND_CONNECTIONS) #landmark 마디 그리기함수

    imageT = text_window(imageT, font, word2) #문자 출력함수 호출
    cv2.imshow("camera", img) #윈도우창에 img 출력
    cv2.imshow("imageT", imageT) #윈도우창에 iamgeT출력
    
    k=cv2.waitKey(1) #입력키 기다림
    if k==27: #esc 입력시 
        break #종료