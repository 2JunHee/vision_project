import cv2 #opencv 라이브러리
import mediapipe as mp #midiapipe 라이브러리
import numpy as np #numpy 라이브러리
from sklearn.preprocessing import StandardScaler #sklearn의 StandardScaler
from sklearn.neighbors import KNeighborsClassifier #sklearn의  KNeighborsClassifier
from sklearn.svm import SVC #sklearn의 SVC
from sklearn.neural_network import MLPClassifier #sklearn MLPClassifier

def file(): #data file 함수 정의
    global angle, label, test_angle, test_label #글로벌 변수 생성
    file = np.genfromtxt('C:/Users/82103\Desktop/vision_end/data_1/train_0.csv', delimiter=',') #train_gesture.csv 파일 불러옴
    angle = file[:,:-1].astype(np.float32) #
    label = file[:, -1].astype(np.float32)

    file = np.genfromtxt('C:/Users/82103\Desktop/vision_end/data_1/test_0.csv',delimiter=',') #test_gesture.csv 파일 불러옴
    test_angle = file[:,:-1].astype(np.float32)
    test_label = file[:, -1].astype(np.float32)
    return angle, label, test_angle, test_label #리턴

def data_Normal(angle, test_angle): #학습 데이터 전처리
    sc = StandardScaler() #표준화 함수
    sc.fit(angle) #표준화 학습
    angle_std =sc.transform(angle) #학습데이터 표준화
    test_angle_std=sc.transform(test_angle) #테스트 학습데이터 표준화
    return angle_std, test_angle_std #리턴

def ML_test(angle_std, test_angle_std, label): #머신러닝 학습 함수
    knn = KNeighborsClassifier(n_neighbors=10) #knn 최근접 10개
    knn.fit(angle_std, label) #knn 학습
    predy_knn = knn.predict(test_angle_std) #knn 추론

    svm = SVC(kernel='rbf', gamma = 'auto', max_iter=1000) #svm, rbf 커널, gamma 오토, 1000번반복
    svm.fit(angle_std, label) #svm 학습
    predy_svm = svm.predict(test_angle_std) #svm 추론

    # mlp = MLPClassifier(hidden_layer_sizes=(100,100), activation='logistic', \
    #                 solver='sgd', alpha=0.0001, batch_size=32, \
    #                 learning_rate_init=0.1, max_iter=5000000) #mlp , 은닉유닛 10, 은닉층 2, 활성화 함수 logistic, 경사하강법의 종류 sgd, 규제 적용 매개변수
    #                 #배치 크기 32, 학습률 초기값 매개변수 0.1, 에포크 횟수 5000000
    mlp=MLPClassifier(solver='sgd', random_state=0, hidden_layer_sizes=[200,200,200,200])
    #mlp, 경사하강법 sgd, 난수생성, 은닉유닛 200 은닉층 4
    mlp.fit(angle_std, label) #mlp 학습
    predy_mlp = mlp.predict(test_angle_std) #mlp 추론

    return predy_knn, predy_mlp, predy_svm #리턴


file() #함수 호출
angle_std, test_angle_std = data_Normal(angle, test_angle) #학습데이터 전처리
predy_knn, predy_mlp, predy_svm=ML_test(angle_std, test_angle_std,label) #머신러닝을 사용해 학습과 추론

print("knn accuracy : {:.2f}". format(np.mean(predy_knn == test_label)) ) #인식률 출력
print("svm accuracy : {:.2f}". format(np.mean(predy_svm == test_label)) ) #인식률 출력
print("mlp accuracy : {:.2f}". format(np.mean(predy_mlp == test_label)) ) #인식률 출력


