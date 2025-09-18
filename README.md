# AI.Data_DL_Project

<p align="center">
   <img width="200" height="346" alt="image" src="https://github.com/user-attachments/assets/3df8a5d8-35c5-4144-a741-e9b0e89b9a5f" />
</p>

<p align="center">
   [사용자 맞춤 식단 및 운동 추천을 위한 AI 딥러닝 예측 서비스 Fitter]
</p>
<br>

---

[프로젝트 서사]
<br>
많은 사람들이 다이어트나 벌크업을 시도하지만, 전문가의 도움 없이 혼자서 진행하는 것은 결코 쉽지 않다. 어떤 운동을 해야 하는지, 어떤 식단을 짜야 하는지에 대한 정보가 부족하기 때문이다. 무작정 헬스장을 등록하고 건강한 식단을 시도하더라도, 본인에게 적합하지 않다면 큰 효과를 기대하기 어렵다. 체중 관리를 위해 하루에 얼마나 많은 칼로리를 소모해야 하는지 계산하는 것조차 번거로운 일이다. 
<br>
이러한 문제를 해결하기 위해 우리는 사용자 대신 귀찮은 부분을 처리해주는 맞춤형 서비스를 제안한다. 사용자는 따로 정보를 검색할 필요 없이, 바로 식단과 운동 루틴을 추천받을 수 있다. 또한, 해당 루틴을 꾸준히 실천했을 때 목표 체중까지 도달하는 예상 기간을 제공함으로써, 사용자가 목표를 달성하기 위해 필요한 모든 서비스를 지원하고 도전 의지를 고취할 수 있을 것이라 기대된다.

<br><br>
---


<p align="center">
   <img src="image6.gif" width="200" />
   <img src="image7.gif" width="200" />
</p>

<p align="center">
   <img src="image8.gif" width="200" />
   <img src="image9.gif" width="200" />
</p>

---

<br><br>
<p align="center">
  <img width="192" height="212" alt="스크린샷 2025-09-18 오후 2 49 31" src="https://github.com/user-attachments/assets/fa5e9f0a-4f7f-469d-87cf-120a33013435" />

</p>

<h2 align="center"> 홍영훈 👋</h2>

<p align="center">
  🔹 데이터 가공,분석 및 AI / 딥러닝 파트<br>
  🔹 데이터 사이언스,머신러닝, 신경망, 예측 모델링에 열정을 가지고 있습니다<br>
  
</p>

<p align="center">
  🌐 <a href="https://github.com/YEONGHUN-H" target="_blank">GitHub 바로가기</a>
</p>

---
<br>
<p align="center">
   [💻데이터 및 AI 파트 간이 절차]
</p>

<br>

---

<br><br>
[BMI DATA 확인]

[데이터 탐색 및 준비]
<br>
<img width="930" height="194" alt="스크린샷 2025-09-18 오후 2 58 35" src="https://github.com/user-attachments/assets/51895a87-599f-416f-87ce-813280fbe66c" />

초기 데이터: https://www.kaggle.com/datasets/rukenmissonnier/age-weight-height-bmi-analysis
<br>
<img width="365" height="305" alt="스크린샷 2025-09-18 오후 3 00 10" src="https://github.com/user-attachments/assets/5578834b-56bb-4ee6-a4eb-13e67b341bc8" />

<데이터 샘플 >
-500개
<항목>
-Age, Weight, Height, BMI
<br>

---
<br>

<p align="center">
   [데이터 가공]
</p>


<br>


초기 데이터를 활용하여 새로운 신체 데이터를 생성.
<br>
🚨 신체 정보 데이터는 개인정보! 기초 데이터를 토대로 임의의 값을 랜덤하게 할당하여 741개의 새로운 신체정보 데이터를 생성 🚨 
<br><br>
BMI 등급(BMI Class): BMI값을 기준으로 체중 상태 분류
목표 체중(TargetWeight): 개인별 목표 체중 설정
활동 수준(Activity Level): 일일 활동량 숫자로 분류 1~4
목표 유형(GoalType): 다이어트,유지,벌크업
성별(Gender): 성별 추가
목표 달성 여부(Achieved): 달성 여부 이진 변수로 표현
목표 달성 기간(DaysToGoal): 목표 달성 기간
<br>
<img width="577" height="213" alt="bmi가공사진" src="https://github.com/user-attachments/assets/98714bc6-ff1e-422d-8bb6-f64f7146a16b" />

---

<br><br>

<p align="center">
   [식단 추천 기준 설정]
</p>




<br><br>
--- 

<br><br>

<p align="center">
   [운동 추천 기준 설정]
</p>


<br><br>

* 운동 추천 로직: BMI와 같은 데이터를 기반으로 다음과 같은 운동을 추천할 수 있음.
    * 저체중: 체중 증가를 위한 근력 운동 + 고칼로리 식단.
    * 정상체중: 균형 잡힌 운동(유산소 + 근력 운동).
    * 과체중: 체중 감소를 위한 유산소 운동 중심 프로그램.
    * 비만: 저강도 유산소 + 식단 조절.
* 기타 기준: 성별, 나이, 활동 수준에 따라 추천을 세분화합니다.

3. 모델 개발
* 데이터 라벨링: 기존 데이터를 이용해 운동 추천을 위한 라벨을 추가. 예를 들어, "운동 유형" 컬럼에 "유산소", "근력", "혼합" 등의 값을 할당. 식단은 “채식”, “육식”
* 모델 학습:
    * 머신러닝 알고리즘(예: 분류 모델): 사용자 데이터를 입력하면 운동 유형을 예측하도록 학습.
    * 추천 시스템: 비슷한 BMI, 나이, 성별을 가진 사용자 그룹을 생성하고 운동과 식단을 추천.
 

4. 예측 및 평가
* 모델 테스트: 새로운 사용자 데이터를 입력해 운동 추천 결과를 확인할 예정.
* 모델 평가: 정확도, F1-score 등을 통해 추천 정확도를 평가할 예정.


아래는 딥러닝 모델과 관련된 내용을 중심으로 작성된 README 문서입니다. 프로젝트의 주요 딥러닝 관련 파일, 학습 및 배포 과정, 그리고 관련 내용이 포함되어 있습니다.

README: 딥러닝 기반 목표 예측 서비스
프로젝트 개요
이 프로젝트는 Feedforward Neural Network (FFNN) 기반으로 사용자의 목표 체중 달성일을 예측하는 딥러닝 모델을 구현하고 배포합니다. Google Colab에서 딥러닝 모델을 학습하였으며, Flask 및 Python 스크립트를 통해 예측 서비스를 제공합니다.

주요 기능
목표 체중 달성일 예측

기능: 사용자의 신체 정보(BMI, 체중, 활동량 등)를 바탕으로 딥러닝 모델이 목표 체중까지 소요되는 기간을 예측.
사용 데이터:
BMI 데이터 (체중, 키, 목표 체중)
사용자 활동 수준 (TDEE, BMR)
예측 결과:
json
코드 복사
{
  "username": "홍길동",
  "days_to_goal": 45,
  "message": "홍길동님, 목표 체중 달성 예상 소요 기간은 약 45일입니다."
}
딥러닝 모델 설계 및 학습

신경망 구조: Feedforward Neural Network (FFNN)
은닉층: [128, 64, 32]
활성화 함수: LeakyReLU
최적화 기법: Adam Optimizer
학습 데이터:
입력 데이터: 사용자의 나이, 키, 현재 체중, 목표 체중, BMI, TDEE 등
출력 데이터: 목표 체중 달성일까지 예상 소요 일수
하이퍼파라미터 탐색: Grid Search를 통해 최적 하이퍼파라미터를 선택.
모델 배포 및 운영

배포 방식:
Python 스탠드얼론 스크립트를 사용하여 모델 예측 수행.
Flask API 서버를 통해 RESTful 서비스로 배포 가능.
입력 방식: JSON 형식.
출력 방식: JSON 형식으로 예측 결과 반환.
프로젝트 파일 구조
plaintext
코드 복사
Project/
│
├── src/
│   ├── python/
│   │   ├── main_predictor.py         # 딥러닝 모델 기반 목표 예측 코드
│   │   ├── Data/
│   │   │   ├── feature_columns.json  # 모델 입력 피처 정보
│   │   │   ├── scaler.joblib         # 데이터 스케일링 파일
│   │   │   ├── P_model.pth           # 학습된 PyTorch 모델
│   │
│   ├── chatbot/
│   │   ├── chatbot_handler.py        # 챗봇 핸들러 코드
│   │   ├── sendbird_integration.py   # SendBird 플랫폼 연동
│   │   ├── gpt_integration.py        # OpenAI GPT API 호출 코드
│   │
├── data/
│   ├── bmi_data.csv                  # BMI 및 체중 데이터셋
│   ├── exercise.csv                  # 운동 추천 데이터셋
│   ├── food_data.csv                 # 식단 추천 데이터셋
│
├── requirements.txt                  # 필요 라이브러리 목록
└── README.md                         # 프로젝트 설명서
딥러닝 모델 설계
Feedforward Neural Network (FFNN)

입력층: 사용자 데이터를 기반으로 한 총 25개의 입력 피처
은닉층: [128, 64, 32]
활성화 함수: LeakyReLU
정규화: Batch Normalization 및 Dropout(0.2)
출력층: 1개의 예측값 (목표 달성일까지의 소요 기간)
최적화 및 손실 함수

최적화 기법: Adam Optimizer
손실 함수: Mean Squared Error (MSE)
하이퍼파라미터 탐색

탐색 기법: Grid Search
탐색 범위:
은닉층 크기: [128, 64, 32]
학습률: [0.01, 0.005]
배치 사이즈: [128, 256]
최적 결과:
은닉층: [128, 64, 32]
학습률: 0.005
배치 사이즈: 256
필요 라이브러리
requirements.txt 파일에서 다음 라이브러리 설치:

plaintext
코드 복사
torch==1.13.1
pandas==1.5.3
numpy==1.22.4
flask==2.2.3
scikit-learn==1.1.3
matplotlib==3.5.3
seaborn==0.11.2
joblib==1.1.0
json==2.0.9
모델 학습 및 실행
데이터 준비

BMI, 체중, TDEE, 활동 수준 등 사용자 데이터를 준비합니다.
데이터는 CSV 형식으로 제공되며, bmi_data.csv를 통해 학습합니다.
모델 학습 (Google Colab)

Google Colab에서 GPU 환경을 사용하여 모델 학습.
학습 코드 (src/python/main_predictor.py) 실행:
python
코드 복사
python main_predictor.py
모델 실행 (Standalone Script)

표준 입력(JSON 형식)을 받아 예측 결과 반환:
bash
코드 복사
python main_predictor.py < input.json
모델 배포 (Flask API)

Flask를 사용해 RESTful API로 배포:
bash
코드 복사
flask run
모델 평가
평가 지표:

MAE (Mean Absolute Error): 8.65
R² (결정계수): 0.99
결과 해석:

높은 R² 값은 모델의 높은 예측 정확도를 나타냅니다.
평균적으로 목표 달성일 예측은 ±8.65일의 오차를 가집니다.
챗봇 연동
SendBird 플랫폼 사용

사용자의 질의에 대해 딥러닝 예측 결과를 응답합니다.
chatbot_handler.py를 실행하여 챗봇 테스트.
OpenAI GPT API 통합

자연어 처리를 위해 GPT API를 호출하여 사용자와 상호작용.


[목표 예측 및 추천 서비스 데이터 파이프라인]
1. 데이터 탐색 및 준비
데이터 확인

BMI 데이터: 741명의 키, 몸무게, 성별, 나이, BMI 및 목표 체중 정보를 포함.
운동 데이터: 운동 이름, 소모 칼로리, 운동 부위(상체, 하체 등), 운동 유형(근력, 유산소 등) 포함.
식단 데이터: 음식 이름, 칼로리, 탄수화물, 단백질, 지방 등 식품 영양 정보를 포함.
데이터셋 구조:
plaintext
코드 복사
BMI 데이터셋: 키, 몸무게, 성별, 나이, BMI, 목표 체중
운동 데이터셋: 운동 이름, 운동 부위, 소모 칼로리, 운동 유형
식단 데이터셋: 음식 이름, 칼로리, 탄수화물, 단백질, 지방
결측치 처리

BMI 데이터: 결측값을 평균 또는 중앙값으로 채움.
Weight, Height, Age의 결측값 → 평균으로 대체.
TargetWeight: 체중의 90% 또는 체중 -10kg로 설정.
운동/식단 데이터: 결측값이 존재하지 않음.
데이터 변환 및 추가 피처 생성

BMI 계산:
BMI = 체중(kg) / (키(m)^2)
BMR 및 TDEE 계산:
BMR (기초대사량):
남성: 10 * 체중 + 6.25 * 키(cm) - 5 * 나이 + 5
여성: 10 * 체중 + 6.25 * 키(cm) - 5 * 나이 - 161
TDEE (하루 총 에너지 소비량): BMR * 활동 수준 계수
새로운 컬럼 생성:
TargetBMI: 목표 체중으로 BMI 재계산.
Calorie_Target: TDEE의 80%로 설정.
관련 코드 예시:

python
코드 복사
# 결측값 처리
bmi_data.fillna({
    'Weight': bmi_data['Weight'].mean(),
    'Height': bmi_data['Height'].mean(),
    'Age': bmi_data['Age'].median(),
    'TargetWeight': bmi_data['Weight'] * 0.9  # 체중의 90%로 기본 설정
}, inplace=True)

# BMI 계산
bmi_data['BMI'] = bmi_data['Weight'] / (bmi_data['Height'] ** 2)
bmi_data['TargetBMI'] = bmi_data['TargetWeight'] / (bmi_data['Height'] ** 2)

# BMR 및 TDEE 계산
bmi_data['BMR'] = 10 * bmi_data['Weight'] + 6.25 * (bmi_data['Height'] * 100) - 5 * bmi_data['Age']
bmi_data['BMR'] += bmi_data['Gender'].map({'Male': 5, 'Female': -161})
bmi_data['TDEE'] = bmi_data['BMR'] * bmi_data['ActivityLevel']
2. 운동 및 식단 추천 기준 설정
운동 추천 기준

BMI 및 활동 수준 기반으로 추천:
저체중 (BMI < 18.5): 체중 증가를 위한 근력 운동 + 고칼로리 식단.
정상 체중 (18.5 ≤ BMI < 25.0): 균형 잡힌 운동(유산소 + 근력 운동).
과체중 (25.0 ≤ BMI < 30.0): 체중 감소를 위한 유산소 운동 중심 프로그램.
비만 (BMI ≥ 30.0): 저강도 유산소 + 식단 조절.
운동 부위 선택: 사용자가 선호하는 부위(예: 하체, 상체) 기반 추천.
식단 추천 기준

목표 칼로리(TDEE 80%)에 맞춰 하루 식단을 구성.
식사 시간(아침, 점심, 저녁, 간식)에 따라 추천.
영양소 비율:
벌크업: 탄수화물 50%, 단백질 30%, 지방 20%.
체중 감소: 탄수화물 40%, 단백질 40%, 지방 20%.
음식 카테고리(밥, 국, 찌개 등)별 추천.
3. 딥러닝 모델 개발
데이터 라벨링

BMI 및 목표 유형(벌크업, 감량 등)에 따라 운동 및 식단을 라벨링.
예:
운동 유형: 유산소, 근력, 혼합
식단 유형: 고단백, 저지방
모델 학습

입력 피처: 나이, 키, 체중, 목표 체중, BMI, 활동 수준, 성별, 목표 유형 등.
출력 피처: 목표 체중 달성일까지 예상 소요 기간.
모델 구조: Feedforward Neural Network (FFNN).
은닉층: [128, 64, 32]
활성화 함수: LeakyReLU
최적화 기법: Adam Optimizer
모델 학습 및 최적화: Grid Search를 사용하여 하이퍼파라미터 최적화.
관련 코드 예시:

python
코드 복사
# Feedforward Neural Network 정의
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim):
        super(FeedforwardNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)
4. 예측 및 평가
모델 테스트

새로운 사용자 데이터를 입력해 목표 달성일을 예측.
예시 입력:
json
코드 복사
{
  "age": 25,
  "height": 170,
  "current_weight": 75,
  "target_weight": 65,
  "bmi": 25.9,
  "tdee": 2200,
  "activity_level": 3,
  "gender": "Male",
  "goal_type": "저지방 고단백"
}
모델 평가

평가 지표:
Mean Absolute Error (MAE): 8.65
R² (결정계수): 0.99
결과 해석

예측 결과는 ±8.65일의 오차 내에서 정확하게 목표 달성일을 예측.
위 내용은 데이터 준비, 가공, 추천 로직, 모델 학습 및 평가까지 딥러닝 기반 목표 예측 서비스의 전체 파이프라인을 정리한 것입니다

