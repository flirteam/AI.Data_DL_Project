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

<p align="center">
   [BMI DATA 확인]
</p>



[데이터 탐색 및 준비]
<br>
<img width="930" height="194" alt="스크린샷 2025-09-18 오후 2 58 35" src="https://github.com/user-attachments/assets/51895a87-599f-416f-87ce-813280fbe66c" />

초기 데이터: https://www.kaggle.com/datasets/rukenmissonnier/age-weight-height-bmi-analysis
<br>

<img width="365" height="305" alt="스크린샷 2025-09-18 오후 3 00 10" src="https://github.com/user-attachments/assets/5578834b-56bb-4ee6-a4eb-13e67b341bc8"
/>
</div>

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
🚨 신체 정보 데이터는 개인정보이기에 기초 데이터를 토대로 임의의 값을 랜덤하게 할당하여 741개의 새로운 신체정보 데이터를 생성하였다. 🚨 

Data Processing

GoalType(목표유형)과 WeightDiff(목표체중과 차이)기반 목표 달성 일수(DaysToGoal) 계산.
<br>
남성: 체중 범위 (70-100)kg 내에서 생성
여성: 체중 범위 (50-75)kg 내에서 생성
<br>
체중 변화량에 따른 DayToGoal 범위

<img width="380" height="194" alt="image" src="https://github.com/user-attachments/assets/26f50cd9-6fe8-4ca0-bd62-1f64138d4ee7" />


균등 분포 난수 함수를 활용하여, 설정된 범위 내에서 임의의 목표 예측일을 생성한다.

<img width="416" height="403" alt="image" src="https://github.com/user-attachments/assets/ce7b45be-3c2c-49cd-985a-d49dafc38844" />

<br><br>
BMI 등급(BMI Class): BMI값을 기준으로 체중 상태 분류 <br>
목표 체중(TargetWeight): 개인별 목표 체중 설정 <br>
활동 수준(Activity Level): 일일 활동량 숫자로 분류 1~4 <br>
목표 유형(GoalType): 다이어트,유지,벌크업 <br>
성별(Gender): 성별 추가 <br>
목표 달성 여부(Achieved): 달성 여부 이진 변수로 표현 <br>
목표 달성 기간(DaysToGoal): 목표 달성 기간 <br>
<br>
<img width="577" height="213" alt="bmi가공사진" src="https://github.com/user-attachments/assets/98714bc6-ff1e-422d-8bb6-f64f7146a16b" />



---

<br><br>

<p align="center">
   [규칙 기반 추천 알고리즘-식단 추천 및 7일치 맞춤형 데이터 생성]
</p>
<br>

<img width="497" height="289" alt="image" src="https://github.com/user-attachments/assets/762d7b27-49d6-4e36-94dd-37a6c34d6ad4" />
<img width="492" height="113" alt="image" src="https://github.com/user-attachments/assets/6b213b80-e15a-49bc-8dd3-e14253b6c5df" />
<br>

일일 칼로리 균형 및 체중 변화 계산
-> 일일 칼로리 균형(칼로리 적자): 총 섭취 칼로리 - (활동대사량+운동)

체중 변화 추정
-> 칼로리 균형*일수/7700(1kg당 필요한 칼로리)

7일 맞춤형 플랜 생성
-> 식단(아침,점심,저녁) 총 섭취 칼로리와 일일 칼로리 균형을 이용.

식단 데이터를 이용하여 사용자에게 맞춤형 식단을 추천한다.

<br>






<br><br>
--- 

<br><br>

<p align="center">
   [규칙 기반 추천 알고리즘-운동 추천 및 7일치 맞춤형 데이터 생성]
</p>


<br><br>

<img width="469" height="344" alt="image" src="https://github.com/user-attachments/assets/c5882d23-1be0-4498-b80f-c2c3f6aef332" />
<br>
운동 추천 로직: BMI와 같은 데이터를 기반으로 다음과 같은 운동을 추천할 수 있게 설정한다.
    * 저체중: 체중 증가를 위한 근력 운동 + 고칼로리 식단.
    * 정상체중: 균형 잡힌 운동(유산소 + 근력 운동).
    * 과체중: 체중 감소를 위한 유산소 운동 중심 프로그램.
<br>
신체정보 데이터 전처리
-> BMI,AMR,BMR 등 게산 진행 및 결측값 처리 후 입력.

칼로리 타겟 설정
-> 사용자 목표, 활동 대사량 기반 맞춤 목표 섭취 칼로리 설정.

운동 데이터 전처리
-> 운동명, 부위, 카테고리별 필요 정보 입력.

<br>
일주일치 운동 정보 추출: 사용자에게 맞는 맞춤형 운동 추천

<br><br>
---
<br>
<p align="center">
   [모델 생성]
</p>
<br>
인공지능(딥러닝) 예측 모델
<br>
사용자의 신체 정보, 식단, 운동 데이터를 종합하여 '목표 달성까지 걸리는 예상 일수(DaysToGoal)'를 예측하는 파이토치(PyTorch) 기반의 심층 신경망(DNN) 회귀(Regression) 모델
<br>
<br>
--
<br>
🧠 1. 핵심 알고리즘: 심층 피드포워드 신경망 (Deep Feedforward NN)
<br>
목표: 다양한 변수(체중, 섭취 칼로리, 운동 시간 등)가 '목표 달성 일수'에 미치는 복잡하고 비선형적인 관계를 AI가 스스로 학습하여 연속적인 수치(일수)를 예측.
<br>
구조 (FeedforwardNNImproved): 여러 개의 은닉층(Hidden Layer)을 거치며 데이터를 분석. 
단순히 층만 깊게 쌓은 것이 아니라, 학습의 안정성을 위해 **배치 정규화(BatchNorm1d)**를 적용하고, 과적합(Overfitting)을 막기 위해 20%의 데이터를 무작위로 끄는 드롭아웃(Dropout) 기법을 사용하여 모델의 일반화 성능을 높임.
<br>
<br>
--
<br>
📊 2. 정교한 데이터 전처리 (Data Preprocessing)
<br>
AI 모델이 데이터를 잘 소화하기 위한 3가지 핵심 전처리.
<br>
파생 변수 생성: '목표 체중'과 '현재 체중'의 단순 수치뿐만 아니라, 그 차이값인 WeightDifference를 새로운 특성(Feature)으로 추가해 모델이 '감량해야 할 절대적인 양'을 직관적으로 학습시킴.
<br>
스케일링 (StandardScaler): 칼로리(수천 단위)와 키/몸무게(백 단위), 시간(십 단위) 등 단위가 다른 변수들을 평균 0, 분산 1의 동일한 스케일로 맞춰주어 특정 변수가 결과를 과도하게 지배하는 것을 방지.
<br>
타겟 변수 로그 변환 (np.log1p): 예측 대상인 '일수(Days)' 데이터가 한쪽으로 치우쳐 있을(Skewed) 가능성에 대비해 로그 변환을 적용. 
데이터의 분포를 정규분포에 가깝게 만들어 예측 정확도를 크게 높이는 기법 사용. 
(평가 시에는 np.expm1로 다시 원래 일수로 복원.)
<br>
<br>
--
<br>
🔍 3. 교차 검증 및 하이퍼파라미터 튜닝 (Grid Search & K-Fold CV)
<br>
모델의 최적 세팅을 찾기 위해 매우 견고한 검증 방식을 채택했습니다.
<br>
K-Fold 교차 검증 (K=5): 데이터를 5개로 쪼개어 번갈아 가며 학습과 검증을 수행합니다. 우연히 운이 좋아 성능이 높게 나오는 것을 방지하고, 모델의 '진짜 실력'을 객관적으로 평가합니다.
<br>
하이퍼파라미터 탐색: 층의 깊이, 학습률(Learning Rate), 배치 사이즈 등 모델의 성능을 결정하는 변수들의 여러 조합을 반복 테스트하여 최적의 조합(best_params)을 자동으로 찾아내도록 설계되었습니다.
<br>
<br>
--
<br>
⚙️ 4. 최신 최적화 및 조기 종료 전략 (Optimization & Early Stopping)
<br>
학습 과정을 효율적으로 통제하는 로직이 포함되어 있습니다.
<br>
Adam 옵티마이저 & L2 규제: 보편적으로 성능이 좋은 Adam을 사용하면서 weight_decay(L2 규제)를 추가해 가중치가 너무 커지는 것 방지.

학습률 스케줄러 (StepLR): 학습이 진행될수록 10 에포크(Epoch)마다 학습률을 절반(0.5)으로 줄여, 목표 지점 근처에서 더 세밀하게 정답을 찾아가도록 유도.

조기 종료 (Early Stopping): 50번의 학습을 다 채우지 않더라도, 7번(patience=7) 연속으로 검증 오차가 개선되지 않으면 학습을 즉시 정지. 이는 시간 낭비를 막고 과적합을 예방하는 필수 기술 사용.

<br>
<br>
--
<br>
💾 5. 평가 및 배포 준비 (Evaluation & Export)
<br>
평가 지표: 다각도로 평가를 하기 위해 실제 걸린 일수와 모델이 예측한 일수의 차이를 직관적으로 보여주는 MAE(평균 절대 오차), 오차의 크기를 제곱하여 큰 페널티를 부여하는 MSE, 모델의 설명력을 나타내는 R²(결정계수) 등 사용.
<br>
추론(Inference) 대비: 학습이 끝난 후 '모델의 가중치(.pth)', '스케일러(.joblib)', '입력 특성 구조(.json)'를 모두 개별 파일로 저장. 
->실제 서비스(웹/앱)에 AI 모델을 배포할 때 입력값이 훈련 때와 동일한 형태와 스케일을 갖도록 보장하는 파이프라인 구성.

<br><br>
---
<br><br>

<p align="center">
   [📈 예측 모델 성능 평가 결과]
</p>

<br>
1. R² (결정계수): 0.99 (99%의 설명력)
<br>

2. Mean Absolute Error (MAE): 8.65일
모델이 예측한 목표 달성일과 실제 달성일의 오차: 8.65일

<br>
(비즈니스적 가치): 다이어트나 벌크업 같은 장기적인 목표(예: 3~6개월)를 설정하는 사용자에게, ±8.65일(약 1주 남짓)이라는 매우 정밀하고 신뢰도 높은 예상 스케줄을 제공할 수 있음. "약 100일 뒤 목표 달성"이라고 예측했다면, 실제로는 91일~108일 사이에 달성하게 되는 식단과 운동 방식을 추천하여 효과적인 실무적 활용도를 가질 것이라고 예상.
<br><br>
---
<br><br>
<p align="center">
   [최종 주요 기능-목표 체중 달성일 예측]
</p>
<p align="center">
   기능: 사용자의 신체 정보(BMI, 체중, 활동량 등)를 바탕으로 딥러닝 모델이 목표 체중까지 소요되는 기간을 예측.
사용 데이터:
BMI 데이터 (체중, 키, 목표 체중)
사용자 활동 수준 (TDEE, BMR)
예측 결과:

{
  "username": "홍길동",
  "days_to_goal": 45,
  "message": "홍길동님, 목표 체중 달성 예상 소요 기간은 약 45일입니다."
}

</p>
<p align="center">
  <img width="300" height="546" alt="chat_image_1" src="https://github.com/user-attachments/assets/75626dfd-4567-4932-8e08-966fdd45b031" />
  &nbsp;&nbsp;&nbsp; <img width="300" height="546" alt="chat_image_2" src="https://github.com/user-attachments/assets/f8586ae7-5d51-4db3-92f3-5098488ea87a" />
</p>


<br><br>
---
<br><br>

<p align="center">
   [AI 기반 맞춤형 헬스케어 챗봇 서비스 구축]
</p>
<br>

<p align="center">
   <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/efd8b056-086c-47f8-b206-d372cc1b7e3f" />
   &nbsp; <img width="300" height="546" alt="image" src="https://github.com/user-attachments/assets/59c57476-9783-4639-be2a-e7a7b5b40b6b" />
</p>

<br><br>

1. 개요 (Overview)

사용자의 신체 정보(BMI 등)와 자체 구축한 알고리즘을 연동하여, 실시간으로 맞춤형 피트니스 계획과 식단을 제안하는 대화형 AI 챗봇 서비스 구현.
<br><br>

2. 기술 스택 (Tech Stack)
<br>
AI & LLM: OpenAI GPT-4o API
<br>
Chat Platform: Sendbird SDK / API
<br>
Data & Backend: Python (식단/운동 데이터 전처리 및 추천 시스템 연동)
<br><br>

3. 핵심 구현 사항 (Key Features & Contributions)
<br>
GPT-4o 기반 개인화 프롬프트 엔지니어링: 단순한 챗봇이 아닌, 사전에 Python으로 구현한 추천 데이터(BMI, 목표, 식단 리스트)를 AI의 시스템 프롬프트(System Prompt)에 주입.
<br>이를 통해 AI가 사용자 맞춤형 데이터를 기반으로 답변하도록 설계하여 할루시네이션(거짓 정보)을 최소화 및 알고리즘 기반으로 예측한 목표 달성일,식단,운동 정보 등 을 제공.
<br><br>
Sendbird를 활용한 실시간 대화 환경 구축: Sendbird 채팅 솔루션을 도입하여 안정적이고 매끄러운 실시간 메시징(Real-time Messaging) UI/UX를 구현.
<br>
사용자와 AI 간의 접근성 높은 인터페이스 구현.
<br><br>
데이터 기반 맞춤형 피트니스 컨설팅: 사용자가 입력한 신체 정보를 바탕으로 예측된 '목표 달성 예상일', '일일 칼로리 균형' 등의 분석 결과를 대화형으로 자연스럽게 안내하여 사용자 경험(UX) 극대화.
<br><br>

5. 향후 개선 목표 (Future Scope)
추천 식단 레시피 정보 연동: 사용자가 챗봇을 통해 추천받은 식단을 실제로 요리할 수 있도록, 상세 레시피와 식재료 정보를 실시간으로 제공하는 기능 확장 구현 예정.






