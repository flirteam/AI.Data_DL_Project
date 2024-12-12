import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from joblib import load
import json

# 1. 사용자 정보 입력
user_info = {
    "username": "김가천",
    "current_weight": 68.0,
    "target_weight": 68.0,
    "height": 175.0,
    "age": 30,
    "gender": "Male",
    "activity_level": 4,  # 1~5 사이 값
    "goal_type": "균형 식단",  # 변경 가능: 저지방 고단백, 균형 식단, 벌크업
    "preferred_body_part": "가슴",  # 선호 운동 부위
    "bmr": 1628.75,
    "tdee": 2524.56,  # 총 에너지 소비량
    "bmi": 22.2,
    "target_bmi": 21.2,
}

# 2. 입력 유효성 확인
goal_type_mapping = {"저지방 고단백": 0, "균형 식단": 1, "벌크업": 2}
valid_goal_types = list(goal_type_mapping.keys())
valid_genders = ['Male', 'Female']
valid_body_parts = ['가슴', '등', '팔', '복부', '다리']

if user_info["goal_type"] not in valid_goal_types:
    raise ValueError(f"Invalid GoalType: {user_info['goal_type']}. Must be one of {valid_goal_types}.")
if user_info["gender"] not in valid_genders:
    raise ValueError(f"Invalid Gender: {user_info['gender']}. Must be 'Male' or 'Female'.")
if user_info["preferred_body_part"] not in valid_body_parts:
    raise ValueError(f"Invalid preferred_body_part: {user_info['preferred_body_part']}. Must be one of {valid_body_parts}.")

# 3. 저장된 Feature 구조 및 Scaler 로드
feature_path = "/content/drive/MyDrive/Colab Notebooks/P프데이터/feature_columns.json"
scaler_path = "/content/drive/MyDrive/Colab Notebooks/P프데이터/scaler.joblib"
model_path = "/content/drive/MyDrive/Colab Notebooks/P프데이터/P_model.pth"

# Feature 구조 로드
with open(feature_path, 'r') as f:
    expected_columns = json.load(f)
print("Feature 구조 로드 성공!")

# Scaler 로드
scaler = load(scaler_path)
print("Scaler 로드 성공!")

# PyTorch 모델 정의
class FeedforwardNNImproved(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes):
        super(FeedforwardNNImproved, self).__init__()
        layers = []
        in_dim = input_dim
        for size in hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(0.2))
            in_dim = size
        layers.append(nn.Linear(in_dim, 1))  # 마지막 출력 레이어
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# 모델 초기화 및 가중치 로드
hidden_layer_sizes = [128, 64, 32]  # 학습 시 사용된 구조
input_dim = len(expected_columns)  # 입력 Feature 수
model = FeedforwardNNImproved(input_dim, hidden_layer_sizes)
model.load_state_dict(torch.load(model_path))
model.eval()
print("모델 로드 성공!")

# 4. 입력 데이터 처리
input_data = pd.DataFrame([{
    "Age": user_info["age"],
    "Height": user_info["height"] / 100,  # cm → m
    "Weight": user_info["current_weight"],
    "TargetWeight": user_info["target_weight"],
    "BMR": user_info["bmr"],
    "TDEE": user_info["tdee"],
    "BMI": user_info["bmi"],
    "TargetBMI": user_info["target_bmi"],
    "Calorie_Target": user_info["tdee"] - 500,  # 예시 칼로리 목표
    "Calorie_Deficit": 500,
    "총 운동시간": 120,
    "하루소모칼로리": 400,
    "총 식사섭취 칼로리": 2000,
    "ActivityLevel": user_info["activity_level"],
    "Gender": user_info["gender"],
    "GoalType": user_info["goal_type"],
    "preferred_body_part": user_info["preferred_body_part"]
}])

# 범주형 변수 처리 (One-Hot Encoding)
categorical_features = ['Gender', 'GoalType', 'preferred_body_part']
input_data = pd.get_dummies(input_data, columns=categorical_features)

# 누락된 열 추가 및 정렬
input_data = input_data.reindex(columns=expected_columns, fill_value=0)

# 데이터 스케일링
X_input = scaler.transform(input_data)

# 5. 모델 예측
with torch.no_grad():
    X_tensor = torch.tensor(X_input, dtype=torch.float32)
    prediction = model(X_tensor).item()
    days_to_goal = np.expm1(prediction)  # 로그 변환 복원

# 6. 식단 및 운동 추천
def recommend_diet_and_exercise(user_info):
    if user_info["goal_type"] == "벌크업":
        diet = "고칼로리, 고단백 식단 추천"
        exercise = "웨이트 트레이닝 초점 프로그램 추천"
    elif user_info["goal_type"] == "저지방 고단백":
        diet = "저지방 고단백 식단 추천"
        exercise = "유산소 + 웨이트 균형 프로그램 추천"
    else:
        diet = "균형 잡힌 식단 추천"
        exercise = "유산소와 웨이트 비율 조정 프로그램 추천"
    return diet, exercise

diet_recommendation, exercise_recommendation = recommend_diet_and_exercise(user_info)

# 7. 결과 출력
print(f"사용자 {user_info['username']}님의 예상 목표 달성 기간은 {days_to_goal:.2f}일입니다.")
print(f"추천 식단: {diet_recommendation}")
print(f"추천 운동: {exercise_recommendation}")