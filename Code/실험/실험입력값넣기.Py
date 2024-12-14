import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from joblib import load
import json

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

# 파일 경로 설정
feature_path = "/content/drive/MyDrive/Colab Notebooks/P프끝/라스트100feature_columns.json"
scaler_path = "/content/drive/MyDrive/Colab Notebooks/P프끝/라스트100scaler.joblib"
model_path = "/content/drive/MyDrive/Colab Notebooks/P프끝/라스트100P_model.pth"

# Feature 구조 로드
with open(feature_path, 'r') as f:
    expected_columns = json.load(f)
print("Feature 구조 로드 성공!")

# Scaler 로드
scaler = load(scaler_path)
print("Scaler 로드 성공!")

hidden_layer_sizes = [128, 64, 32]  # 저장된 모델의 구조에 맞춤
input_dim = len(expected_columns)  # 입력 특성 수
model = FeedforwardNNImproved(input_dim, hidden_layer_sizes)
model.load_state_dict(torch.load(model_path))
model.eval()
print("모델 로드 성공!")

def predict_days_to_goal(user_info):
    # 입력 데이터 처리
    input_data = pd.DataFrame([{
        "Age": user_info["age"],
        "Height": user_info["height"] / 100,  # cm → m
        "Weight": user_info["current_weight"],
        "TargetWeight": user_info["target_weight"],
        "BMR": user_info["bmr"],
        "TDEE": user_info["tdee"],
        "BMI": user_info["bmi"],
        "TargetBMI": user_info["target_bmi"],
        "Calorie_Target": user_info["calorie_target"],
        "Calorie_Deficit": user_info["calorie_deficit"],
        "총 운동시간": user_info["total_exercise_time"],
        "하루소모칼로리": user_info["daily_calorie_burn"],
        "총 식사섭취 칼로리": user_info["total_food_calories"],
        "ActivityLevel": user_info["activity_level"],
        "Gender": user_info["gender"],
        "GoalType": user_info["goal_type"],
        "preferred_body_part": user_info["preferred_body_part"]
    }])

    # 범주형 변수 처리
    categorical_features = ['Gender', 'GoalType', 'preferred_body_part']
    input_data = pd.get_dummies(input_data, columns=categorical_features)

    # 누락된 열 추가 및 정렬
    missing_columns = set(expected_columns) - set(input_data.columns)
    for col in missing_columns:
        input_data[col] = 0
    input_data = input_data[expected_columns]  # 열 순서 재정렬

    # 데이터 스케일링
    X_input = scaler.transform(input_data)

    # 모델 예측
    with torch.no_grad():
        X_tensor = torch.tensor(X_input, dtype=torch.float32)
        prediction = model(X_tensor).item()
        days_to_goal = np.expm1(prediction)  # 로그 변환 복원

    # 체중 변화 계산
    weight_difference = user_info["current_weight"] - user_info["target_weight"]

    # 체중 변화가 특정 임계값(예: 10kg) 이상일 때만 조정
    if np.abs(weight_difference) > 15:
        adjustment_factor = 1 + 0.6 * (np.abs(weight_difference) - 10)  # 10kg 초과분에만 가중치 적용
        days_to_goal *= adjustment_factor

    return days_to_goal

# 테스트용 사용자 데이터
def test_predict():
    user_info_small_change = {
        "age": 30,
        "current_weight": 75.0,
        "target_weight": 70.0,  # 체중 변화 5kg
        "height": 170.0,
        "bmr": 1500.0,
        "tdee": 2500.0,
        "bmi": 26.0,
        "target_bmi": 24.2,
        "calorie_target": 1800,
        "calorie_deficit": 700,
        "total_exercise_time": 60,
        "daily_calorie_burn": 500,
        "total_food_calories": 1800,
        "activity_level": 3,
        "gender": "Female",
        "goal_type": "체중감량",
        "preferred_body_part": "복부"
    }

    user_info_large_change = {
    "age": 40,
    "current_weight": 90.0,  # 현재 체중
    "target_weight": 70.0,  # 목표 체중 (체중 변화 20kg)
    "height": 175.0,  # 키
    "bmr": 1800.0,  # 기초대사량
    "tdee": 2800.0,  # 총 에너지 소모
    "bmi": 29.4,  # 현재 BMI
    "target_bmi": 22.9,  # 목표 BMI
    "calorie_target": 1600,  # 칼로리 목표
    "calorie_deficit": 1200,  # 칼로리 결핍
    "total_exercise_time": 120,  # 총 운동 시간
    "daily_calorie_burn": 800,  # 하루 칼로리 소모
    "total_food_calories": 1600,  # 하루 식사 섭취 칼로리
    "activity_level": 4,  # 활동 수준
    "gender": "Male",  # 성별
    "goal_type": "체중감량",  # 목표 유형
    "preferred_body_part": "하체"  # 선호하는 신체 부위
}

    days_small_change = predict_days_to_goal(user_info_small_change)
    days_large_change = predict_days_to_goal(user_info_large_change)

    print(f"체중 변화 5kg -> 예상 목표 달성 기간: {days_small_change:.2f}일")
    print(f"체중 변화 20kg -> 예상 목표 달성 기간: {days_large_change:.2f}일")

# 테스트 실행
test_predict()