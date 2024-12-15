#bmi 데이터 생체데이터로 만들기 
#목표일 재계산해주는 코드
import pandas as pd
import numpy as np

def augment_and_update_days_to_goal(file_path, output_path):
    # 원본 데이터 읽기
    data = pd.read_csv(file_path)

    # DaysToGoal 계산 함수
    def calculate_days_to_goal(goal_type, weight_diff):
        if goal_type == "저지방 고단백":
            if abs(weight_diff) <= 5:
                days = np.random.uniform(3, 18)  # 몸무게 차이가 ±5kg일 때
            elif abs(weight_diff) <= 10:
                days = np.random.uniform(88, 104)  # 몸무게 차이가 ±10kg일 때
            elif abs(weight_diff) <= 20:
                normalized_diff = (abs(weight_diff) - 10) / 10
                days = np.random.uniform(204, 294) * normalized_diff + 204
            elif abs(weight_diff) <= 30:
                normalized_diff = (abs(weight_diff) - 20) / 10
                days = np.random.uniform(387, 627) * normalized_diff + 387
            else:
                days = np.random.uniform(387, 627)  # 기본값

            return int(np.clip(days, 3, 627))

        elif goal_type == "균형 식단":
            return int(np.random.uniform(12, 33))  # 균형 식단은 12~33일 사이

        elif goal_type == "벌크업":
            if abs(weight_diff) <= 5:
                days = np.random.uniform(3, 18)
            elif abs(weight_diff) <= 10:
                days = np.random.uniform(88, 104)
            elif abs(weight_diff) <= 20:
                normalized_diff = (abs(weight_diff) - 10) / 10
                days = np.random.uniform(204, 294) * normalized_diff + 204
            elif abs(weight_diff) <= 30:
                normalized_diff = (abs(weight_diff) - 20) / 10
                days = np.random.uniform(387, 627) * normalized_diff + 387
            else:
                days = np.random.uniform(387, 627)

            return int(np.clip(days, 3, 627))

    # 기존 데이터의 DaysToGoal 업데이트
    for idx, row in data.iterrows():
        weight_diff = row['TargetWeight'] - row['Weight']
        data.at[idx, 'DaysToGoal'] = calculate_days_to_goal(row['GoalType'], weight_diff)

    # 데이터 증강 설정
    num_samples = 150  # 남자/여자 각각 150명 추가

    # 랜덤 데이터 생성
    def generate_samples(gender, weight_range, goal_types):
        ages = np.random.randint(20, 41, num_samples)
        heights = np.random.uniform(1.5, 1.9, num_samples) if gender == "Male" else np.random.uniform(1.4, 1.7, num_samples)
        weights = np.random.uniform(weight_range[0], weight_range[1], num_samples)
        bmis = weights / (heights ** 2)

        goal_type_list = []
        target_weights = []
        days_to_goal = []

        for weight in weights:
            goal = np.random.choice(goal_types)
            goal_type_list.append(goal)

            if goal == "균형 식단":
                target_weight = np.random.uniform(-5, 5) + weight
            elif goal == "저지방 고단백":
                target_weight = weight - np.random.uniform(0, 10)
            elif goal == "벌크업":
                target_weight = weight + np.random.uniform(0, 10)

            target_weights.append(target_weight)

            # DaysToGoal 계산
            weight_diff = target_weight - weight
            days_to_goal.append(calculate_days_to_goal(goal, weight_diff))

        bmi_classes = [
            "Underweight" if bmi < 18.5 else
            "Normal weight" if 18.5 <= bmi < 25.0 else
            "Overweight" if 25.0 <= bmi < 30.0 else
            "Obese Class 1" if 30.0 <= bmi < 35.0 else
            "Obese Class 2" if 35.0 <= bmi < 40.0 else
            "Obese Class 3"
            for bmi in bmis
        ]

        return pd.DataFrame({
            "Age": ages,
            "Height": heights,
            "Weight": weights,
            "Bmi": bmis,
            "BmiClass": bmi_classes,
            "Gender": [gender] * num_samples,
            "TargetWeight": target_weights,
            "GoalType": goal_type_list,
            "ActivityLevel": np.random.randint(1, 5, num_samples),
            "DaysToGoal": days_to_goal,
            "Achieved": np.random.choice([0, 1], num_samples)
        })

    # 남자 데이터 생성
    male_data = generate_samples(
        gender="Male",
        weight_range=(60, 80),
        goal_types=["저지방 고단백", "균형 식단", "벌크업"]
    )

    # 여자 데이터 생성
    female_data = generate_samples(
        gender="Female",
        weight_range=(40, 65),
        goal_types=["저지방 고단백", "균형 식단", "벌크업"]
    )

    # 기존 데이터와 결합
    augmented_data = pd.concat([data, male_data, female_data], ignore_index=True)

    # 저장
    augmented_data.to_csv(output_path, index=False)
    print(f"Augmented and updated data saved to: {output_path}")

# 사용 예시
input_file_path = '/content/drive/MyDrive/Colab Notebooks/P프데이터증강/제발2goal_BMI.csv'  # 입력 파일 경로
output_file_path = '/content/drive/MyDrive/Colab Notebooks/P프데이터증강/Maxgoal_BMI.csv'  # 출력 파일 경로
augment_and_update_days_to_goal(input_file_path, output_file_path)
