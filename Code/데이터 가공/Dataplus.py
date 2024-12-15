#체중 변화에 따라 데이터 가공해서 기존 데이터에 새롭게 만든 데이터 추가해주는 코드
import pandas as pd
import numpy as np

def calculate_days_to_goal(goal_type, weight_diff):
    """GoalType과 WeightDiff를 기반으로 DaysToGoal 계산"""
    try:
        if goal_type == "저지방 고단백":
            if abs(weight_diff) <= 15:
                days = np.random.uniform(65, 120)
            elif abs(weight_diff) <= 25:
                days = np.random.uniform(120, 250)
            elif abs(weight_diff) <= 35:
                days = np.random.uniform(250, 400)
            else:
                days = np.random.uniform(400, 500)

        elif goal_type == "균형 식단":
            days = np.random.uniform(30, 60)  # 체중 변화량 큰 경우를 반영

        elif goal_type == "벌크업":
            if abs(weight_diff) <= 15:
                days = np.random.uniform(65, 120)
            elif abs(weight_diff) <= 25:
                days = np.random.uniform(120, 200)
            elif abs(weight_diff) <= 35:
                days = np.random.uniform(200, 350)
            else:
                days = np.random.uniform(350, 500)

        return int(np.clip(days, 3, 500))  # 최대값 500으로 축소
    except Exception as e:
        print(f"Error in calculate_days_to_goal: {e}")
        return 0  # 기본값 반환

def remove_outliers(data, target_column="DaysToGoal", min_value=3, max_value=500):
    """이상치 제거"""
    filtered_data = data[(data[target_column] >= min_value) & (data[target_column] <= max_value)]
    return filtered_data

def generate_samples(gender, weight_range, goal_types, num_samples=100):
    """랜덤 데이터 생성 (체중 변화량 13kg 이상 반영)"""
    try:
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
                target_weight = weight + np.random.uniform(-15, 15)
            elif goal == "저지방 고단백":
                target_weight = weight - np.random.uniform(10, 40)
            elif goal == "벌크업":
                target_weight = weight + np.random.uniform(10, 40)

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
    except Exception as e:
        print(f"Error in generate_samples: {e}")
        return pd.DataFrame()  # 빈 데이터프레임 반환

def add_large_weight_change_data(existing_file_path, output_file_path):
    """체중 변화량이 큰 데이터를 기존 데이터에 추가"""
    try:
        # 기존 데이터 로드
        existing_data = pd.read_csv(existing_file_path)

        # 새로운 데이터 생성
        male_data = generate_samples(
            gender="Male",
            weight_range=(70, 100),  # 체중 변화량 큰 범위 조정  남자 70키로에서 100키로까지의 범위로 생성
            goal_types=["저지방 고단백", "균형 식단", "벌크업"],
            num_samples=100
        )

        female_data = generate_samples(
            gender="Female",
            weight_range=(50, 75),  # 체중 변화량 큰 범위 조정    여자 50키로에서 75키로까지의 범위로 생성
            goal_types=["저지방 고단백", "균형 식단", "벌크업"],
            num_samples=100
        )

        # 기존 데이터와 새 데이터 병합
        new_data = pd.concat([male_data, female_data], ignore_index=True)
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)

        # 이상치 제거 후 저장
        combined_data = remove_outliers(combined_data, target_column="DaysToGoal", min_value=3, max_value=500)
        combined_data.to_csv(output_file_path, index=False)
        print(f"Combined data with large weight change saved to: {output_file_path}")
    except Exception as e:
        print(f"Error in add_large_weight_change_data: {e}")

# 사용 예시
existing_file_path = '/content/drive/MyDrive/Colab Notebooks/P프끝/데이터증강_추가.csv'  # 기존 데이터 경로
output_file_path = '/content/drive/MyDrive/Colab Notebooks/P프끝/데이터증강_추가2.csv'  # 병합된 데이터 저장 경로
add_large_weight_change_data(existing_file_path, output_file_path)
