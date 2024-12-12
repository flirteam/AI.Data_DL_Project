#최종 모델학습에 필요한 데이터 가공
import pandas as pd
import numpy as np
import random

# 파일 경로 설정
bmi_path = '/content/drive/MyDrive/Colab Notebooks/P프로젝트/lastgoal_BMI.csv'
exercise_path = '/content/drive/MyDrive/Colab Notebooks/P프로젝트/exercise.csv'
food_path = '/content/drive/MyDrive/Colab Notebooks/P프로젝트/lastfood_data.csv'
output_path = '/content/drive/MyDrive/Colab Notebooks/P프데이터/Finish_FEBMI.csv'

# 1. BMI 데이터 로드 및 전처리
bmi_data = pd.read_csv(bmi_path)

# BMI 계산
bmi_data['BMI'] = bmi_data['Weight'] / (bmi_data['Height'] ** 2)
bmi_data['TargetBMI'] = bmi_data['TargetWeight'] / (bmi_data['Height'] ** 2)

# BMR 계산
bmi_data['BMR'] = 10 * bmi_data['Weight'] + 6.25 * (bmi_data['Height'] * 100) - 5 * bmi_data['Age']
bmi_data['BMR'] += bmi_data['Gender'].map({'Male': 5, 'Female': -161})

# TDEE 계산
bmi_data['TDEE'] = bmi_data['BMR'] * bmi_data['ActivityLevel']

# 체중 감량 목표에 따라 칼로리 적자 설정
def calculate_calorie_deficit(bmi):
    if bmi < 18.5:  # 저체중
        return 0  # 칼로리 적자 없음
    elif 18.5 <= bmi < 23:  # 정상체중
        return 250  # 적은 칼로리 적자
    elif 23 <= bmi < 30:  # 과체중
        return 500  # 일반적인 칼로리 적자
    else:  # 비만
        return 750  # 높은 칼로리 적자

# 칼로리 적자 계산 및 적용
bmi_data['Calorie_Deficit'] = bmi_data['BMI'].apply(calculate_calorie_deficit)
bmi_data['Calorie_Target'] = bmi_data['TDEE'] - bmi_data['Calorie_Deficit']


# 목표별 영양소 비율
goal_ratios = {
    "저지방 고단백": {"carb_ratio": 0.4, "protein_ratio": 0.4, "fat_ratio": 0.2},
    "균형 식단": {"carb_ratio": 0.5, "protein_ratio": 0.3, "fat_ratio": 0.2},
    "벌크업": {"carb_ratio": 0.6, "protein_ratio": 0.3, "fat_ratio": 0.1}
}

# BMI 데이터에 user_id 추가
bmi_data['user_id'] = range(1, len(bmi_data) + 1)

# 2. 운동 데이터 로드 및 필터링
exercise_base = pd.read_csv(exercise_path)
exercise_base = exercise_base[["name", "body_part", "exercise_category", "base_calories_burned", "description"]]

# 3. 음식 데이터 로드
food_data = pd.read_csv(food_path)
food_data.columns = food_data.columns.str.strip().str.lower()

# 열 이름 매핑
column_mapping = {
    '식품명': 'name',
    '식품대분류명': 'category',
    '에너지(kcal)': 'calories',
    '탄수화물(g)': 'carbs',
    '단백질(g)': 'protein',
    '지방(g)': 'fat',
    '식품중량': 'serving_size'
}
food_data = food_data.rename(columns=column_mapping)
food_data = food_data[["name", "category", "calories", "carbs", "protein", "fat", "serving_size"]]

# 운동 시간 계산 함수
def calculate_recommended_duration(current_weight, target_weight):
    weight_diff = abs(current_weight - target_weight)
    base_duration = 15  # 기본 시간
    additional_minutes = int(weight_diff * 1.25)  # 체중 차이에 따른 추가 시간
    return min(base_duration + additional_minutes, 30)  # 최대 30분

# 음식 선택 로직
def select_food(foods, carb_target, protein_target, fat_target, portion_factor):
    foods['score'] = abs(foods['carbs'] - carb_target) + \
                     abs(foods['protein'] - protein_target) + \
                     abs(foods['fat'] - fat_target)

    best_foods = foods.nsmallest(10, 'score')
    selected = best_foods.sample()

    portion = (portion_factor / selected['calories'].values[0]) * 100

    return {
        "food_name": selected['name'].values[0],
        "portion": round(portion, 2),
        "carb": round(selected['carbs'].values[0] * (portion / 100), 2),
        "protein": round(selected['protein'].values[0] * (portion / 100), 2),
        "fat": round(selected['fat'].values[0] * (portion / 100), 2),
        "calories": round(selected['calories'].values[0] * (portion / 100), 2)
    }

# 식단 추천 로직
def recommend_diet(calorie_target, food_data, carb_target, protein_target, fat_target):
    meal_ratios = {
        "breakfast": 0.2,
        "lunch": 0.35,
        "snack": 0.15,
        "dinner": 0.3
    }

    recommended_meals = {}
    used_foods = set()

    for meal, ratio in meal_ratios.items():
        meal_calories = calorie_target * ratio
        meal_carb_target = carb_target * ratio
        meal_protein_target = protein_target * ratio
        meal_fat_target = fat_target * ratio

        available_foods = food_data[~food_data['name'].isin(used_foods)]
        if available_foods.empty:
            continue

        selected_meal = select_food(available_foods, meal_carb_target, meal_protein_target, meal_fat_target, meal_calories)
        recommended_meals[meal] = selected_meal
        used_foods.add(selected_meal['food_name'])

    return recommended_meals

# BMI에 따른 AMR 조정
def adjust_amr_based_on_bmi(amr, bmi):
    if bmi < 18.5:
        return amr * 1.1
    elif 18.5 <= bmi < 23:
        return amr
    elif bmi >= 23:
        return amr * 0.9
    return amr

# 운동 추천
def recommend_exercises(user_physical_info, exercise_base, preference=None):
    bmi = user_physical_info['BMI']

    if preference:
        candidate_exercises = exercise_base[exercise_base['body_part'] == preference]
    else:
        if bmi < 18.5:
            candidate_exercises = exercise_base[exercise_base['exercise_category'].isin(['상체근력운동', '하체근력운동', '코어강화운동'])]
        elif 18.5 <= bmi < 23.0:
            candidate_exercises = exercise_base[exercise_base['exercise_category'].isin(['전신운동', '코어강화운동', '유산소운동'])]
        else:
            candidate_exercises = exercise_base[exercise_base['exercise_category'].isin(['유산소운동', '유연성운동'])]

    if candidate_exercises.empty:
        candidate_exercises = exercise_base

    return candidate_exercises.sample(n=min(4, len(candidate_exercises)), replace=False)

# 7일치 운동 및 식단 통합 추천
def recommend_7_days_plan(bmi_data, exercise_base, food_data, output_csv_path):
    full_plan = []

    for _, user in bmi_data.iterrows():
        user_info = {'BMI': user['BMI'], 'user_id': user['user_id'], 'TDEE': user['TDEE'], 'GoalType': user['GoalType']}

        if user_info['GoalType'] not in goal_ratios:
            print(f"Error: {user_info['GoalType']} not in goal_ratios")
            continue

        adjusted_amr = adjust_amr_based_on_bmi(user_info['TDEE'], user_info['BMI'])
        carb_target = (adjusted_amr * goal_ratios[user_info['GoalType']]['carb_ratio']) / 4
        protein_target = (adjusted_amr * goal_ratios[user_info['GoalType']]['protein_ratio']) / 4
        fat_target = (adjusted_amr * goal_ratios[user_info['GoalType']]['fat_ratio']) / 9

        for day in range(1, 8):
            preference = random.choice(['가슴', '어깨', '등', '하체', None])
            exercises = recommend_exercises(user_info, exercise_base, preference)

            exercise_entry = {'user_id': user['user_id'], 'day': day, 'preferred_body_part': preference}
            total_calories_burned = 0
            total_duration = 0

            for i, (_, exercise) in enumerate(exercises.iterrows(), 1):
                duration = calculate_recommended_duration(user['Weight'], user['TargetWeight'])
                calories_burned = duration * exercise['base_calories_burned']
                total_calories_burned += calories_burned
                total_duration += duration

                exercise_entry[f'운동{i}'] = exercise['name']
                exercise_entry[f'운동부위{i}'] = exercise['body_part']
                exercise_entry[f'운동시간{i}'] = duration
                exercise_entry[f'소모칼로리{i}'] = calories_burned

            exercise_entry['총 운동시간'] = total_duration
            exercise_entry['하루소모칼로리'] = total_calories_burned

            daily_diet = recommend_diet(adjusted_amr, food_data, carb_target, protein_target, fat_target)
            diet_entry = {f"{meal}_{key}": value for meal, details in daily_diet.items() for key, value in details.items()}

            total_food_calories = sum(details['calories'] for details in daily_diet.values())
            diet_entry['총 식사섭취 칼로리'] = total_food_calories

            full_plan.append({**exercise_entry, **diet_entry})

    final_df = pd.DataFrame(full_plan)

    final_merged_data = pd.merge(bmi_data, final_df, on='user_id', how='left')

    columns_order = ['user_id', 'day'] + [col for col in final_merged_data.columns if col not in ['user_id', 'day']]
    final_merged_data = final_merged_data[columns_order]

    final_merged_data.to_csv(output_csv_path, index=False)
    print(f"7일치 데이터가 저장되었습니다: {output_csv_path}")

# 실행
recommend_7_days_plan(bmi_data, exercise_base, food_data, output_path)
