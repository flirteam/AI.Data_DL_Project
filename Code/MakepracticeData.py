import pandas as pd
import numpy as np
import random

# 파일 경로 설정
bmi_path = '/content/drive/MyDrive/Colab Notebooks/P프데이터증강/제발4goal_BMI.csv'
exercise_path = '/content/drive/MyDrive/Colab Notebooks/P프로젝트/exercise.csv'
food_path = '/content/drive/MyDrive/Colab Notebooks/P프로젝트/lastfood_data.csv'
output_path = '/content/drive/MyDrive/Colab Notebooks/P프데이터증강/찐막이길_FEBMI.csv'

# 1. BMI 데이터 로드 및 전처리
bmi_data = pd.read_csv(bmi_path)

# 결측값 처리
bmi_data.fillna({
    'Weight': bmi_data['Weight'].mean(),
    'Height': bmi_data['Height'].mean(),
    'Age': bmi_data['Age'].median(),
    'TargetWeight': bmi_data['Weight'].mean() * 0.9
}, inplace=True)

# BMI 계산
bmi_data['BMI'] = bmi_data['Weight'] / (bmi_data['Height'] ** 2)
bmi_data['TargetBMI'] = bmi_data['TargetWeight'] / (bmi_data['Height'] ** 2)

# BMR 및 TDEE 계산
bmi_data['BMR'] = 10 * bmi_data['Weight'] + 6.25 * (bmi_data['Height'] * 100) - 5 * bmi_data['Age']
bmi_data['BMR'] += bmi_data['Gender'].map({'Male': 5, 'Female': -161})
bmi_data['TDEE'] = bmi_data['BMR'] * bmi_data['ActivityLevel']

# 칼로리 적자 설정
bmi_data['Calorie_Deficit'] = bmi_data['BMI'].apply(lambda bmi: 0 if bmi < 18.5 else (250 if bmi < 23 else (500 if bmi < 30 else 750)))
bmi_data['Calorie_Target'] = bmi_data['TDEE'] - bmi_data['Calorie_Deficit']

bmi_data['user_id'] = range(1, len(bmi_data) + 1)

# 2. 운동 데이터 로드 및 필터링
exercise_base = pd.read_csv(exercise_path)
exercise_base.dropna(subset=['name', 'body_part', 'exercise_category'], inplace=True)

# 3. 음식 데이터 로드
food_data = pd.read_csv(food_path)

# 열 이름 매핑 (먼저 수행)
food_data.rename(columns={
    '식품명': 'name',
    '식품대분류명': 'category',
    '에너지(kcal)': 'calories',
    '탄수화물(g)': 'carbs',
    '단백질(g)': 'protein',
    '지방(g)': 'fat',
    '식품중량': 'serving_size'
}, inplace=True)

# 열 이름을 소문자로 변환 (선택 사항)
food_data.columns = food_data.columns.str.strip().str.lower()

# 필수 열에 대해 결측값 제거
required_columns = ['name', 'calories', 'carbs', 'protein', 'fat']
food_data.dropna(subset=required_columns, inplace=True)


# 4. 추천 함수 정의

def calculate_recommended_duration(current_weight, target_weight):
    weight_diff = abs(current_weight - target_weight)
    base_duration = 15
    additional_minutes = int(weight_diff * 1.25)
    return min(base_duration + additional_minutes, 30)

def select_food(foods, carb_target, protein_target, fat_target, portion_factor):
    if foods.empty:
        return {
            "food_name": "기본 음식",
            "portion": 100,
            "carb": carb_target,
            "protein": protein_target,
            "fat": fat_target,
            "calories": portion_factor
        }

    foods['score'] = abs(foods['carbs'] - carb_target) + abs(foods['protein'] - protein_target) + abs(foods['fat'] - fat_target)
    selected = foods.nsmallest(1, 'score')

    portion = (portion_factor / selected['calories'].values[0]) * 100

    return {
        "food_name": selected['name'].values[0],
        "portion": round(portion, 2),
        "carb": round(selected['carbs'].values[0] * (portion / 100), 2),
        "protein": round(selected['protein'].values[0] * (portion / 100), 2),
        "fat": round(selected['fat'].values[0] * (portion / 100), 2),
        "calories": round(selected['calories'].values[0] * (portion / 100), 2)
    }

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
        return exercise_base.sample(n=min(4, len(exercise_base)), replace=False)

    return candidate_exercises.sample(n=min(4, len(candidate_exercises)), replace=False)

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

def adjust_amr_based_on_bmi(amr, bmi):
    if bmi < 18.5:
        return amr * 1.1
    elif 18.5 <= bmi < 23:
        return amr
    else:
        return amr * 0.9

def recommend_7_days_plan(bmi_data, exercise_base, food_data, output_csv_path):
    full_plan = []

    for _, user in bmi_data.iterrows():
        try:
            user_info = {'BMI': user['BMI'], 'user_id': user['user_id'], 'TDEE': user['TDEE'], 'GoalType': user.get('GoalType', '균형식단')}

            adjusted_amr = adjust_amr_based_on_bmi(user_info['TDEE'], user_info['BMI'])
            carb_target = (adjusted_amr * 0.5) / 4
            protein_target = (adjusted_amr * 0.3) / 4
            fat_target = (adjusted_amr * 0.2) / 9

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
        except Exception as e:
            print(f"사용자 {user['user_id']}의 데이터 생성 중 오류 발생: {e}")

    final_df = pd.DataFrame(full_plan)
    final_merged_data = pd.merge(bmi_data, final_df, on='user_id', how='left')
    final_merged_data.to_csv(output_csv_path, index=False)
    print(f"7일치 데이터가 저장되었습니다: {output_csv_path}")

# 실행
recommend_7_days_plan(bmi_data, exercise_base, food_data, output_path)