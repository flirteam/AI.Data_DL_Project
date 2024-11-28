import pandas as pd
####데이터 가공하는 첫번째 과정 #####
input_file = 'myPC/input_data.csv' 
output_file = 'myPC/output_data_with_gender.csv' 
df = pd.read_csv(input_file)

# Gender 열 추가 (Height 기준으로 1.70m 밑은 Female, 이상은 Male)
df['Gender'] = df['Height'].apply(lambda x: 'Female' if x < 1.70 else 'Male')

print(df.head(5))

df.to_csv(output_file, index=False)
print(f"가공 완료. 데이터가 '{output_file}'에 저장되었습니다.")
