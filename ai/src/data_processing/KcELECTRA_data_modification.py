import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 현재 스크립트의 경로를 기반으로 파일 경로 설정
train_csv_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'KcELECTRA_review_train_set_v1.01.csv')

test_csv_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'KcELECTRA_review_test_set_v1.01.csv')

# 데이터 불러오기
train_data = pd.read_csv(train_csv_file_path)
test_data = pd.read_csv(test_csv_file_path)

# KcELECTRA 모델과 토크나이저 설정
model_name = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # num_labels를 2로 설정

# KcELECTRA를 이용해 감성 예측을 하고 확률로 변환하는 함수
def predict_sentiment_with_probabilities(review):
    inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=256)
    outputs = model(**inputs)
    logits = outputs.logits

    # 소프트맥스를 적용하여 확률로 변환
    probabilities = torch.softmax(logits, dim=-1)
    
    # 각 클래스의 확률을 퍼센트로 변환
    probabilities_percent = probabilities * 100

    return probabilities_percent

# 이진 분류 결과를 3개의 클래스로 매핑하는 함수
def map_to_three_classes(probabilities_percent, threshold=20):
    # 확률 차이를 기반으로 3개 클래스 중 하나를 선택
    diff = abs(probabilities_percent[0, 0] - probabilities_percent[0, 1])

    # 텐서를 파이썬 숫자로 변환하여 비교
    if diff.item() >= threshold:
        if probabilities_percent[0, 0].item() > probabilities_percent[0, 1].item():
            return 0  # "별로"
        else:
            return 2  # "좋다"
    else:
        return 1  # "그냥저냥"

# 불일치하는 라벨을 직접 수정하는 함수
def interactive_label_correction(data):
    corrected_data = data.copy()
    for index, row in corrected_data.iterrows():
        review = row['Review_Text']
        true_label = row['Label']
        probabilities_percent = predict_sentiment_with_probabilities(review)

        predicted_label = map_to_three_classes(probabilities_percent)

        if true_label != predicted_label:
            print(f"리뷰: {review}")
            print(f"정답 라벨: {true_label} | 예측 라벨: {predicted_label}")
            print(f"클래스별 확률: {probabilities_percent[0, 0].item():.2f}% (별로), {probabilities_percent[0, 1].item():.2f}% (좋다)")
            new_label = input("올바른 라벨을 입력하세요 (0: 별로, 1: 그냥저냥, 2: 좋다) (아무 입력 없이 Enter를 누르면 기존 라벨 유지): ").strip()
            print("\n")

            if new_label in ['0', '1', '2']:
                corrected_data.at[index, 'Label'] = int(new_label)
            else:
                print("잘못된 입력입니다. 기존 라벨을 유지합니다.")

    return corrected_data

# 함수 적용하여 트레인과 테스트 데이터 수정
corrected_train_data = interactive_label_correction(train_data)
corrected_test_data = interactive_label_correction(test_data)

# 수정된 데이터를 새로운 CSV 파일로 저장
corrected_train_csv_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'KcELECTRA_review_train_set_corrected_v1.02.csv')

corrected_test_csv_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'KcELECTRA_review_test_set_corrected_v1.02.csv')

corrected_train_data.to_csv(corrected_train_csv_file_path, index=False, encoding='utf-8-sig')
corrected_test_data.to_csv(corrected_test_csv_file_path, index=False, encoding='utf-8-sig')

print("성공!")