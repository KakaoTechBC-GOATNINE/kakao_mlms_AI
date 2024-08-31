from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 모델 로드
model_name = "ilmin/KcELECTRA_sentiment_model_v1.01"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.eval()

def KcELECTRA_predict_review_score(review_text: str) -> float:
    try:
        # 텍스트를 전처리하고, 토큰화된 입력 데이터 생성
        inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True)
        # print(f"Tokenized Input: {inputs}")
        
        # 모델 예측 (평가 모드에서)
        with torch.no_grad():
            outputs = model(**inputs)
        # print(f"Model Outputs: {outputs.logits}")
        
        # 소프트맥스를 통해 긍정 및 부정 확률 계산
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        negative_score = probabilities[0][0].item()  # 안좋은 리뷰 (0)의 확률
        neutral_score = probabilities[0][1].item()  # 평범한 리뷰 (1)의 확률
        positive_score = probabilities[0][2].item()  # 좋은 리뷰 (2)의 확률
        
        # 감정 점수 계산
        sentiment_score = (-1.5 * negative_score) + (0.5 * neutral_score) + (1.5 * positive_score)
        # print(f"P: {positive_score}")
        # print(f"n: {neutral_score}")
        # print(f"N: {negative_score}")
        # print(f"Sentiment Score: {sentiment_score}")    
        return sentiment_score
    
    except Exception as e:
        print(f"Error processing review: {review_text}")
        print(f"Error: {e}")
        return 0.0  # 오류 발생 시 기본값 반환
