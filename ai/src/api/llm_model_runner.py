import math, re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.data_processing.api_preprocessing import process_reviews

# 모델 로드
model_name = "maywell/Synatra-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)   
model = AutoModelForCausalLM.from_pretrained(model_name)

# GPU 사용 설정 (가능할 경우)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
def predict_review_score(review_text: str) -> int:
    try:
        # 직접 프롬프트 생성
        prompt = f"Review: \"{review_text}\"\nRate the sentiment of this review as 0 (Bad), 1 (Average), or 2 (Good)."
        
        # 텍스트를 토크나이즈
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        # 모델 예측 생성
        generated_ids = model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],  # attention_mask를 포함
            max_new_tokens=50,  # 추가로 생성할 토큰 수
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id  # 패딩 토큰 ID를 설정
        )

        # 출력 디코딩
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        print(f"Model Response: {response}")

        # 감정 레이블 추출
        if "0" in response:
            return 0
        elif "1" in response:
            return 1
        elif "2" in response:
            return 2
        else:
            print("Failed to classify the sentiment. Defaulting to 1 (Average).")
            return 1  # 분류 실패 시 기본값으로 중립을 반환

    except Exception as e:
        print(f"Error processing review: {review_text}")
        print(f"Error: {e}")
        return 1  # 오류 발생 시 기본값 반환

def rank_restaurants(reviews, target_keywords):
    recommendations = []
    
    for store in reviews:
        try:
            store_name, store_score, store_address, store_reviews = store
            print("-------------------------------")
            print(f"Processing store: {store_name}")
            
            # 리뷰 텍스트 리스트를 전처리하여 가져옴
            processed_reviews = process_reviews('||'.join(store_reviews))
            total_score = 0.0
            keyword_count = 0
            num_reviews = len(processed_reviews)

            for review_text in processed_reviews:
                print("-------------------------------")
                print(f"Processing review: {review_text}")
                # 각 리뷰 텍스트에 대해 리뷰 점수를 예측
                review_score = predict_review_score(review_text)
                print(f"Review Sentiment Score: {review_score}")
                total_score += review_score
                
                # 키워드가 리뷰에 포함되어 있는지 확인
                for keyword in target_keywords:
                    if re.search(keyword, review_text, re.IGNORECASE):
                        keyword_count += 1
                        break  # 키워드가 하나라도 있으면 더 이상 체크하지 않음

            # 리뷰 점수의 평균 계산
            avg_score = total_score / num_reviews if num_reviews > 0 else 0

            # 리뷰 수에 따른 가중치 계산 (예: log 스케일로 신뢰도를 증가시킴)
            review_weight = math.log(1 + num_reviews)
            
            # 키워드 가중치 계산 (키워드가 포함된 리뷰의 비율)
            keyword_weight = keyword_count / num_reviews if num_reviews > 0 else 0
            
            # 최종 점수 계산 (리뷰 점수에 키워드 가중치와 리뷰 수 가중치를 곱함)
            weighted_score = avg_score * review_weight * (1 + keyword_weight)

            print(f"Weighted Score for {store_name}: {weighted_score}")
            print("\n")
            
            recommendations.append({
                "store_name": store_name,
                "address": store_address,
                "positive_score": weighted_score,
                "keyword_mention_count": keyword_count
            })

        except Exception as e:
            print(f"Error processing store: {store_name}")
            print(f"Error: {e}")

    # 가게를 신뢰도 반영 점수에 따라 내림차순으로 정렬
    ranked_recommendations = sorted(recommendations, key=lambda x: x['positive_score'], reverse=True)
    print(f"\n\n{ranked_recommendations}\n\n")

    return ranked_recommendations