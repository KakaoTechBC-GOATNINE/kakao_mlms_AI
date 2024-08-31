import math, re
from src.data_processing.api_preprocessing import process_reviews
from src.api.KcELENTRA_runner import KcELECTRA_predict_review_score
from src.api.BiLSTM_runner import BiLSTM_predict_review_score
from src.models.HDBSCAN_clustering import analyze_reviews_by_clustering

# 앙상블 기법으로 두 모델 식당랭킹화진행
def rank_restaurants_keywords(reviews, keyword):
    recommendations = []
    # 모델별 가중치 정도
    electra_weight = 0.85
    bilstm_weight = 0.15
    
    for store in reviews:
        try:
            store_name, store_score, store_address, store_reviews = store
            # print("-------------------------------")
            # print(f"Processing store: {store_name}")
            
            # 리뷰 텍스트 리스트를 전처리하여 가져옴
            processed_reviews = process_reviews('||'.join(store_reviews))
            total_score_electra = 0.0
            num_reviews = len(processed_reviews)

            # KcELECTRA 모델을 사용하여 각 리뷰의 점수를 예측
            for review_text in processed_reviews:
                # print("-------------------------------")
                # print(f"Processing review: {review_text}")
                review_score = KcELECTRA_predict_review_score(review_text)
                # print(f"score: {review_score}\n")

                # 키워드 가중치 계산 (키워드가 포함된 리뷰의 비율)
                if review_score >= 1 and re.search(keyword, review_text):
                    review_score *= 1.5

                total_score_electra += review_score

            # KcELECTRA 평균 점수 계산
            avg_positive_score_electra = total_score_electra / num_reviews if num_reviews > 0 else 0

            # BiLSTM 모델을 사용하여 모든 리뷰를 하나의 문자열로 결합하여 점수 예측
            combined_reviews = ' '.join(processed_reviews)
            avg_positive_score_bilstm = BiLSTM_predict_review_score(combined_reviews)

            # 리뷰 수에 따른 가중치 계산 (예: log 스케일로 신뢰도를 증가시킴)
            weight = math.log(1 + num_reviews)  # log 스케일을 이용하여 가중치 계산
            
            # 최종 신뢰도 반영 점수 계산
            weighted_score = (electra_weight * avg_positive_score_electra + bilstm_weight * avg_positive_score_bilstm) * weight

            # print(f"Weighted Score for {store_name}: {weighted_score}")
            # print("\n")

            # # 가게별 리뷰 클러스터링 - 전체 클러스터링하던것 원하는 만큼 돌리도록 로직변경
            # clusters_terms = analyze_reviews_by_clustering(processed_reviews, top_n=5) #상위 5개단어 추출

            # print("가장 큰 군집에서 자주 언급된 단어들:")
            # for term, freq in clusters_terms:
            #     print(f"{term}: {freq}")
            
            recommendations.append({
                "store_name": store_name,
                "address": store_address,
                "positive_score": weighted_score
            })
        except Exception as e:
            print(f"Error processing store: {store_name}")
            print(f"Error: {e}")

    # 가게를 신뢰도 반영 점수에 따라 내림차순으로 정렬
    ranked_recommendations = sorted(recommendations, key=lambda x: x['positive_score'], reverse=True)
    # print(f"\n\n{ranked_recommendations}\n\n")

    return ranked_recommendations