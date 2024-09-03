from src.data_processing.api_preprocessing import process_reviews
from src.models.HDBSCAN_clustering import analyze_reviews_by_clustering
from typing import List, Dict

"""
랭킹된 추천 리스트의 리뷰를 처리하고 클러스터링하는 함수.

매개변수:
- ranked_recommendations (List[Dict]): 랭킹된 가게 추천 리스트.
- reviews (List[tuple]): 가게 이름, 평점, 주소, 리뷰 리스트가 포함된 튜플 리스트.
- top_n (int): 클러스터링된 리뷰에서 추출할 상위 키워드 개수 (기본값 10).

반환값:
- List[Dict]: 각 가게 추천에 클러스터링된 키워드가 추가된 업데이트된 리스트.
"""
def cluster_reviews_runner(ranked_recommendations: List[Dict], reviews: List[tuple], top_n: int = 10) -> List[Dict]:
    for rec in ranked_recommendations[:5] + ranked_recommendations[-5:]:
        store_name = rec["store_name"]
        
        # reviews에서 해당 가게의 리뷰를 찾기
        for store in reviews:
            try:
                current_store_name, store_score, store_address, store_reviews = store
                
                if current_store_name == store_name:
                    # 리뷰를 전처리
                    processed_reviews = process_reviews('||'.join(store_reviews))
                    
                    # 클러스터링하여 상위 N개의 키워드 추출
                    clusters_terms = analyze_reviews_by_clustering(processed_reviews, top_n=top_n)
                    
                    # 클러스터링 결과를 ranked_recommendations에 추가
                    rec["clustered_terms"] = clusters_terms
                    break  # 해당 가게를 찾았으므로 내부 루프 종료
                    
            except ValueError:
                print(f"리뷰 데이터 처리 중 오류 발생: {store}")
    
    return ranked_recommendations