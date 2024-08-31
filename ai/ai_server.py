from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.data_processing.location_keyword import get_location_name, extract_dong_name
from src.data_processing.kakao_review_data_crawling import crawl_restaurant_reviews, save_to_csv
from src.api.ensemble_ranking import rank_restaurants_keywords
from src.api.HDBSCAN_runner import cluster_reviews_runner

# FastAPI 애플리케이션 생성
app = FastAPI()

# 요청 본문 데이터 모델 정의
class KeywordLocationRequest(BaseModel):
    latitude: float  # 위도
    longitude: float  # 경도
    keyword: str  # 사용자가 검색하는 키워드

# 전처리된 리뷰의 데이터 모델 정의
class ReviewData(BaseModel):
    store_name: str
    address: str
    score: float
    review_texts: List[str]

# 모델이 예측한 결과를 담는 데이터 모델 정의
class RecommendationResult(BaseModel):
    store_name: str
    address: str
    score: float
    rank: int

@app.post("/api/v1/stores/ai")
def restaurant_recommendation_api(request: KeywordLocationRequest):
    try:
        # 위치 이름 가져오기
        address = get_location_name(request.latitude, request.longitude)

        # 동네 이름 추출
        dong_name = extract_dong_name(address)
        print(dong_name)

        # 동네 이름과 키워드 결합
        combined = f"{dong_name} {request.keyword}"

        # 리뷰 데이터 크롤링
        reviews = crawl_restaurant_reviews(combined, pages=3) # 최대 3페이지 크롤링

        # 리뷰 데이터 수집용    
        # save_to_csv(reviews, 'restaurant_reviews.csv') 

        # 가게 리뷰를 처리하고 랭킹화 및 클러스터링
        ranked_recommendations = rank_restaurants_keywords(reviews, request.keyword)

        # 랭킹화된 리뷰들중 10개(상,하위 5개씩) 클러스터링 한것 리스트에 추가 
        ranked_recommendations = cluster_reviews_runner(ranked_recommendations, reviews, top_n=5)

        # 추천 레스토랑 리스트의 길이가 10개 이상인지 확인
        if len(ranked_recommendations) > 10:
            # 상위 5개와 하위 5개만 선택
            combined_recommendations = ranked_recommendations[:5] + ranked_recommendations[-5:]
        else:
            # 전체 리스트를 그대로 사용
            combined_recommendations = ranked_recommendations
        
        

        return {
            "status": "success",
            "keyword": combined,
            "ranked_resturant": [
                {
                    "store_name": rec["store_name"],
                    "address": rec["address"],
                    "score": rec["positive_score"],
                    "clustered_terms": rec["clustered_terms"] 
                }
                for rec in combined_recommendations
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# 서버 실행 명령: uvicorn ai_server:app --reload