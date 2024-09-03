from fastapi import FastAPI, HTTPException
from elasticsearch import Elasticsearch
from pydantic import BaseModel
from typing import List
from src.data_processing.location_keyword import get_location_name, extract_dong_name
# from src.data_processing.kakao_review_data_crawling import crawl_restaurant_reviews, save_to_csv
from src.data_processing.kakao_review_crawling_ES import crawl_restaurant_reviews
from src.api.ensemble_ranking import rank_restaurants_keywords
from src.api.HDBSCAN_runner import cluster_reviews_runner
from src.api.keyword_checking_ES import check_and_get_document
import os
import time
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()


# FastAPI 애플리케이션 생성
app = FastAPI()

# Elasticsearch 클라이언트 생성
es = Elasticsearch(
    [os.getenv("ELASTICSEARCH_URL")],
    http_auth=(os.getenv("ELASTICSEARCH_USER"), os.getenv("ELASTICSEARCH_PASSWORD"))
)

# huggingface/tokenizers 병렬 warning 해결
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        start_time = time.time()  # 전체 처리 시작 시간
        
        print("좌표를 이름으로 변경중")
        # 위치 이름 가져오기
        step_start_time = time.time()  # 각 단계 시작 시간
        address = get_location_name(request.latitude, request.longitude)
        step_end_time = time.time()  # 각 단계 끝 시간xx
        print(f"좌표를 이름으로 변경완료 (걸린 시간: {step_end_time - step_start_time:.2f}초)")
        
        # 동네 이름 추출
        step_start_time = time.time()
        dong_name = extract_dong_name(address)
        step_end_time = time.time()
        # 동네 이름과 키워드 결합
        combined = f"{dong_name} {request.keyword}"
        print(f"동네이름 및 키워드: {combined} (걸린 시간: {step_end_time - step_start_time:.2f}초)")
        
        print("데이터처리 시작")
        step_start_time = time.time()

        # DB에 키워드 검색결과 있는지 확인
        reviews = check_and_get_document(es, combined)
        if reviews == "none":
            # 리뷰 데이터 크롤링
            print("DB에 키값 존재 하지않음, 크롤링 시작")
            reviews = crawl_restaurant_reviews(es, combined, pages=3) # 최대 3페이지 크롤링
        step_end_time = time.time()
        print(f"데이터처리 완료 (걸린 시간: {step_end_time - step_start_time:.2f}초)")
        
        # 리뷰 데이터 수집용    
        # save_to_csv(reviews, 'restaurant_reviews.csv') 
        
        print("랭킹화 시작")
        # 가게 리뷰를 처리하고 랭킹화
        step_start_time = time.time()
        ranked_recommendations = rank_restaurants_keywords(reviews, request.keyword)
        step_end_time = time.time()
        print(f"랭킹화 완료 (걸린 시간: {step_end_time - step_start_time:.2f}초)")
        
        print("클러스터링 시작")
        # 랭킹화된 리뷰들 중 10개(상,하위 5개씩) 클러스터링 한 것 리스트에 추가 
        step_start_time = time.time()
        ranked_recommendations = cluster_reviews_runner(ranked_recommendations, reviews, top_n=10)
        step_end_time = time.time()
        print(f"클러스터링 완료 (걸린 시간: {step_end_time - step_start_time:.2f}초)")
        
        # 추천 레스토랑 리스트의 길이가 10개 이상인지 확인
        if len(ranked_recommendations) > 10:
            # 상위 5개와 하위 5개만 선택
            combined_recommendations = ranked_recommendations[:5] + ranked_recommendations[-5:]
        else:
            # 전체 리스트를 그대로 사용
            combined_recommendations = ranked_recommendations
        
        total_time = time.time() - start_time  # 전체 처리 시간 계산
        print(f"전체 처리 완료 (총 걸린 시간: {total_time:.2f}초)\n")

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