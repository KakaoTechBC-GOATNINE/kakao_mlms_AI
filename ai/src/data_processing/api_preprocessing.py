import re
import pandas as pd

def preprocess_review(review):
    """리뷰 데이터에서 리뷰 텍스트만 추출합니다."""
    parts = review.split('|')

    # 리뷰 텍스트 부분
    review_text = parts[-1].strip() if len(parts) > 1 else ''  # 마지막 부분을 리뷰 텍스트로 간주

    # 리뷰 텍스트가 비어 있는 경우 None 반환
    if not review_text:
        return None

    return review_text

def process_reviews(reviews):
    """리뷰 리스트에서 각 리뷰의 텍스트만 추출하고, 클린화 및 스플릿을 진행하여 반환합니다."""
    processed_reviews = []

    # 리뷰가 NaN이거나 문자열이 아닌 경우 처리
    if pd.isna(reviews) or not isinstance(reviews, str):
        return processed_reviews  # 문제가 있는 경우 빈 리스트 반환

    # 예외 처리
    try:
        review_list = reviews.split('||')  # 여러 리뷰가 ||로 구분된 경우 분할
    except AttributeError:
        return processed_reviews  # 문제가 있는 데이터를 건너뛰고 빈 리스트 반환

    for review in review_list:
        try:
            processed_review = preprocess_review(review)
            if processed_review:
                # 클린화된 리뷰 텍스트
                cleaned_text = clean_review_text(processed_review)
                # 클린화된 리뷰 텍스트를 지정된 길이로 분할
                split_reviews = split_long_review(cleaned_text, label=1)  # 레이블은 상황에 맞게 설정
                processed_reviews.extend([text for text, _ in split_reviews])
        except Exception as e:
            print(f"Error processing review: {review}, Error: {e}")
            continue
    
    return processed_reviews

def clean_review_text(text):
    """리뷰 텍스트에서 한국어와 공백만 남기고 모든 불필요한 문자를 제거합니다."""
    cleaned_text = re.sub(r'[^가-힣\s]', '', text)
    return cleaned_text.strip()

def split_long_review(text, label, max_length=256):
    """
    리뷰 텍스트를 지정된 길이로 분할하고, 레이블을 유지한 채 새로운 행으로 추가합니다.
    """
    return [(text[i:i + max_length], label) for i in range(0, len(text), max_length)]
