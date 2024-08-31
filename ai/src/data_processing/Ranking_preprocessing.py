import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os
import re
from collections import Counter
from konlpy.tag import Mecab

# CSV 파일 경로 설정
input_csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw', 'restaurant_reviews.csv')
output_csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'BiLSTM_review.csv')

# Mecab 객체 생성
mecab = Mecab()

def clean_review_text(review):
    # 리뷰 텍스트 정제
    review = re.sub(r'\|\|', ' ', review)  # 구분자 제거
    review = re.sub(r'레벨\d+\s\|\s\d+\s\|\s\d+\.\d+\s\|\s\d+%;\s\|', '', review)  # 불필요한 정보 제거
    review = review.strip()  # 양쪽 공백 제거
    return review

def preprocess_review(review):
    # 텍스트 정제
    review = clean_review_text(review)
    review = re.sub(r'[^가-힣\s]', '', review)  # 특수문자 제거
    tokens = mecab.morphs(review)  # 형태소 분석
    return ' '.join(tokens)

def add_detailed_rating_label(score):
    return 1 if score >= 3 else 0

def process_df(df,output_csv_file_path):
    df['processed_review'] = df['Reviews'].apply(preprocess_review)
    
    df['label'] = df['Score'].apply(add_detailed_rating_label)
    
    # 결측치 처리 (여기서는 processed_review만 확인)
    df = df.dropna(subset=['processed_review','label'])
    
    # 전처리된 데이터 저장
    df.to_csv(output_csv_file_path, index=False, encoding='utf-8-sig')
    print(f"전처리된 데이터가 {output_csv_file_path}에 저장되었습니다.")

# 데이터 처리 및 CSV 저장 실행
if __name__ == "__main__":
    df = pd.read_csv(input_csv_file_path, encoding='utf-8-sig')
    df = process_df(df,output_csv_file_path)
