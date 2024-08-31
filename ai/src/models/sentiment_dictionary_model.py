import os
import pandas as pd
import numpy as np
import requests
import re
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Okt 객체 생성
okt = Okt()

# 외부 한국어 불용어 리스트 가져오기
url = 'https://gist.githubusercontent.com/spikeekips/40d99e1ddc41b06412a6b913d9c9b5b8/raw/3ec2b98210903bd4c8f5b8481e0a8fd0de5e8fc3/stopwords-ko.txt'
response = requests.get(url)
stopwords = set(response.text.splitlines())

# CSV 파일 경로 설정
input_csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw', 'restaurant_reviews.csv')
output_csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'ranked_restaurant_reviews.csv')

# CSV 파일 읽기
df = pd.read_csv(input_csv_file_path, encoding='utf-8-sig')

class ReviewRankingModel:
    def __init__(self, max_features=5000):
        # Okt 객체와 TF-IDF 벡터라이저, 로지스틱 회귀 모델을 생성
        self.okt = Okt()
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = LogisticRegression()
        self.max_features = max_features
    
    def preprocess_reviews_data(self, reviews):
        # 한국어 전처리, 특수문자 제거
        reviews = re.sub(r'\|\|', ' ', reviews)
        reviews = re.sub(r'[^가-힣\s]', '', reviews)
        reviews = reviews.lower() #소문자로 변환은 항상 해주는게 좋다고함
        return reviews
    
    def preprocess_review(self, reviews):
        # 형태소 분석
        tokens = self.okt.pos(reviews, norm=True, stem=True)
        tokens = [word for word, tag in tokens if tag in ['Noun', 'Adjective'] and word not in stopwords]
        return ' '.join(tokens)
    
    def load_data(self, file_path):
        # 전처리한 데이터 저장
        self.df = pd.read_csv(file_path, encoding='utf-8-sig')
        self.df['processed_review'] = self.df['Reviews'].apply(self.preprocess_review)
        
    def add_rating_label(self):
        # 평점에 따른 레이블 생성
        def rating_label(rating):
            if rating > 3.0:
                return 2
            elif rating == 3.0:
                return 1
            else:
                return 0

        # 레이블을 데이터프레임에 추가
        self.df['rating_label'] = self.df['Score'].apply(rating_label)
        
    def train_sentiment_model(self, test_size=0.2, random_state=42):
        # 리뷰 데이터를 학습 및 테스트 세트로 나누고, TF-IDF 벡터화한 후 로지스틱 회귀 모델을 훈련

        # 긍정적/부정적 키워드를 기반으로 레이블 생성
        positive_keywords = ['좋다', '맛있다', '추천', '최고', '훌륭하다', '기분좋다']  # 긍정적인 리뷰 더 있으면 추가
        negative_keywords = ['나쁘다', '별로', '불친절', '실망', '싫다', '최악']  # 부정적인 리뷰 더 있으면 추가

        def label_review(review):
            if any(positive_word in review for positive_word in positive_keywords):
                return 1
            elif any(negative_word in review for negative_word in negative_keywords):
                return 0
            else:
                return None  # 레이블을 지정할 수 없는 경우

        # 데이터프레임에 'label' 열을 추가하고, 각 리뷰에 대해 레이블을 생성
        self.df['label'] = self.df['Reviews'].apply(label_review)

        # 레이블이 지정되지 않은 리뷰는 제거
        self.df = self.df.dropna(subset=['label'])
        self.df['label'] = self.df['label'].astype(int)  # 레이블을 정수형으로 변환

        # 텍스트 데이터 TF-IDF 벡터화 및 평점 결합
        X_tfidf = self.vectorizer.fit_transform(self.df['processed_review'])
        X = np.hstack([X_tfidf.toarray(), self.df[['Score']].values])  # TF-IDF 벡터와 평점 결합
        
        X_train, X_test, y_train, y_test = train_test_split(X, self.df['label'], test_size=test_size, random_state=random_state)
        
        self.model.fit(X_train, y_train)
        
        # 모델 성능 평가
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        print(f"Sentiment Model Accuracy: {accuracy}")
        print(f"Sentiment Model F1 Score: {f1}")
        print(f"Sentiment Model Precision: {precision}")
        print(f"Sentiment Model Recall: {recall}")
    
    def rank_reviews(self):
        # 리뷰 간의 코사인 유사도를 계산하고, 감정 분석 결과와 결합하여 리뷰를 랭킹화
        
        # TF-IDF 벡터화 및 평점 결합
        tfidf_matrix = self.vectorizer.transform(self.df['processed_review'])
        tfidf_matrix = np.hstack([tfidf_matrix.toarray(), self.df[['Score']].values])  # TF-IDF 벡터와 평점 결합
        
        # 코사인 유사도 계산
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        self.df['similarity_score'] = cosine_sim_matrix.mean(axis=1)
        
        # 감정 분석 예측
        self.df['sentiment'] = self.model.predict(tfidf_matrix)
        
        # 최종 점수 계산 및 랭킹
        self.df['final_score'] = (self.df['sentiment'] * 0.4) + (self.df['Score'] * 0.4) + (self.df['rating_label'] * 0.2)  # 감정 점수, 평점, 레이블의 가중 평균
        self.df['rank'] = self.df['final_score'].rank(ascending=False)
        
        # 랭킹 순서대로 정렬 (1등이 맨 위)
        self.df = self.df.sort_values(by='rank', ascending=True)
    
    def save_ranked_reviews(self, output_file_path):
        # rank csv파일 저장 내용
        rank_df = pd.DataFrame({
            'rank': self.df['rank'].astype(int), 
            'Name' : self.df['Name'],
            'Rating': self.df['Score'],
            'rating_label': self.df['rating_label'],
            'final_score': self.df['final_score'],
            'sentiment': self.df['sentiment']
        })

        # 랭킹화된 리뷰를 저장
        rank_df.to_csv(output_file_path, index=False)
    
    def fit_and_rank(self, input_file_path, output_file_path):
        self.load_data(input_file_path)
        self.add_rating_label()
        self.train_sentiment_model()
        self.rank_reviews()
        self.save_ranked_reviews(output_file_path)
        print(f"Ranked reviews saved to {output_file_path}")

# 리뷰 랭킹 모델 생성 및 실행
review_ranking_model = ReviewRankingModel()
review_ranking_model.fit_and_rank(input_csv_file_path, output_csv_file_path)