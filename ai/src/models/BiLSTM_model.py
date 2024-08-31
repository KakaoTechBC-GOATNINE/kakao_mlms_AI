import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, SpatialDropout1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class RestaurantRankingBiLSTMModel:
    def __init__(self, max_features=5000, max_len=300):
        self.max_features = max_features
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_features)
        self.model = None

    def load_data(self, file_path):
        # 데이터 로드
        self.df = pd.read_csv(file_path, encoding='utf-8-sig')
        print(f"Data loaded from {file_path}")
    
    def preprocess_data(self):
        # NaN 값을 빈 문자열로 대체하고 모든 값을 문자열로 변환
        self.df['processed_review'] = self.df['processed_review'].fillna('').astype(str)
        
        # 텍스트 데이터 시퀀스로 변환 및 패딩 적용
        self.tokenizer.fit_on_texts(self.df['processed_review'])
        X_seq = self.tokenizer.texts_to_sequences(self.df['processed_review'])
        X_padded = pad_sequences(X_seq, maxlen=self.max_len)
        
        # 라벨
        y = self.df['label'].values
        
        return X_padded, y
    
    def build_model(self):
        # BiLSTM 모델 정의
        self.model = Sequential()
        self.model.add(Embedding(self.max_features, 128, input_length=self.max_len))
        self.model.add(SpatialDropout1D(0.2))
        self.model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
        self.model.add(Dense(1, activation='sigmoid'))
        
        # 모델 컴파일
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=64):
        # 모델 학습
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)
    
    def evaluate_model(self, X_val, y_val):
        # 검증 데이터 성능 평가
        y_pred = (self.model.predict(X_val) > 0.5).astype(int)

        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
    
    def save_model(self, model_path, tokenizer_path):
        # 모델 저장
        self.model.save(model_path)
        print(f"학습한 모델이{model_path}에 저장되었습니다.")

        # 토크나이저 저장
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"토크나이저가 {tokenizer_path}에 저장되었습니다.")
    
    def fit_and_evaluate(self, file_path, model_save_path, tokenizer_save_path):
        # 데이터 로드 및 전처리
        self.load_data(file_path)
        X, y = self.preprocess_data()
        
        # 데이터 분할
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 모델 생성 및 학습
        self.build_model()
        self.train_model(X_train, y_train, X_val, y_val)
        
        # 모델 평가
        self.evaluate_model(X_val, y_val)
        
        # 모델 저장
        self.save_model(model_save_path, tokenizer_save_path)

# 모델 훈련 및 평가 실행
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_csv_file_path = os.path.join(base_dir, 'data', 'processed', 'BiLSTM_review.csv')
    model_save_path = os.path.join(base_dir, 'models', 'restaurant_ranking_bilstm.h5')
    tokenizer_save_path = os.path.join(base_dir, 'models', 'restaurant_ranking_tokenizer.pickle')
    #모델 저장부분에 경고가 뜨는데 딱히 문제가 있진않지만 확장자명을(.keras)로 바꾸는것이 호환성에 좋다고함.
    
    bilstm_model = RestaurantRankingBiLSTMModel()
    bilstm_model.fit_and_evaluate(input_csv_file_path, model_save_path, tokenizer_save_path)
