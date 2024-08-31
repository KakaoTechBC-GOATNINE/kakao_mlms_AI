import os, pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# BiLSTM 모델 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '../models/BiLSTM/restaurant_ranking_bilstm.h5')
tokenizer_path = os.path.join(current_dir, '../models/BiLSTM/restaurant_ranking_tokenizer.pickle')
# 모델 터미널에서 로드되는 상태보여주는 것 삭제
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

bilstm_model = load_model(model_path)
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# 텍스트 전처리 함수
def preprocess_text(text, tokenizer, max_len=300):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences

# 리뷰 점수를 예측하는 함수
def BiLSTM_predict_review_score(review_text):
    # 텍스트 전처리
    processed_text = preprocess_text(review_text, tokenizer)
    
    # 모델 예측
    prediction = bilstm_model.predict(processed_text, verbose=0)
    
    # 레이블 0과 레이블 1의 확률 추출
    label_0_prob = prediction[0][0]
    label_1_prob = 1 - label_0_prob  # 레이블 1의 확률은 (1 - 레이블 0의 확률)로 계산

    # 레이블 1에서 레이블 0의 값을 뺀 값 계산
    score_difference = label_1_prob - label_0_prob

    return score_difference
