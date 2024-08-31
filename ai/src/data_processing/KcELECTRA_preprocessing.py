import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(input_csv_file_path):
    """CSV 파일을 읽어 데이터프레임으로 로드합니다."""
    return pd.read_csv(input_csv_file_path, encoding='utf-8-sig', low_memory=False)

def preprocess_review(review):
    """리뷰 데이터에서 불필요한 텍스트를 제거하고 필요한 정보를 추출합니다."""
    parts = review.split('|')

    # 별점 변환 (100% -> 5.0 점수로 변환)
    if len(parts) > 3:
        score_str = parts[3].replace('%', '').replace(';', '').strip()
        try:
            parts[3] = str(float(score_str) / 20)
        except ValueError:
            parts[3] = '0'  # 변환할 수 없는 경우 기본값 설정

    # 리뷰 텍스트 부분
    review_text = parts[4].strip() if len(parts) > 4 else ''

    # 리뷰 텍스트가 비어 있는 경우 None 반환
    if not review_text:
        return None

    return parts[2].strip(), parts[3].strip(), review_text

def process_reviews(reviews):
    processed_reviews = []

    # 리뷰가 NaN이거나 문자열이 아닌 경우 처리
    if pd.isna(reviews) or not isinstance(reviews, str):
        print(processed_reviews)
        print(f"Skipping review processing. Invalid data type: {reviews}")
        return processed_reviews  # 문제가 있는 경우 빈 리스트 반환

    # 예외 처리
    try:
        review_list = reviews.split('||')
    except AttributeError as e:
        print(f"Error: Review data is not a string: {reviews}")
        return processed_reviews  # 문제가 있는 데이터를 건너뛰고 빈 리스트 반환

    for review in review_list:
        try:
            if len(review.split('|')) >= 5:
                processed_review = preprocess_review(review)
                if processed_review:
                    processed_reviews.append(processed_review)
        except Exception as e:
            print(f"Error processing review: {review}")
            print(f"Error: {e}")
            # 문제가 있는 특정 리뷰를 건너뛰고 다음 리뷰로 진행
    
    return processed_reviews

def clean_review_text(text):
    """리뷰 텍스트에서 한글, 영어, 숫자, 공백 및 일부 이모지를 남기고 모든 불필요한 문자를 제거합니다."""
    # 이모지 범위를 직접 추가 (예: 얼굴 이모지)
    cleaned_text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s!\"#\$%&\'\(\)\*\+,\-\.\/:;<=>\?@\[\]\^_`\{\|\}~\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]', '', text)
    return cleaned_text.strip()

def split_long_review(text, label, max_length=256):
    """리뷰 텍스트를 지정된 길이로 분할하고, 레이블을 유지한 채 새로운 행으로 추가합니다."""
    return [(text[i:i + max_length], label) for i in range(0, len(text), max_length)]

def label_by_absolute_difference(row):
    """사용자의 평균 평점과 레스토랑 평점 간의 차이를 기반으로 레이블을 지정합니다."""
    avg_rating = row['User_Avg_Rating']
    rating_diff = row['Restaurant_Rating'] - avg_rating

    # 레이블 범위 조정
    if abs(rating_diff) >= avg_rating * 0.40:
        return 0 if rating_diff < 0.40 else 4  # 아주 나쁨 / 아주 좋음
    elif abs(rating_diff) >= avg_rating * 0.02:  # 0.02로 범위를 조정
        return 1 if rating_diff < 0.02 else 3  # 나쁨 / 좋음
    else:
        return 2  # 평균

def preprocess_data(df):
    """전체 데이터를 전처리합니다."""
    final_data = []

    # csv내에서 데이터를 직접 수정을 한뒤에 생긴 문제 해결절차 or 크롤링시 네트워크 문제로 인한 데이터 이슈
    for column in df.columns:
        if df[column].dtype == 'float64':
            df[column] = df[column].fillna(0)  # 숫자형 열은 0으로 대체
        else:
            df[column] = df[column].fillna('')  # 문자열 열은 빈 문자열로 대체

    # 각 리뷰에 대해 전처리를 수행합니다.
    for reviews in df['Reviews']:
        processed_reviews = process_reviews(reviews)  # 리뷰를 개별적으로 전처리합니다.
        if processed_reviews:
            final_data.extend(processed_reviews)  # 전처리된 리뷰를 최종 데이터 리스트에 추가합니다.

    # 최종 데이터를 데이터프레임으로 변환합니다.
    processed_df = pd.DataFrame(final_data, columns=['User_Avg_Rating', 'Restaurant_Rating', 'Review_Text'])

    # 'User_Avg_Rating'과 'Restaurant_Rating'을 숫자(float) 형식으로 변환합니다.
    processed_df['User_Avg_Rating'] = processed_df['User_Avg_Rating'].astype(float)
    processed_df['Restaurant_Rating'] = processed_df['Restaurant_Rating'].astype(float)

    # 리뷰 텍스트를 전처리하여 한국어와 공백만 남기고 나머지 문자는 제거합니다.
    processed_df['Review_Text'] = processed_df['Review_Text'].apply(clean_review_text)

    # 리뷰 텍스트가 공백만 있는 경우를 제거합니다.
    processed_df = processed_df[processed_df['Review_Text'].str.strip().astype(bool)]

    # 사용자 평균 평점과 레스토랑 평점 간의 차이를 기반으로 레이블을 생성합니다.
    processed_df['Label'] = processed_df.apply(label_by_absolute_difference, axis=1)

    # 리뷰 텍스트를 256자로 제한하고, 초과하는 경우 분할하여 새로운 행으로 추가
    split_reviews = []
    for _, row in processed_df.iterrows():
        if len(row['Review_Text']) > 256:
            split_reviews.extend(split_long_review(row['Review_Text'], row['Label']))
        else:
            split_reviews.append((row['Review_Text'], row['Label']))

    # 필요한 열(Review_Text와 Label)만 남깁니다.
    processed_df = processed_df[['Review_Text', 'Label']]

    # 결측치가 있는 행을 제거합니다.
    processed_df.dropna(subset=['Review_Text', 'Label'], inplace=True)

    return processed_df 

def save_data(processed_df, output_csv_file_path):
    """전처리된 데이터를 CSV 파일로 저장합니다."""
    processed_df.to_csv(output_csv_file_path, index=False, encoding='utf-8-sig')

def split_data(processed_df, train_csv_file_path, test_csv_file_path, test_size=0.25):
    """레이블이 0, 2, 4인 데이터만 필터링하여 train/test로 나누고 각각 CSV 파일로 저장합니다."""
    # 레이블이 0, 2, 4인 데이터만 필터링
    filtered_df = processed_df[processed_df['Label'].isin([0, 2, 4])]

    # 레이블 4를 1로 변환, 2는 그대로 유지
    filtered_df.loc[:, 'Label'] = filtered_df['Label'].map({0: 0, 2: 1, 4: 2})
    
    # train/test split
    train_df, test_df = train_test_split(filtered_df, test_size=test_size, random_state=42, stratify=filtered_df['Label'])

    # 데이터를 CSV 파일로 저장
    train_df.to_csv(train_csv_file_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_csv_file_path, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    # CSV 파일 경로 설정
    input_csv_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw', 'restaurant_reviews.csv')
    output_csv_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'KcELECTRA_preprocessed_reviews.csv')
    train_csv_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'KcELECTRA_review_train_set_v1.01.csv')
    test_csv_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'KcELECTRA_review_test_set_v1.01.csv')

    # 데이터 로드
    df = load_data(input_csv_file_path)

    # 데이터 전처리
    processed_df = preprocess_data(df)

    # 전처리된 데이터 저장
    save_data(processed_df, output_csv_file_path)

    # 데이터 분할 및 저장
    split_data(processed_df, train_csv_file_path, test_csv_file_path)