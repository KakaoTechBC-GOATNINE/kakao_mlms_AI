from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from collections import Counter
import numpy as np
from scipy.sparse import issparse
import os

# sentence bert 모델 실행
current_dir = os.path.dirname(os.path.abspath(__file__))
model_directory = os.path.join(current_dir, "paraphrase-multilingual-MiniLM-L12-v2")
model = SentenceTransformer(model_directory)

# 최신 BERT 기반 모델을 사용한 문장 임베딩
def embed_reviews(reviews):
    if not reviews:  # 데이터가 없을 경우
        return np.array([])
    embeddings = model.encode(reviews, show_progress_bar=False)
    return embeddings

def cluster_reviews(embeddings):
    if len(embeddings) <= 2:  # 데이터 포인트가 2개 이하인 경우
        return [0] * len(embeddings)  # 하나의 군집으로 간주
    
    if len(embeddings) > 10:  # 데이터 포인트가 충분히 많을 때만 UMAP 사용
        try:
            n_neighbors = min(10, len(embeddings) // 2)
            reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=5, metric='cosine')
            umap_embeddings = reducer.fit_transform(embeddings)
            
            # 희소 행렬을 밀집 행렬로 변환
            if issparse(umap_embeddings):
                umap_embeddings = umap_embeddings.toarray()
        except ValueError as e:
            print(f"UMAP 차원 축소 중 오류 발생: {e}")
            umap_embeddings = embeddings  # UMAP 오류 시 원래 임베딩 사용
    else:
        umap_embeddings = embeddings  # 데이터가 적을 경우 UMAP 생략
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom')
    labels = clusterer.fit_predict(umap_embeddings)
    return labels

# 군집별로 자주 등장하는 단어를 추출
def extract_top_terms_by_cluster(processed_reviews, labels, top_n):
    clusters = {}
    for cluster_label in set(labels):
        cluster_reviews = [review for review, label in zip(processed_reviews, labels) if label == cluster_label]
        if cluster_reviews:  # 클러스터가 비어 있지 않은 경우에만 처리
            all_words = ' '.join(cluster_reviews).split()
            if len(all_words) == 0:
                clusters[cluster_label] = []
            else:
                most_common_words = Counter(all_words).most_common(top_n)
                clusters[cluster_label] = most_common_words
        else:
            clusters[cluster_label] = []  # 빈 클러스터 처리
    return clusters

# 가장 큰 군집에서 자주 등장하는 단어를 추출
def extract_top_terms_from_largest_cluster(processed_reviews, labels, top_n):
    if len(labels) == 0:  # 레이블이 없을 경우
        return []
    
    clusters = {}
    largest_cluster_label = None
    max_size = 0
    
    for cluster_label in set(labels):
        cluster_reviews = [review for review, label in zip(processed_reviews, labels) if label == cluster_label]
        if cluster_reviews:  # 클러스터가 비어 있지 않은 경우에만 처리
            all_words = ' '.join(cluster_reviews).split()
            most_common_words = Counter(all_words).most_common(top_n)
            clusters[cluster_label] = most_common_words
            
            # 가장 큰 군집 찾기
            if len(cluster_reviews) > max_size:
                max_size = len(cluster_reviews)
                largest_cluster_label = cluster_label

    if largest_cluster_label is not None:
        return clusters[largest_cluster_label]
    else:
        return []  # 군집이 없을 경우 빈 리스트 반환

# 전체 프로세스 실행
def analyze_reviews_by_clustering(reviews, top_n):
    if not reviews:  # 리뷰가 없는 경우
        return {}
    
    embeddings = embed_reviews(reviews)
    if len(embeddings) == 0:
        return {}
    
    labels = cluster_reviews(embeddings)
    clusters_terms = extract_top_terms_from_largest_cluster(reviews, labels, top_n)
    return clusters_terms




















# import os
# import time
# from sentence_transformers import SentenceTransformer
# import umap
# import hdbscan
# from collections import Counter
# import numpy as np
# from scipy.sparse import issparse

# # sentence bert 모델 실행
# current_dir = os.path.dirname(os.path.abspath(__file__))
# model_directory = os.path.join(current_dir, "paraphrase-multilingual-MiniLM-L12-v2")
# model = SentenceTransformer(model_directory)

# # 최신 BERT 기반 모델을 사용한 문장 임베딩
# def embed_reviews(reviews):
#     if not reviews:  # 데이터가 없을 경우
#         return np.array([])

#     start_time = time.time()
#     embeddings = model.encode(reviews, show_progress_bar=False)
#     end_time = time.time()
    
#     print(f"Time for embedding reviews: {end_time - start_time:.4f} seconds")
#     return embeddings

# def cluster_reviews(embeddings):
#     if len(embeddings) <= 2:  # 데이터 포인트가 2개 이하인 경우
#         return [0] * len(embeddings)  # 하나의 군집으로 간주
    
#     if len(embeddings) > 10:  # 데이터 포인트가 충분히 많을 때만 UMAP 사용
#         try:
#             start_time = time.time()
#             n_neighbors = min(10, len(embeddings) // 2)
#             reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=5, metric='cosine')
#             umap_embeddings = reducer.fit_transform(embeddings)
#             end_time = time.time()
            
#             print(f"Time for UMAP reduction: {end_time - start_time:.4f} seconds")
            
#             # 희소 행렬을 밀집 행렬로 변환
#             if issparse(umap_embeddings):
#                 umap_embeddings = umap_embeddings.toarray()
#         except ValueError as e:
#             print(f"UMAP 차원 축소 중 오류 발생: {e}")
#             umap_embeddings = embeddings  # UMAP 오류 시 원래 임베딩 사용
#     else:
#         umap_embeddings = embeddings  # 데이터가 적을 경우 UMAP 생략
    
#     start_time = time.time()
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom')
#     labels = clusterer.fit_predict(umap_embeddings)
#     end_time = time.time()
    
#     print(f"Time for HDBSCAN clustering: {end_time - start_time:.4f} seconds")
#     return labels

# # 군집별로 자주 등장하는 단어를 추출
# def extract_top_terms_by_cluster(processed_reviews, labels, top_n):
#     start_time = time.time()
#     clusters = {}
#     for cluster_label in set(labels):
#         cluster_reviews = [review for review, label in zip(processed_reviews, labels) if label == cluster_label]
#         if cluster_reviews:  # 클러스터가 비어 있지 않은 경우에만 처리
#             all_words = ' '.join(cluster_reviews).split()
#             if len(all_words) == 0:
#                 clusters[cluster_label] = []
#             else:
#                 most_common_words = Counter(all_words).most_common(top_n)
#                 clusters[cluster_label] = most_common_words
#         else:
#             clusters[cluster_label] = []  # 빈 클러스터 처리
#     end_time = time.time()
    
#     print(f"Time for extracting top terms by cluster: {end_time - start_time:.4f} seconds")
#     return clusters

# # 가장 큰 군집에서 자주 등장하는 단어를 추출
# def extract_top_terms_from_largest_cluster(processed_reviews, labels, top_n):
#     start_time = time.time()
    
#     if len(labels) == 0:  # 레이블이 없을 경우
#         return []
    
#     clusters = {}
#     largest_cluster_label = None
#     max_size = 0
    
#     for cluster_label in set(labels):
#         cluster_reviews = [review for review, label in zip(processed_reviews, labels) if label == cluster_label]
#         if cluster_reviews:  # 클러스터가 비어 있지 않은 경우에만 처리
#             all_words = ' '.join(cluster_reviews).split()
#             most_common_words = Counter(all_words).most_common(top_n)
#             clusters[cluster_label] = most_common_words
            
#             # 가장 큰 군집 찾기
#             if len(cluster_reviews) > max_size:
#                 max_size = len(cluster_reviews)
#                 largest_cluster_label = cluster_label

#     if largest_cluster_label is not None:
#         result = clusters[largest_cluster_label]
#     else:
#         result = []  # 군집이 없을 경우 빈 리스트 반환
    
#     end_time = time.time()
#     print(f"Time for extracting top terms from largest cluster: {end_time - start_time:.4f} seconds")
    
#     return result

# # 전체 프로세스 실행
# def analyze_reviews_by_clustering(reviews, top_n):
#     if not reviews:  # 리뷰가 없는 경우
#         return {}
    
#     start_time = time.time()
#     embeddings = embed_reviews(reviews)
#     end_time = time.time()
#     print(f"Total time for embedding reviews: {end_time - start_time:.4f} seconds")
    
#     if len(embeddings) == 0:
#         return {}
    
#     start_time = time.time()
#     labels = cluster_reviews(embeddings)
#     end_time = time.time()
#     print(f"Total time for clustering reviews: {end_time - start_time:.4f} seconds")
    
#     start_time = time.time()
#     clusters_terms = extract_top_terms_from_largest_cluster(reviews, labels, top_n)
#     end_time = time.time()
#     print(f"Total time for extracting terms from largest cluster: {end_time - start_time:.4f} seconds\n")
    
#     return clusters_terms