def check_and_get_document(es, combined):
    location_with_combined = combined.replace(" ", "_")
    # combined에서 첫 번째 단어 추출 (지역명)
    first_word = location_with_combined.split('_')[0]

    # Elasticsearch에서 문서를 조회 (options() 사용)
    try:
        existing_doc = es.options(ignore_status=[404]).get(index=first_word, id=location_with_combined)
        document = existing_doc['_source']
        return document['restaurants_reviews']
    except KeyError:
        # 두 조건 중 하나라도 만족하지 않는 경우, none 리턴후 리뷰 크롤링 진행 
        return "none"
