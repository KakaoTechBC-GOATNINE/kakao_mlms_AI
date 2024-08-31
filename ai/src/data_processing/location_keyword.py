from fastapi import HTTPException
import requests

# # 카카오 API URL과 API 키 설정
KAKAO_API_URL = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
KAKAO_API_KEY = "330cca7280154dc662775507f3b51ac4"

# 좌표를 주소로 변환하는 함수
def get_location_name(latitude: float, longitude: float) -> str:
    headers = {
        'Authorization': f'KakaoAK {KAKAO_API_KEY}'  # 카카오 API 인증 헤더
    }
    params = {
        'x': longitude,
        'y': latitude
    }
    
    # 카카오 API에 요청을 보내고 응답을 받음
    response = requests.get(KAKAO_API_URL, headers=headers, params=params)
    
    # 응답이 성공적일 경우
    if response.status_code == 200:
        data = response.json()
        # 'documents' 필드가 존재하고 데이터가 있을 경우
        if 'documents' in data and data['documents']:
            address = data['documents'][0]['address']['address_name']
            return address
        else:
            raise HTTPException(status_code=404, detail="Address not found in the documents")
    else:
        raise HTTPException(status_code=response.status_code, detail="Error from Kakao API")
    
# 주소에서 동네 이름을 추출하는 함수
def extract_dong_name(address: str) -> str:
    parts = address.split()
    # 주소의 첫 번째부터 세 번째 부분을 결합하여 동네 이름으로 사용
    if len(parts) >= 3:
        return ' '.join(parts[:3])
    return address