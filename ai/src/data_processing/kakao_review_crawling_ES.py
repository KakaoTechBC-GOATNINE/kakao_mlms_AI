import time, os, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch
from datetime import datetime

warnings.filterwarnings('ignore')

def setup_driver():
    """크롬 드라이버를 설정하고 반환합니다."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("start-maximized")
    options.add_argument("disable-infobars")
    options.add_argument("--disable-browser-side-navigation")
    options.add_argument("--disable-blink-features=AutomationControlled")
    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)
    
    try:
        # 로컬에 설치되어있는 크롬사용하도록 변경
        driver = webdriver.Chrome(options=options)
    except Exception as e:
        print(f"Error using ChromeDriverManager: {e}")
        print("Falling back to the default ChromeDriver")
        # 이슈 생길경우, 크롬드라이버 설치하도록 시도
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    return driver

def setup_elasticsearch():
    """Elasticsearch 클라이언트를 설정합니다."""
    es = Elasticsearch(
        hosts=[{
            'host': 'localhost',
            'port': 9200,
            'scheme': 'http'
        }]
    )
    return es

def search_location(driver, location):
    """카카오 맵에서 특정 위치를 검색합니다."""
    url = "https://map.kakao.com/"
    driver.get(url)
    search_area = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="search.keyword.query"]'))
    )
    search_area.send_keys(location)
    driver.find_element(By.XPATH, '//*[@id="search.keyword.submit"]').send_keys(Keys.ENTER)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="info.main.options"]/li[2]/a'))
    )
    # 가려진 요소를 숨기기
    driver.execute_script("document.getElementById('dimmedLayer').style.display = 'none';")
    # 요소가 클릭 가능할 때까지 기다림
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, '//*[@id="info.main.options"]/li[2]/a'))
    ).click()

def extract_reviews(driver):
    """음식점의 리뷰를 추출합니다."""
    reviews = []
    reviews_count = 0
    while True:
        try:
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            more_reviews_button = soup.select_one('span:contains("후기 더보기")')
            
            if not more_reviews_button or reviews_count == 10:
                break

            more_reviews_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//span[contains(text(), "후기 더보기")]'))
            )
            
            more_reviews_button.click()
            reviews_count += 1  # 더보기버튼 한번 누를때마다 추가, (리뷰수 5개추가)
            time.sleep(0.5)

        except Exception as e:
            print(f"Exception while clicking more reviews button: {e}")
            break

    try:
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        review_elements = soup.select('.list_evaluation > li')
        
        for review in review_elements:
            try:
                level = review.select_one('a > div > div > span:nth-of-type(2)').text.strip()
                num_reviews = review.select_one('div > span:nth-of-type(3)').text.strip()
                avg_reviews = review.select_one('div > span:nth-of-type(5)').text.strip()
                star = review.select_one('.ico_star.inner_star')['style'].split(':')[1].strip()
                text = review.select_one('.txt_comment > span').text.strip()
                combined_review = f"{level} | {num_reviews} | {avg_reviews} | {star} | {text}"
                reviews.append(combined_review)
            except (IndexError, AttributeError) as e:
                print(f"Error extracting review parts: {e}")
                continue

    except Exception as e:
        print(f"Exception while extracting reviews: {e}")

    if not reviews:
        reviews.append(' ')

    driver.switch_to.window(driver.window_handles[0])
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
    return reviews

def extract_restaurant_info(driver, location, page_number):
    """음식점의 정보를 추출하고 리뷰를 추가합니다."""
    search_location(driver, location)

    if page_number > 1:
        try:
            xpath = f'/html/body/div[5]/div[2]/div[1]/div[7]/div[6]/div/a[{page_number}]'
            page_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )
            driver.execute_script("arguments[0].click();", page_button)
            time.sleep(1)  
        except Exception as e:
            print(f"Error navigating to page {page_number}: {e}")
            return []

    time.sleep(0.2)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    restaurant_elements = soup.select('.placelist > .PlaceItem')
    restaurant_list = []

    for i, restaurant in enumerate(restaurant_elements):
        name = restaurant.select('.head_item > .tit_name > .link_name')[0].text
        score = restaurant.select('.rating > .score > em')[0].text
        addr = restaurant.select('.addr > p')[0].text
        more_reviews_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, f'//*[@id="info.search.place.list"]/li[{i+1}]/div[5]/div[4]/a[1]'))
        )
        driver.execute_script("arguments[0].click();", more_reviews_button)
        driver.switch_to.window(driver.window_handles[-1])
        time.sleep(1)
        reviews = extract_reviews(driver)
        restaurant_list.append([name, score, addr, reviews])

    driver.quit()
    return restaurant_list

def get_total_pages(driver):
    """총 페이지 수를 반환합니다."""
    try:
        no_result = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div#info\\.noPlace'))
        )
        time.sleep(0.1)
        if no_result.is_displayed():
            return 0 
    except Exception as e:
        print(f"Error on get_total_pages/noPlace: {e}")
    
    try:
        page_elements = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div#info\\.search\\.page div.pageWrap > a'))
        )
        time.sleep(0.1)
        pages = len([page for page in page_elements if 'HIDDEN' not in page.get_attribute('class')])
    except Exception as e:
        print(f"Error on get_total_pages: {e}")

    return pages

def save_to_elasticsearch(es, index_name, location_with_underscores, restaurants):
    """크롤링된 데이터를 Elasticsearch에 저장하거나 기존 문서에 추가합니다."""
    try:
        existing_doc = es.get(index=index_name, id=location_with_underscores, ignore=[404])
        if existing_doc['found']:
            existing_restaurants = existing_doc['_source'].get('restaurants', [])
            existing_restaurants.extend(restaurants)  # 기존 리스트에 새로운 식당 리스트 추가
        else:
            existing_restaurants = restaurants  # 문서가 없으면 새로운 리스트를 사용
    except KeyError:
        existing_restaurants = restaurants  # 문서가 없으면 새로운 리스트를 사용

    # 문서 업데이트
    doc = {
        'location_keyword': location_with_underscores,
        'restaurants_reviews': existing_restaurants,
        'stored_at': datetime.now()
    }

    # 문서를 업데이트하거나 생성
    es.index(index=index_name, id=location_with_underscores, body=doc)

def crawl_restaurant_reviews(es, location, pages):
    """특정 위치에서 여러 페이지에 걸쳐 음식점 리뷰를 크롤링하고 Elasticsearch에 저장합니다."""
    driver = setup_driver()
    search_location(driver, location)
    total_pages = get_total_pages(driver)
    pages = min(pages, total_pages)
    driver.quit()

    all_restaurants = []

    # ThreadPoolExecutor를 사용하여 병렬로 페이지를 크롤링
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(extract_restaurant_info, setup_driver(), location, page) for page in range(1, pages + 1)]
        for future in as_completed(futures):
            try:
                all_restaurants.extend(future.result())
            except Exception as e:
                print(f"Error extracting restaurant info: {e}")

    # Elasticsearch 설정 및 데이터 저장
    # es = setup_elasticsearch()
    index_name = location.split(' ')[0]  # 지역명을 인덱스 이름으로 사용
    location_with_underscores = location.replace(" ", "_") # 엘라스틱서치 id값 서칭할때 띄어쓰기 문제해결

    save_to_elasticsearch(es, index_name, location_with_underscores, all_restaurants)

    return all_restaurants