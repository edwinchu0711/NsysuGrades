import os
import cv2
import base64
import numpy as np
import tensorflow as tf
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import requests
from bs4 import BeautifulSoup
import re
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# =====================
# 基本設定 (需與訓練端完全一致)
# =====================
IMG_WIDTH = 124
IMG_HEIGHT = 24
DIGITS = 4
CHARACTERS = "0123456789"
TFLITE_NAME = "model.tflite"

# =====================
# 載入 TFLite 模型
# =====================
def load_tflite_model():
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, TFLITE_NAME)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到 TFLite 模型檔：{model_path}")

    with open(model_path, "rb") as f:
        model_bytes = f.read()

    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("✅ TFLite 模型載入完成")
    return interpreter, input_details, output_details

# =====================
# 圖片預處理 (完全同步訓練端邏輯)
# =====================
def preprocess_image_for_model(image_path):
    # 1. 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"無法讀取圖片：{image_path}")

    # 2. 轉灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 自適應二值化：讓字更凸顯，字為白，背景為黑
    bin_img = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # 4. 形態學操作：去噪點
    kernel = np.ones((2, 2), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

    # 5. Resize 到統一大小
    bin_img = cv2.resize(bin_img, (IMG_WIDTH, IMG_HEIGHT))

    # 6. 正規化 + 維度擴充
    bin_img = bin_img.astype(np.float32) / 255.0
    bin_img = np.expand_dims(bin_img, axis=-1)  # (H, W, 1)
    bin_img = np.expand_dims(bin_img, axis=0)   # (1, H, W, 1)
    
    return bin_img

# =====================
# 預測函式
# =====================
def predict_captcha(interpreter, input_details, output_details, image_path):
    input_data = preprocess_image_for_model(image_path)

    # 設定輸入數值
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # 關鍵：根據輸出的名稱排序 (確保 digit0 在第一位)
    sorted_outputs = sorted(output_details, key=lambda x: x['name'])

    digits = []
    for od in sorted_outputs:
        probs = interpreter.get_tensor(od["index"])
        idx = int(np.argmax(probs, axis=-1)[0])
        digits.append(CHARACTERS[idx])

    # --- 新增：辨識完成後刪除暫存圖片 ---
    if os.path.exists(image_path):
        os.remove(image_path)
    return "".join(digits)


def login(driver, interpreter, input_details, output_details, account, password):
    try:
        # 2. 獲取驗證碼圖片
        captcha_element = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.NAME, "imgVC"))
        )
        captcha_base64 = captcha_element.screenshot_as_base64
        img_path = "temp_captcha.png"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(captcha_base64))
        
        code = predict_captcha(interpreter, input_details, output_details, img_path)
        print(f"辨識結果: {code}")

        # 3. 輸入帳號
        acc_input = driver.find_element(By.CSS_SELECTOR, 'input[name="SID"]')
        acc_input.clear()
        acc_input.send_keys(account)

        # 4. 輸入密碼
        pwd_input = driver.find_element(By.CSS_SELECTOR, 'input[name="PASSWD"]')
        pwd_input.clear()
        pwd_input.send_keys(password)

        # 5. 填入驗證碼
        valid_input = driver.find_element(By.NAME, "ValidCode")
        valid_input.clear()
        valid_input.send_keys(code)

        # 6. 按下「確定送出」按鈕
        submit_btn = driver.find_element(By.CSS_SELECTOR, 'input.login_btn_01[wfd-id="id6"]')
        submit_btn.click()
        
    except Exception as e:
        print(f"發生錯誤: {e}")


def get_requests_session_with_cookies(driver):
    """
    將 Selenium 的 Cookie 注入到一個真正的 requests.Session 物件中
    """
    session = requests.Session()
    selenium_cookies = driver.get_cookies()
    for cookie in selenium_cookies:
        session.cookies.set(cookie['name'], cookie['value'])
    return session

def scrape_all_courses(driver):
    print("\n🚀 開始執行『開放成績』高速爬取模式 (Requests)...")
    
    # 1. 準備基礎資料
    list_url = "https://selcrs.nsysu.edu.tw/scoreqry/sco_query.asp?action=813&KIND=1&LANGS=cht"
    query_url = "https://selcrs.nsysu.edu.tw/scoreqry/sco_query.asp?ACTION=814&KIND=1&LANGS=cht"
    
    session = get_requests_session_with_cookies(driver)
    
    headers = {
        "User-Agent": driver.execute_script("return navigator.userAgent;"),
        "Referer": list_url,
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://selcrs.nsysu.edu.tw"
    }

    # 2. 取得課程清單
    course_info_map = {}
    try:
        res_list = session.get(list_url, headers=headers, timeout=10)
        res_list.encoding = 'utf-8' 
        soup_list = BeautifulSoup(res_list.text, 'html.parser')
        
        rows = soup_list.find_all('tr')[1:] 
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 3:
                course_no = cols[1].get_text(strip=True)
                course_name = cols[2].get_text(strip=True)
                
                if course_no and not re.search(r'[\u4e00-\u9fff]', course_no):
                    course_info_map[course_no] = course_name

        print(f"✅ 成功取得課程清單，共 {len(course_info_map)} 門課程。")

    except Exception as e:
        print(f"❌ 取得清單失敗: {e}")
        return []

    # 3. 逐一抓取詳細成績
    score_data = [] 

    for course_no, course_name in course_info_map.items():
        print(f"正在抓取: [{course_no}] {course_name}      ", end="\r")
        
        payload = {"CRSNO": course_no, "SCO_TYP_COD": "--"}
        
        try:
            response = session.post(query_url, headers=headers, data=payload, timeout=10)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            rows = soup.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 6:
                    text_cols = [c.get_text(strip=True) for c in cols]
                    if text_cols[0].isdigit():
                        score_data.append({
                            "課程名稱": course_name,
                            "學年度": text_cols[0],
                            "學期": text_cols[1],
                            "成績項目": text_cols[2],
                            "百分比": text_cols[3],
                            "原始分數": text_cols[4],
                            "等第成績": text_cols[5],
                            "備註": text_cols[6] if len(text_cols) > 6 else ""
                        })

        except Exception as e:
            print(f"\n❌ 抓取 {course_no} 時發生錯誤: {e}")

    print("\n✅ 開放成績爬取任務完成。")
    return score_data


def get_selenium_cookies(driver):
    """將 Selenium 的 Cookie 轉換為 requests 可用的格式"""
    selenium_cookies = driver.get_cookies()
    cookies = {cookie['name']: cookie['value'] for cookie in selenium_cookies}
    return cookies

def scrape_historical_data(driver):
    """
    重寫：使用 requests 進行歷史成績爬取
    """
    # 1. 從 Selenium 獲取登入後的 Session
    session_cookies = get_selenium_cookies(driver)
    
    # 2. 準備 Requests 環境
    url = "https://selcrs.nsysu.edu.tw/scoreqry/sco_query.asp?ACTION=804&KIND=2&LANGS=cht"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
        "Referer": "https://selcrs.nsysu.edu.tw/scoreqry/sco_query.asp?action=702",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    grades_data = []
    rank_data = []
    
    years = ["113", "112", "111", "110"] 
    sems = ["0", "1", "2", "3"]

    print("🚀 開始使用 Requests 批次抓取歷史成績...")

    for year in years:
        for sem in sems:
            print(f"正在抓取：{year}學年度 第{sem}學期...")
            payload = f"SYEAR={year}&SEM={sem}"
            
            try:
                response = requests.post(url, headers=headers, cookies=session_cookies, data=payload, timeout=10)
                response.encoding = response.apparent_encoding 
                
                if "無此學期成績" in response.text:
                    print(f"ℹ️ {year}-{sem} 無成績資料")
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # --- A. 處理成績表 ---
                rows = soup.find_all('tr')
                for row in rows:
                    cols = [td.get_text(strip=True) for td in row.find_all('td')]
                    
                    if len(cols) >= 6 and year in cols[0]:
                        grades_data.append({
                            "學年度": year,
                            "學期": sem,
                            "學年度_原": cols[0],
                            "學期_原": cols[1],
                            "課程編號": cols[2],
                            "課程名稱": cols[3],
                            "學分數": cols[4],
                            "成績": cols[5]
                        })

                # --- B. 處理排名統計表 ---
                if "修習學分" in response.text:
                    for table in soup.find_all('table'):
                        if "修習學分" in table.get_text():
                            rank_info = extract_rank_info(year, sem, table.get_text())
                            if rank_info:
                                rank_data.append(rank_info)
                            break

            except Exception as e:
                print(f"❌ 抓取 {year}-{sem} 時發生錯誤: {e}")

    return grades_data, rank_data


def extract_rank_info(year, sem, raw_table_text):
    """提取排名資訊"""
    patterns = {
        "修習學分": r"修習學分：(\d+)",
        "實得學分": r"實得學分：(\d+)",
        "平均分數": r"平均分數：([\d\.]+)",
        "本學期名次": r"本學期名次：(\d+)",
        "全班人數": r"全班人數：(\d+)"
    }

    clean_text = raw_table_text.replace("\n", " ").replace("&nbsp;", "")
    
    rank_info = {"學年度": year, "學期": sem}
    
    for key, p in patterns.items():
        match = re.search(p, clean_text)
        rank_info[key] = match.group(1) if match else "無"
    
    return rank_info


def setup_chrome_driver():
    """設定 Chrome Driver for Render"""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    
    # Render 環境會自動提供 chromedriver
    driver = webdriver.Chrome(options=options)
    return driver


def perform_scraping(account, password):
    """執行爬蟲主流程"""
    try:
        interpreter, input_details, output_details = load_tflite_model()
        driver = setup_chrome_driver()

        url = "https://selcrs.nsysu.edu.tw/scoreqry/sco_query_login.asp"
        driver.get(url)
        
        # 開始登入
        login_successful = False
        max_attempts = 5
        attempts = 0
        
        while not login_successful and attempts < max_attempts:
            attempts += 1
            login(driver, interpreter, input_details, output_details, account, password)
            try:
                WebDriverWait(driver, 1.5).until(EC.alert_is_present())
                alert = driver.switch_to.alert
                alert_text = alert.text
                print(f"警示訊息: {alert_text}")
                if "驗證碼錯誤" in alert_text or "Verified Code Error" in alert_text:
                    print("❌ 辨識錯誤，正在關閉視窗並重新嘗試...")
                    alert.accept()
                    driver.get(url)
                    continue
                else:
                    print(f"登入失敗，原因: {alert_text}")
                    alert.accept()
                    driver.quit()
                    return {"error": alert_text}
            except:
                print("✅ 未偵測到錯誤彈窗，檢查是否成功進入系統...")
                login_successful = True

        if not login_successful:
            driver.quit()
            return {"error": "登入失敗，超過最大嘗試次數"}

        # 開放成績查詢
        score_link = "https://selcrs.nsysu.edu.tw/scoreqry/sco_query.asp?action=700&KIND=1&LANGS=cht"
        driver.get(score_link)
        score_data = scrape_all_courses(driver)

        # 學期成績查詢
        grades_link = "https://selcrs.nsysu.edu.tw/scoreqry/sco_query.asp?action=700&KIND=2&LANGS=cht"
        driver.get(grades_link)
        grades_data, rank_data = scrape_historical_data(driver)

        driver.quit()

        return {
            "success": True,
            "開放成績": score_data,
            "學期成績": grades_data,
            "排名資訊": rank_data
        }

    except Exception as e:
        return {"error": str(e)}


# =====================
# Flask API 路由
# =====================
@app.route('/api/scrape', methods=['POST'])
def scrape():
    """
    API 端點：接收帳號密碼，返回爬取結果
    
    Request Body:
    {
        "account": "學號",
        "password": "密碼"
    }
    """
    data = request.get_json()
    
    if not data or 'account' not in data or 'password' not in data:
        return jsonify({"error": "請提供 account 和 password"}), 400
    
    account = data['account']
    password = data['password']
    
    result = perform_scraping(account, password)
    
    if "error" in result:
        return jsonify(result), 500
    
    return jsonify(result), 200


@app.route('/health', methods=['GET'])
def health():
    """健康檢查端點"""
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
