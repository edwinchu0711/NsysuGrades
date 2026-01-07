import os
import cv2
import base64
import numpy as np
import tensorflow as tf
import requests
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =====================
# 全域設定
# =====================
IMG_WIDTH = 124
IMG_HEIGHT = 24
DIGITS = 4
CHARACTERS = "0123456789"
TFLITE_NAME = "model.tflite"

app = FastAPI()

# =====================
# Pydantic 模型
# =====================
class CrawlRequest(BaseModel):
    account: str
    password: str
    task: str  # "score", "grades", "both", "test"

# =====================
# 輔助函式：模型載入與預測
# =====================
def load_tflite_model():
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, TFLITE_NAME)
    
    if not os.path.exists(model_path):
        # 為了避免 API 啟動失敗，這裡僅 print 警告，實際呼叫時若無模型會報錯
        print(f"⚠️ 警告：找不到 TFLite 模型檔：{model_path}")
        return None, None, None

    with open(model_path, "rb") as f:
        model_bytes = f.read()

    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return interpreter, input_details, output_details

def preprocess_image_from_bytes(img_bytes):
    """
    修改版：直接從記憶體 Bytes 處理圖片，不讀寫硬碟
    """
    # 將 bytes 轉換為 numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("無法解析圖片數據")

    # 轉灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 自適應二值化
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # 形態學操作
    kernel = np.ones((2, 2), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

    # Resize
    bin_img = cv2.resize(bin_img, (IMG_WIDTH, IMG_HEIGHT))

    # 正規化 + 維度擴充
    bin_img = bin_img.astype(np.float32) / 255.0
    bin_img = np.expand_dims(bin_img, axis=-1)  # (H, W, 1)
    bin_img = np.expand_dims(bin_img, axis=0)   # (1, H, W, 1)
    
    return bin_img

def predict_captcha(interpreter, input_details, output_details, img_base64_str):
    """
    接收 Base64 字串，進行預測
    """
    # 解碼 Base64
    img_bytes = base64.b64decode(img_base64_str)
    input_data = preprocess_image_from_bytes(img_bytes)

    # 設定輸入數值
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # 排序輸出
    sorted_outputs = sorted(output_details, key=lambda x: x['name'])

    digits = []
    for od in sorted_outputs:
        probs = interpreter.get_tensor(od["index"])
        idx = int(np.argmax(probs, axis=-1)[0])
        digits.append(CHARACTERS[idx])

    return "".join(digits)

# =====================
# Selenium 驅動與登入
# =====================
def get_driver():
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    # API 模式強烈建議開啟 headless
    options.add_argument("--headless") 
    
    # 若在 Docker 或 Linux 環境，路徑需自行調整；Windows 可註解掉或指定路徑
    # chrome_driver_path = r"path/to/chromedriver"
    # service = Service(chrome_driver_path)
    
    # 這裡假設已安裝 chromedriver 於系統路徑，直接初始化
    driver = webdriver.Chrome(options=options)
    return driver

def login_process(driver, interpreter, input_details, output_details, account, password):
    """
    登入流程，回傳 (是否成功, 訊息)
    """
    url = "https://selcrs.nsysu.edu.tw/scoreqry/sco_query_login.asp"
    driver.get(url)
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # 1. 獲取驗證碼圖片
            captcha_element = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.NAME, "imgVC"))
            )
            captcha_base64 = captcha_element.screenshot_as_base64
            
            # 2. 辨識
            code = predict_captcha(interpreter, input_details, output_details, captcha_base64)
            print(f"嘗試登入 #{attempt+1}, 辨識結果: {code}")

            # 3. 輸入資料
            driver.find_element(By.CSS_SELECTOR, 'input[name="SID"]').clear()
            driver.find_element(By.CSS_SELECTOR, 'input[name="SID"]').send_keys(account)
            
            driver.find_element(By.CSS_SELECTOR, 'input[name="PASSWD"]').clear()
            driver.find_element(By.CSS_SELECTOR, 'input[name="PASSWD"]').send_keys(password)
            
            driver.find_element(By.NAME, "ValidCode").clear()
            driver.find_element(By.NAME, "ValidCode").send_keys(code)

            # 4. 送出 (根據你的 code 調整 selector)
            try:
                submit_btn = driver.find_element(By.CSS_SELECTOR, 'input.login_btn_01')
            except:
                # 備用方案
                submit_btn = driver.find_element(By.CSS_SELECTOR, 'input[type="submit"]')
            submit_btn.click()

            # 5. 處理 Alert (驗證碼錯誤或登入失敗)
            try:
                WebDriverWait(driver, 2).until(EC.alert_is_present())
                alert = driver.switch_to.alert
                alert_text = alert.text
                print(f"Alert: {alert_text}")
                
                alert.accept() # 關閉視窗
                
                if "驗證碼錯誤" in alert_text or "Verified Code Error" in alert_text:
                    driver.get(url) # 重新整理換新驗證碼
                    continue
                else:
                    return False, f"登入失敗: {alert_text}"
            except:
                # 沒有 Alert，檢查是否跳轉
                if "sco_query.asp" in driver.current_url or "Main" in driver.title:
                    return True, "登入成功"
                else:
                    # 有時候沒跳轉也沒alert，可能是成功
                     return True, "登入成功(預判)"

        except Exception as e:
            print(f"登入過程異常: {e}")
            driver.refresh()
            
    return False, "超過最大重試次數，驗證碼辨識失敗"

# =====================
# 爬蟲邏輯 (Requests)
# =====================
def get_requests_session_with_cookies(driver):
    session = requests.Session()
    selenium_cookies = driver.get_cookies()
    for cookie in selenium_cookies:
        session.cookies.set(cookie['name'], cookie['value'])
    return session

def scrape_score(driver):
    """
    對應原 scrape_all_courses (開放成績/當前學期成績)
    回傳: List of Dict
    """
    print("🚀 執行: scrape_score")
    list_url = "https://selcrs.nsysu.edu.tw/scoreqry/sco_query.asp?action=813&KIND=1&LANGS=cht"
    query_url = "https://selcrs.nsysu.edu.tw/scoreqry/sco_query.asp?ACTION=814&KIND=1&LANGS=cht"
    
    session = get_requests_session_with_cookies(driver)
    headers = {
        "User-Agent": "Mozilla/5.0", 
        "Referer": list_url,
        "Origin": "https://selcrs.nsysu.edu.tw"
    }

    # 1. 取得課程清單
    course_info_map = {}
    try:
        res_list = session.get(list_url, headers=headers, timeout=10)
        res_list.encoding = 'utf-8' # 或 big5
        soup_list = BeautifulSoup(res_list.text, 'html.parser')
        rows = soup_list.find_all('tr')[1:]
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 3:
                course_no = cols[1].get_text(strip=True)
                course_name = cols[2].get_text(strip=True)
                if course_no and not re.search(r'[\u4e00-\u9fff]', course_no):
                    course_info_map[course_no] = course_name
    except Exception as e:
        return {"error": f"取得課程清單失敗: {str(e)}"}

    # 2. 詳細成績
    results = []
    for course_no, course_name in course_info_map.items():
        payload = {"CRSNO": course_no, "SCO_TYP_COD": "--"}
        try:
            resp = session.post(query_url, headers=headers, data=payload, timeout=10)
            resp.encoding = 'utf-8'
            soup = BeautifulSoup(resp.text, 'html.parser')
            rows = soup.find_all('tr')
            for row in rows:
                cols = [c.get_text(strip=True) for c in row.find_all('td')]
                if len(cols) >= 6 and cols[0].isdigit():
                    results.append({
                        "course_name": course_name,
                        "year": cols[0],
                        "semester": cols[1],
                        "item": cols[2],
                        "percentage": cols[3],
                        "raw_score": cols[4],
                        "grade": cols[5],
                        "note": cols[6] if len(cols)>6 else ""
                    })
        except:
            pass
            
    return results

def scrape_grades(driver):
    """
    對應原 scrape_historical_data (歷年成績)
    回傳: Dict 包含 "grades"(成績單) 和 "ranks"(排名)
    """
    print("🚀 執行: scrape_grades")
    session = get_requests_session_with_cookies(driver)
    url = "https://selcrs.nsysu.edu.tw/scoreqry/sco_query.asp?ACTION=804&KIND=2&LANGS=cht"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://selcrs.nsysu.edu.tw/scoreqry/sco_query.asp?action=702"
    }

    years = ["113", "112", "111", "110"]
    sems = ["0", "1", "2", "3"]
    
    all_grades = []
    all_ranks = []

    for year in years:
        for sem in sems:
            payload = {"SYEAR": year, "SEM": sem}
            try:
                response = session.post(url, headers=headers, data=payload, timeout=5)
                response.encoding = response.apparent_encoding # 處理編碼
                
                if "無此學期成績" in response.text:
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # A. 成績表
                rows = soup.find_all('tr')
                for row in rows:
                    cols = [td.get_text(strip=True) for td in row.find_all('td')]
                    # 判斷邏輯
                    if len(cols) >= 6 and year in cols[0]:
                        all_grades.append({
                            "year": year,
                            "sem": sem,
                            "year_raw": cols[0],
                            "sem_raw": cols[1],
                            "course_id": cols[2],
                            "course_name": cols[3],
                            "credits": cols[4],
                            "score": cols[5]
                        })

                # B. 排名統計 (解析文字)
                if "修習學分" in response.text:
                    for table in soup.find_all('table'):
                        txt = table.get_text()
                        if "修習學分" in txt:
                            # Regex 提取
                            clean_text = txt.replace("\n", " ").replace("&nbsp;", "")
                            patterns = {
                                "taken_credits": r"修習學分：(\d+)",
                                "earned_credits": r"實得學分：(\d+)",
                                "avg_score": r"平均分數：([\d\.]+)",
                                "class_rank": r"本學期名次：(\d+)",
                                "class_size": r"全班人數：(\d+)"
                            }
                            rank_data = {"year": year, "sem": sem}
                            for key, p in patterns.items():
                                match = re.search(p, clean_text)
                                rank_data[key] = match.group(1) if match else "N/A"
                            all_ranks.append(rank_data)
                            break
            except Exception as e:
                print(f"歷年成績錯誤 ({year}-{sem}): {e}")

    return {"grades": all_grades, "ranks": all_ranks}

# =====================
# API 路由
# =====================
@app.post("/crawl")
async def start_crawl(req: CrawlRequest):
    if req.task == "test":
        return {"status": "success", "message": "API is working"}
    
    if not req.account or not req.password:
        raise HTTPException(status_code=422, detail="帳號與密碼為必填欄位")

    # 載入模型 (建議在 startup event 載入一次全域使用，這裡為求簡便每次載入)
    # 若請求量大，請將 load_tflite_model 移至 app startup
    interpreter, input_details, output_details = load_tflite_model()
    
    if interpreter is None:
        raise HTTPException(status_code=500, detail="伺服器端缺少 TFLite 模型檔案")

    driver = None
    data = {}
    
    try:
        driver = get_driver()
        
        # 執行登入
        success, msg = login_process(driver, interpreter, input_details, output_details, req.account, req.password)
        if not success:
            return {"status": "failed", "message": msg}
        
        # 執行任務
        # 1. 開放成績 / 課程細項成績
        if req.task in ["score", "both"]:
            # 先切換到開放成績頁面以更新 Session 狀態
            driver.get("https://selcrs.nsysu.edu.tw/scoreqry/sco_query.asp?action=700&KIND=1&LANGS=cht")
            data["score_task"] = scrape_score(driver)
        
        # 2. 歷年成績 / 學期總成績
        if req.task in ["grades", "both"]:
            # 先切換到歷年成績頁面以更新 Session 狀態
            driver.get("https://selcrs.nsysu.edu.tw/scoreqry/sco_query.asp?action=700&KIND=2&LANGS=cht")
            data["grades_task"] = scrape_grades(driver)

        return {"status": "success", "results": data}
    
    except Exception as e:
        return {"status": "error", "message": f"執行中斷: {str(e)}"}
    
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    import uvicorn
    # 確保 model.tflite 在同一目錄下
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))