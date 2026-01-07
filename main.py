import os
import cv2
import base64
import numpy as np
import tensorflow as tf
import csv
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# =====================
# 基本設定
# =====================
IMG_WIDTH = 124
IMG_HEIGHT = 24
CHARACTERS = "0123456789"
TFLITE_NAME = "model.tflite"

app = FastAPI()

# API 請求格式
class CrawlRequest(BaseModel):
    account: str
    password: str
    task: str  # "score" (分數), "grades" (成績), "both" (都要)

# =====================
# 模型載入與驗證碼辨識
# =====================
def load_tflite_model():
    model_path = os.path.join(os.path.dirname(__file__), TFLITE_NAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到 TFLite 模型檔：{model_path}")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

def predict_captcha(interpreter, input_details, output_details, image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    bin_img = cv2.resize(bin_img, (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
    bin_img = np.expand_dims(np.expand_dims(bin_img, axis=-1), axis=0)
    
    interpreter.set_tensor(input_details[0]["index"], bin_img)
    interpreter.invoke()
    sorted_outputs = sorted(output_details, key=lambda x: x['name'])
    return "".join([CHARACTERS[int(np.argmax(interpreter.get_tensor(od["index"]), axis=-1)[0])] for od in sorted_outputs])

# =====================
# Selenium 環境設定 (配合 Docker 內路徑)
# =====================
def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    
    # 指向 Dockerfile 中安裝的 Chromium 路徑
    options.binary_location = "/usr/bin/chromium"
    service = Service("/usr/bin/chromium-driver")
    
    return webdriver.Chrome(service=service, options=options)

# =====================
# 登入與爬蟲邏輯
# =====================
def login_process(driver, interpreter, input_details, output_details, acc, pwd):
    url = "https://selcrs.nsysu.edu.tw/scoreqry/sco_query_login.asp"
    driver.get(url)
    
    for attempt in range(5):
        try:
            captcha_element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.NAME, "imgVC")))
            captcha_bytes = base64.b64decode(captcha_element.screenshot_as_base64)
            code = predict_captcha(interpreter, input_details, output_details, captcha_bytes)
            
            driver.find_element(By.NAME, "SID").clear()
            driver.find_element(By.NAME, "SID").send_keys(acc)
            driver.find_element(By.NAME, "PASSWD").clear()
            driver.find_element(By.NAME, "PASSWD").send_keys(pwd)
            driver.find_element(By.NAME, "ValidCode").clear()
            driver.find_element(By.NAME, "ValidCode").send_keys(code)
            driver.find_element(By.CSS_SELECTOR, 'input.login_btn_01').click()
            
            WebDriverWait(driver, 2).until(EC.alert_is_present())
            alert = driver.switch_to.alert
            msg = alert.text
            alert.accept()
            if "驗證碼錯誤" in msg:
                continue
            return False, msg
        except:
            # 沒有 Alert 代表可能成功
            return True, "成功"
    return False, "重試次數過多"

def scrape_score(driver):
    """開放成績查詢 (單科詳細)"""
    score_link = "https://selcrs.nsysu.edu.tw/scoreqry/sco_query.asp?action=813&KIND=1&LANGS=cht"
    driver.get(score_link)
    results = []
    
    try:
        WebDriverWait(driver, 5).until(EC.frame_to_be_available_and_switch_to_it((By.NAME, "mtn_down1")))
        radios = driver.find_elements(By.NAME, "CRSNO")
        total = len(radios)
        
        for i in range(total):
            driver.switch_to.default_content()
            driver.switch_to.frame("mtn_down1")
            r = driver.find_elements(By.NAME, "CRSNO")[i]
            # 抓取名稱
            name_td = r.find_element(By.XPATH, "./parent::font/parent::td/following-sibling::td[2]")
            course_name = name_td.text.strip()
            
            r.click()
            driver.find_element(By.NAME, "B1").click()
            
            driver.switch_to.default_content()
            driver.switch_to.frame("mtn_down2")
            rows = driver.find_elements(By.CSS_SELECTOR, "table tr")[1:]
            details = []
            for row in rows:
                cols = [td.text.strip() for td in row.find_elements(By.TAG_NAME, "td")]
                if cols: details.append(cols)
            results.append({"course": course_name, "details": details})
    except Exception as e:
        print(f"Scrape Score Error: {e}")
    return results

def scrape_grades(driver):
    """學期成績查詢 (歷年排名)"""
    grades_link = "https://selcrs.nsysu.edu.tw/scoreqry/sco_query.asp?action=700&KIND=2&LANGS=cht"
    driver.get(grades_link)
    grade_list = []
    rank_list = []
    
    patterns = {
        "修習學分": r"修習學分：(\d+)",
        "實得學分": r"實得學分：(\d+)",
        "平均分數": r"平均分數：([\d\.]+)",
        "名次": r"本學期名次：(\d+)",
        "全班人數": r"全班人數：(\d+)"
    }

    try:
        WebDriverWait(driver, 5).until(EC.frame_to_be_available_and_switch_to_it((By.NAME, "mtn_down1")))
        years = [opt.text.strip() for opt in Select(driver.find_element(By.NAME, "SYEAR")).options[:3]]
        
        for y_idx, year in enumerate(years):
            for s_idx in range(2):
                driver.switch_to.default_content()
                driver.switch_to.frame("mtn_down1")
                Select(driver.find_element(By.NAME, "SYEAR")).select_by_index(y_idx)
                sem_sel = Select(driver.find_element(By.NAME, "SEM"))
                if s_idx >= len(sem_sel.options): break
                sem_name = sem_sel.options[s_idx].text.strip()
                sem_sel.select_by_index(s_idx)
                driver.find_element(By.NAME, "B1").click()
                
                driver.switch_to.default_content()
                try:
                    WebDriverWait(driver, 3).until(EC.frame_to_be_available_and_switch_to_it((By.NAME, "mtn_down2")))
                    tables = driver.find_elements(By.TAG_NAME, "table")
                    for table in tables:
                        raw_text = table.text
                        if "課程編號" in raw_text:
                            for row in table.find_elements(By.TAG_NAME, "tr")[1:]:
                                cols = [td.text.strip() for td in row.find_elements(By.TAG_NAME, "td")]
                                if cols: grade_list.append({"year": year, "sem": sem_name, "data": cols})
                        elif "修習學分" in raw_text:
                            clean_t = raw_text.replace("\n", " ")
                            stat = {"year": year, "sem": sem_name}
                            for k, p in patterns.items():
                                m = re.search(p, clean_t)
                                stat[k] = m.group(1) if m else "無"
                            rank_list.append(stat)
                except: continue
    except Exception as e:
        print(f"Scrape Grades Error: {e}")
    return {"grades": grade_list, "ranks": rank_list}

# =====================
# API 入口
# =====================
@app.post("/crawl")
async def start_crawl(req: CrawlRequest):
    interpreter, input_details, output_details = load_tflite_model()
    driver = get_driver()
    data = {}
    
    try:
        success, msg = login_process(driver, interpreter, input_details, output_details, req.account, req.password)
        if not success:
            return {"status": "failed", "message": msg}
        
        if req.task in ["score", "both"]:
            data["score_task"] = scrape_score(driver)
        
        if req.task in ["grades", "both"]:
            # 注意：這裡修正為對應上面定義的 scrape_grades 函數
            data["grades_task"] = scrape_grades(driver)

        return {"status": "success", "results": data}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        driver.quit()

if __name__ == "__main__":
    import uvicorn
    # Render 啟動通常會自動帶入 PORT 環境變數
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
