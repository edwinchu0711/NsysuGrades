import os
import cv2
import base64
import numpy as np
import tensorflow as tf
import csv
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# =====================
# 1. 全域初始化 (啟動時執行一次)
# =====================
IMG_WIDTH = 124
IMG_HEIGHT = 24
CHARACTERS = "0123456789"
TFLITE_NAME = "model.tflite"

app = FastAPI()

def load_tflite_model():
    model_path = os.path.join(os.path.dirname(__file__), TFLITE_NAME)
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

# 在啟動時載入模型，減少 API 請求耗時
interpreter, input_details, output_details = load_tflite_model()

class CrawlRequest(BaseModel):
    account: str = Field("", description="帳號")
    password: str = Field("", description="密碼")
    task: str = Field(..., pattern="^(score|grades|both|test)$")

# =====================
# 2. 核心功能函數 (優化版)
# =====================

def predict_captcha(image_bytes):
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

def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    # 加速設定：禁用圖片與不必要的資源載入
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.notifications": 2
    }
    options.add_experimental_option("prefs", prefs)
    
    if os.path.exists("/usr/bin/chromium"):
        options.binary_location = "/usr/bin/chromium"
    
    # 讓 Selenium 自動管理 Driver 路徑，提升相容性
    return webdriver.Chrome(options=options)

def login_process(driver, acc, pwd):
    url = "https://selcrs.nsysu.edu.tw/scoreqry/sco_query_login.asp"
    driver.get(url)
    
    for _ in range(5):
        try:
            captcha_el = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.NAME, "imgVC")))
            captcha_bytes = base64.b64decode(captcha_el.screenshot_as_base64)
            code = predict_captcha(captcha_bytes)
            
            driver.find_element(By.NAME, "SID").send_keys(acc)
            driver.find_element(By.NAME, "PASSWD").send_keys(pwd)
            driver.find_element(By.NAME, "ValidCode").send_keys(code)
            driver.find_element(By.CSS_SELECTOR, 'input.login_btn_01').click()
            
            WebDriverWait(driver, 2).until(EC.alert_is_present())
            alert = driver.switch_to.alert
            msg = alert.text
            alert.accept()
            if "驗證碼錯誤" in msg:
                driver.get(url) # 重新整理頁面
                continue
            return False, msg
        except:
            return True, "Success"
    return False, "Retry limit reached"

# =====================
# 3. 爬蟲任務
# =====================

def scrape_score(driver):
    driver.get("https://selcrs.nsysu.edu.tw/scoreqry/sco_query.asp?action=813&KIND=1&LANGS=cht")
    results = []
    try:
        WebDriverWait(driver, 5).until(EC.frame_to_be_available_and_switch_to_it((By.NAME, "mtn_down1")))
        total = len(driver.find_elements(By.NAME, "CRSNO"))
        for i in range(total):
            driver.switch_to.default_content()
            driver.switch_to.frame("mtn_down1")
            r = driver.find_elements(By.NAME, "CRSNO")[i]
            course_name = r.find_element(By.XPATH, "./parent::font/parent::td/following-sibling::td[2]").text.strip()
            r.click()
            driver.find_element(By.NAME, "B1").click()
            
            driver.switch_to.default_content()
            driver.switch_to.frame("mtn_down2")
            rows = driver.find_elements(By.CSS_SELECTOR, "table tr")[1:]
            details = [[td.text.strip() for td in row.find_elements(By.TAG_NAME, "td")] for row in rows]
            results.append({"course": course_name, "details": [d for d in details if d]})
    except: pass
    return results

def scrape_grades(driver):
    driver.get("https://selcrs.nsysu.edu.tw/scoreqry/sco_query.asp?action=700&KIND=2&LANGS=cht")
    grade_list, rank_list = [], []
    patterns = {"修習學分": r"修習學分：(\d+)", "實得學分": r"實得學分：(\d+)", "平均分數": r"平均分數：([\d\.]+)", "名次": r"名次：(\d+)", "全班人數": r"全班人數：(\d+)"}

    try:
        WebDriverWait(driver, 5).until(EC.frame_to_be_available_and_switch_to_it((By.NAME, "mtn_down1")))
        # 只抓取最近 2 年以加速
        years = [opt.text.strip() for opt in Select(driver.find_element(By.NAME, "SYEAR")).options[:2]]
        
        for y_idx, year in enumerate(years):
            for s_idx in range(2):
                driver.switch_to.default_content()
                driver.switch_to.frame("mtn_down1")
                # 使用 JavaScript 快速選取選單項目
                driver.execute_script(f"document.getElementsByName('SYEAR')[0].selectedIndex = {y_idx};")
                sem_sel = Select(driver.find_element(By.NAME, "SEM"))
                if s_idx >= len(sem_sel.options): break
                sem_name = sem_sel.options[s_idx].text.strip()
                driver.execute_script(f"document.getElementsByName('SEM')[0].selectedIndex = {s_idx};")
                driver.find_element(By.NAME, "B1").click()
                
                driver.switch_to.default_content()
                try:
                    WebDriverWait(driver, 3).until(EC.frame_to_be_available_and_switch_to_it((By.NAME, "mtn_down2")))
                    for table in driver.find_elements(By.TAG_NAME, "table"):
                        t_text = table.text
                        if "課程編號" in t_text:
                            for row in table.find_elements(By.TAG_NAME, "tr")[1:]:
                                cols = [td.text.strip() for td in row.find_elements(By.TAG_NAME, "td")]
                                if cols: grade_list.append({"year": year, "sem": sem_name, "data": cols})
                        elif "修習學分" in t_text:
                            stat = {"year": year, "sem": sem_name}
                            for k, p in patterns.items():
                                m = re.search(p, t_text.replace("\n", " "))
                                stat[k] = m.group(1) if m else "無"
                            rank_list.append(stat)
                except: continue
    except: pass
    return {"grades": grade_list, "ranks": rank_list}

# =====================
# 4. API 路由
# =====================

@app.get("/")
async def root():
    return {"status": "running", "endpoint": "/crawl"}

@app.post("/crawl")
async def start_crawl(req: CrawlRequest):
    # --- 新功能：test 任務 ---
    if req.task == "test":
        return {"status": "success", "results": "OK"}

    # 執行爬蟲前檢查帳密
    if not req.account or not req.password:
        raise HTTPException(status_code=422, detail="Missing account or password for this task")

    driver = get_driver()
    data = {}
    try:
        success, msg = login_process(driver, req.account, req.password)
        if not success:
            return {"status": "failed", "message": msg}
        
        if req.task in ["score", "both"]:
            data["score_task"] = scrape_score(driver)
        if req.task in ["grades", "both"]:
            data["grades_task"] = scrape_grades(driver)

        return {"status": "success", "results": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        driver.quit()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))