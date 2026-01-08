import os
import cv2
import base64
import numpy as np
import csv
import re
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
# 關鍵修改：優先嘗試載入 tflite_runtime
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        raise ImportError("請安裝 tflite-runtime 或 tensorflow")
# =====================
# 基本設定
# =====================
IMG_WIDTH = 124
IMG_HEIGHT = 24
CHARACTERS = "0123456789"
TFLITE_NAME = "model.tflite"

app = FastAPI()

class CrawlRequest(BaseModel):
    account: str
    password: str
    task: str  # "score", "grades", "both", "test"

# =====================
# 模型與預處理函數 (保持原邏輯)
# =====================
def load_tflite_model():
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, TFLITE_NAME)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到 TFLite 模型檔：{model_path}")

    with open(model_path, "rb") as f:
        model_bytes = f.read()

    # interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("✅ TFLite 模型載入完成")
    return interpreter, input_details, output_details

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
# Selenium 環境設定 (Render 專用)
# =====================
def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    
    # 移除手動指定 Service 路徑,讓 Selenium Manager 自動偵測
    # 僅保留 binary_location 告訴 Selenium 瀏覽器在哪
    if os.path.exists("/usr/bin/chromium"):
        options.binary_location = "/usr/bin/chromium"
    elif os.path.exists("/usr/bin/google-chrome"):
        options.binary_location = "/usr/bin/google-chrome"
        
    return webdriver.Chrome(options=options)
    # """設定 Chrome Driver for GCP VM"""
    # options = Options()
    # options.add_argument("--headless")        # 必備：無介面模式
    # options.add_argument("--disable-gpu")
    # options.add_argument("--no-sandbox")      # 必備：權限修正
    # options.add_argument("--disable-dev-shm-usage") # 必備：防止記憶體溢出
    
    # # 自動下載並啟動對應版本的 Chromedriver
    # service = Service(ChromeDriverManager().install())
    # driver = webdriver.Chrome(service=service, options=options)
    # return driver
# =====================
# 功能函數 (改為 return 數據)
# =====================
def login_process(driver, interpreter, input_details, output_details, acc, pwd):
    url = "https://selcrs.nsysu.edu.tw/scoreqry/sco_query_login.asp"
    driver.get(url)
    
    for attempt in range(5): # 最多嘗試 5 次驗證碼
        captcha_element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.NAME, "imgVC")))
        captcha_bytes = base64.b64decode(captcha_element.screenshot_as_base64)
        code = predict_captcha(interpreter, input_details, output_details, captcha_bytes)
        
        driver.find_element(By.NAME, "SID").clear()
        driver.find_element(By.NAME, "SID").send_keys(acc)
        driver.find_element(By.NAME, "PASSWD").clear()
        driver.find_element(By.NAME, "PASSWD").send_keys(pwd)
        driver.find_element(By.NAME, "ValidCode").send_keys(code)
        
        # 使用 JS 點擊登入按鈕
        login_btn = driver.find_element(By.CSS_SELECTOR, 'input.login_btn_01')
        driver.execute_script("arguments[0].click();", login_btn)
        
        try:
            WebDriverWait(driver, 2).until(EC.alert_is_present())
            alert = driver.switch_to.alert
            if "驗證碼錯誤" in alert.text:
                alert.accept()
                continue
            else:
                alert.accept()
                return False, "帳號或密碼錯誤"
        except:
            return True, "登入成功"
    return False, "驗證碼多次辨識失敗"

def scrape_score(driver):
    score_link = "https://selcrs.nsysu.edu.tw/scoreqry/sco_query.asp?action=700&KIND=1&LANGS=cht"
    driver.get(score_link)
    results = []
    driver.switch_to.frame("mtn_down1")
    radios = driver.find_elements(By.NAME, "CRSNO")
    for i in range(len(radios)):
        driver.switch_to.default_content()
        driver.switch_to.frame("mtn_down1")
        r = driver.find_elements(By.NAME, "CRSNO")[i]
        name = r.find_element(By.XPATH, "./parent::font/parent::td/following-sibling::td[2]").text.strip()
        
        # 使用 JS 點擊 radio button
        driver.execute_script("arguments[0].click();", r)
        
        # 使用 JS 點擊查詢按鈕
        submit_btn = driver.find_element(By.NAME, "B1")
        driver.execute_script("arguments[0].click();", submit_btn)
        
        driver.switch_to.default_content()
        driver.switch_to.frame("mtn_down2")
        try:
            rows = driver.find_elements(By.CSS_SELECTOR, "table tr")[1:]
            for row in rows:
                cols = [td.text.strip() for td in row.find_elements(By.TAG_NAME, "td")]
                results.append({"課程名稱": name, "詳情": cols})
        except: pass
    return results

def scrape_grades(driver):
    grades_link = "https://selcrs.nsysu.edu.tw/scoreqry/sco_query.asp?action=700&KIND=2&LANGS=cht"
    driver.get(grades_link)
    grade_list = []
    rank_list = []
    
    driver.switch_to.frame("mtn_down1")
    years = [opt.text.strip() for opt in Select(driver.find_element(By.NAME, "SYEAR")).options[:3]]
    
    for y_idx, year in enumerate(years):
        for s_idx in range(2):
            driver.switch_to.default_content()
            driver.switch_to.frame("mtn_down1")
            Select(driver.find_element(By.NAME, "SYEAR")).select_by_index(y_idx)
            sem_sel = Select(driver.find_element(By.NAME, "SEM"))
            if s_idx >= len(sem_sel.options): break
            sem = sem_sel.options[s_idx].text.strip()
            sem_sel.select_by_index(s_idx)
            
            # 使用 JS 點擊查詢按鈕
            submit_btn = driver.find_element(By.NAME, "B1")
            driver.execute_script("arguments[0].click();", submit_btn)
            
            driver.switch_to.default_content()
            try:
                WebDriverWait(driver, 3).until(EC.frame_to_be_available_and_switch_to_it((By.NAME, "mtn_down2")))
                tables = driver.find_elements(By.TAG_NAME, "table")
                for table in tables:
                    text = table.text
                    if "課程編號" in text:
                        for row in table.find_elements(By.TAG_NAME, "tr")[1:]:
                            grade_list.append([year, sem] + [td.text.strip() for td in row.find_elements(By.TAG_NAME, "td")])
                    elif "修習學分" in text:
                        stats = [td.text.strip() for td in table.find_elements(By.TAG_NAME, "td") if td.text.strip()]
                        rank_list.append({"學年度": year, "學期": sem, "數值": stats})
            except: pass
    return {"成績明細": grade_list, "排名統計": rank_list}

# =====================
# API 路由
# =====================
@app.post("/api/scrape")
async def start_crawl(req: CrawlRequest):
    if req.task == "test":
        return {"status": "success", "message": "ok"}
    # 1. 檢查帳密是否為空
    if not req.account or not req.password:
        raise HTTPException(status_code=422, detail="帳號與密碼為必填欄位")

    interpreter, input_details, output_details = load_tflite_model()
    driver = get_driver()
    data = {}
    
    try:
        success, msg = login_process(driver, interpreter, input_details, output_details, req.account, req.password)
        if not success:
            return {"status": "failed", "message": msg}
        
        # 2. 執行任務
        if req.task in ["score", "both"]:
            data["score_task"] = scrape_score(driver)
        
        if req.task in ["grades", "both"]:
            # 確保這裡呼叫的是 scrape_grades
            data["grades_task"] = scrape_grades(driver)

        return {"status": "success", "results": data}
    
    except Exception as e:
        # 回傳具體的錯誤訊息方便除錯
        return {"status": "error", "message": f"執行中斷: {str(e)}"}
    finally:
        if 'driver' in locals():
            driver.quit()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
