import os
import cv2
import base64
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait, Select

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
# TFLite OCR
# =====================
def load_tflite_model():
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, TFLITE_NAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite 模型不存在：{model_path}")

    with open(model_path, "rb") as f:
        model_bytes = f.read()
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLite 模型載入完成")
    return interpreter, input_details, output_details

def predict_captcha(interpreter, input_details, output_details, image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    bin_img = cv2.resize(bin_img, (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
    bin_img = np.expand_dims(np.expand_dims(bin_img, axis=-1), axis=0)
    interpreter.set_tensor(input_details[0]["index"], bin_img)
    interpreter.invoke()
    sorted_outputs = sorted(output_details, key=lambda x: x['name'])
    return "".join([CHARACTERS[int(np.argmax(interpreter.get_tensor(od["index"]), axis=-1)[0])] for od in sorted_outputs])

# =====================
# Selenium
# =====================
def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-background-networking")
    return webdriver.Chrome(options=options)

# =====================
# 登入
# =====================
def login_process(driver, interpreter, input_details, output_details, acc, pwd):
    url = "https://selcrs.nsysu.edu.tw/scoreqry/sco_query_login.asp"
    driver.get(url)
    for attempt in range(5):
        captcha_el = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.NAME, "imgVC"))
        )
        captcha_bytes = base64.b64decode(captcha_el.screenshot_as_base64)
        code = predict_captcha(interpreter, input_details, output_details, captcha_bytes)

        driver.find_element(By.NAME, "SID").clear()
        driver.find_element(By.NAME, "SID").send_keys(acc)
        driver.find_element(By.NAME, "PASSWD").clear()
        driver.find_element(By.NAME, "PASSWD").send_keys(pwd)
        driver.find_element(By.NAME, "ValidCode").send_keys(code)
        driver.find_element(By.CSS_SELECTOR, 'input.login_btn_01').click()

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
    return False, "驗證碼辨識失敗"

# =====================
# 抓開放成績 (score)
# =====================
def scrape_score(driver):
    score_link = "https://selcrs.nsysu.edu.tw/scoreqry/sco_query.asp?action=700&KIND=1&LANGS=cht"
    driver.get(score_link)
    driver.switch_to.frame("mtn_down1")
    results = []

    # 先抓所有 CRSNO 與名稱
    radios = driver.find_elements(By.NAME, "CRSNO")
    courses = []
    for r in radios:
        name = r.find_element(By.XPATH, "./parent::font/parent::td/following-sibling::td[2]").text.strip()
        value = r.get_attribute("value")
        courses.append({"name": name, "value": value})

    # 批量抓
    for course in courses:
        driver.execute_script(f"""
            document.getElementsByName('CRSNO')[0].value = '{course['value']}';
            document.getElementsByName('B1')[0].click();
        """)
        driver.switch_to.default_content()
        driver.switch_to.frame("mtn_down2")
        try:
            rows = driver.find_elements(By.CSS_SELECTOR, "table tr")[1:]
            for row in rows:
                cols = [td.text.strip() for td in row.find_elements(By.TAG_NAME, "td")]
                results.append({"課程名稱": course['name'], "詳情": cols})
        except:
            continue
        driver.switch_to.default_content()
        driver.switch_to.frame("mtn_down1")

    return results

# =====================
# 抓學期成績 (grades)
# =====================
def scrape_grades(driver):
    grades_link = "https://selcrs.nsysu.edu.tw/scoreqry/sco_query.asp?action=700&KIND=2&LANGS=cht"
    driver.get(grades_link)
    driver.switch_to.frame("mtn_down1")

    grade_list = []
    rank_list = []

    years = [opt.text.strip() for opt in Select(driver.find_element(By.NAME, "SYEAR")).options[:3]]

    for y_idx, year in enumerate(years):
        for s_idx in range(2):
            driver.execute_script(f"document.getElementsByName('SYEAR')[0].selectedIndex = {y_idx};")
            driver.execute_script(f"document.getElementsByName('SEM')[0].selectedIndex = {s_idx};")
            driver.find_element(By.NAME, "B1").click()

            driver.switch_to.default_content()
            try:
                WebDriverWait(driver, 5).until(EC.frame_to_be_available_and_switch_to_it((By.NAME, "mtn_down2")))
                tables = driver.find_elements(By.TAG_NAME, "table")
                for table in tables:
                    text = table.text
                    if "課程編號" in text:
                        for row in table.find_elements(By.TAG_NAME, "tr")[1:]:
                            grade_list.append([year, s_idx+1] + [td.text.strip() for td in row.find_elements(By.TAG_NAME, "td")])
                    elif "修習學分" in text:
                        stats = [td.text.strip() for td in table.find_elements(By.TAG_NAME, "td") if td.text.strip()]
                        rank_list.append({"學年度": year, "學期": s_idx+1, "數值": stats})
            except:
                continue
            driver.switch_to.default_content()
            driver.switch_to.frame("mtn_down1")

    return {"成績明細": grade_list, "排名統計": rank_list}

# =====================
# API 路由
# =====================
@app.post("/crawl")
async def start_crawl(req: CrawlRequest):
    if req.task == "test":
        return {"status": "success", "message": "ok"}

    if not req.account or not req.password:
        raise HTTPException(status_code=422, detail="帳號與密碼為必填欄位")

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
            data["grades_task"] = scrape_grades(driver)

        return {"status": "success", "results": data}
    except Exception as e:
        return {"status": "error", "message": f"執行中斷: {str(e)}"}
    finally:
        driver.quit()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
