# 使用 3.9-slim 是正確的，這對 tflite-runtime 支援最穩定
FROM python:3.9-slim

# 更新 apt 來源並安裝必要工具
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    curl \
    # 安裝 Chromium 與 Driver
    chromium \
    chromium-driver \
    # OpenCV 必備的底層函式庫 (headless 版本也需要這些)
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 設定環境變數
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver
# 避免 Python 產生 .pyc 檔案，加速啟動並節省空間
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 先複製 requirements.txt 利用 Docker 快取層
COPY requirements.txt .

# 升級 pip 並安裝依賴
# 注意：如果 pip install tflite-runtime 失敗，可以考慮直接指向官方的 .whl 網址
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# 暴露 Render 會使用的 PORT (雖然 Render 會自動對應，但宣告是好習慣)
EXPOSE 10000

# 啟動命令
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
