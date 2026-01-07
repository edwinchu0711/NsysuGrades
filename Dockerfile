FROM python:3.9-slim

# 安裝系統依賴，修正 Debian Trixie 找不到舊套件的問題
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    curl \
    # Chromium 瀏覽器與驅動
    chromium \
    chromium-driver \
    # 必要的系統函式庫 (移除過時的 libgconf-2-4 與 libgl1-mesa-glx)
    libglib2.0-0 \
    libnss3 \
    libfontconfig1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 設定環境變數，讓 Selenium 知道 Chromium 的位置
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 啟動命令
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
