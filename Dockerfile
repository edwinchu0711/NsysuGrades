# 使用帶有 Python 的輕量級 Linux 映像檔
FROM python:3.9-slim

# 安裝必要的 Linux 套件與 Chrome 瀏覽器
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    curl \
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    libgl1-mesa-glx \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# 設定環境變數，讓 Selenium 知道去哪裡找 Chrome
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromium-driver

# 設定工作目錄
WORKDIR /app

# 複製專案檔案
COPY . .

# 安裝 Python 套件
RUN pip install --no-cache-dir -r requirements.txt

# 暴露 FastAPI 預設埠
EXPOSE 8000

# 啟動命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
