# SuperOCR Docker Image
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Предотвращение интерактивных запросов при установке пакетов
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:99

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenjp2-7-dev \
    pkg-config \
    libhdf5-dev \
    xvfb \
    x11vnc \
    fluxbox \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование файлов проекта
COPY requirements.txt .
COPY local.env.example local.env
COPY gui_run.py .
COPY surya/ ./surya/
COPY *.md ./

# Создание виртуального окружения и установка зависимостей
RUN python3 -m venv surya_env && \
    . surya_env/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install -r requirements.txt

# Создание директорий для данных
RUN mkdir -p /app/data/input /app/data/output /app/logs

# Создание скрипта запуска
RUN echo '#!/bin/bash\n\
source /app/surya_env/bin/activate\n\
Xvfb :99 -screen 0 1024x768x24 &\n\
export DISPLAY=:99\n\
cd /app\n\
python3 gui_run.py' > /app/start.sh && \
    chmod +x /app/start.sh

# Создание скрипта для VNC
RUN echo '#!/bin/bash\n\
source /app/surya_env/bin/activate\n\
Xvfb :99 -screen 0 1024x768x24 &\n\
export DISPLAY=:99\n\
x11vnc -display :99 -nopw -listen 0.0.0.0 -xkb &\n\
cd /app\n\
python3 gui_run.py' > /app/start_vnc.sh && \
    chmod +x /app/start_vnc.sh

# Открытие портов
EXPOSE 5900 6080

# Создание пользователя без root прав
RUN useradd -m -s /bin/bash superocr && \
    chown -R superocr:superocr /app

USER superocr

# Команда по умолчанию
CMD ["/app/start_vnc.sh"]
