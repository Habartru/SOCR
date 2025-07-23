# 🐧 Установка SuperOCR на Linux

Полное руководство по установке и настройке SuperOCR на Linux серверах.

## 📋 Системные требования

- **ОС**: Ubuntu 20.04+ / CentOS 8+ / Debian 11+
- **Python**: 3.8+
- **GPU**: NVIDIA GPU с CUDA 11.8+ (рекомендуется)
- **ОЗУ**: Минимум 8GB, рекомендуется 16GB+
- **Место на диске**: 10GB+ свободного места

## 🚀 Быстрая установка

### 1. Обновление системы

```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y

# CentOS/RHEL
sudo yum update -y
# или для новых версий
sudo dnf update -y
```

### 2. Установка системных зависимостей

```bash
# Ubuntu/Debian
sudo apt install -y python3 python3-pip python3-venv git wget curl \
    build-essential libssl-dev libffi-dev python3-dev \
    libjpeg-dev libpng-dev libtiff-dev libopenjp2-7-dev \
    pkg-config libhdf5-dev

# CentOS/RHEL
sudo yum install -y python3 python3-pip python3-devel git wget curl \
    gcc gcc-c++ make openssl-devel libffi-devel \
    libjpeg-turbo-devel libpng-devel libtiff-devel openjpeg2-devel \
    pkgconfig hdf5-devel
```

### 3. Установка CUDA (для GPU ускорения)

```bash
# Скачивание и установка CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Добавление в PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 4. Клонирование проекта

```bash
git clone https://github.com/YOUR_USERNAME/SuperOCR.git
cd SuperOCR
```

### 5. Создание виртуального окружения

```bash
python3 -m venv surya_env
source surya_env/bin/activate
```

### 6. Установка Python зависимостей

```bash
# Обновление pip
pip install --upgrade pip setuptools wheel

# Установка PyTorch с CUDA поддержкой
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Установка основных зависимостей
pip install surya-ocr requests pillow numpy opencv-python
pip install pdf2image pymupdf python-dotenv
```

### 7. Настройка окружения

```bash
# Копирование конфигурационного файла
cp local.env.example local.env

# Редактирование настроек
nano local.env
```

### 8. Проверка установки

```bash
# Проверка CUDA
python3 -c "import torch; print(f'CUDA доступна: {torch.cuda.is_available()}')"

# Проверка Surya OCR
python3 -c "from surya.detection import DetectionPredictor; print('Surya OCR установлена успешно')"
```

## 🖥️ Настройка для серверного использования

### Установка X11 forwarding (для GUI)

```bash
# Ubuntu/Debian
sudo apt install -y xvfb x11vnc fluxbox

# CentOS/RHEL
sudo yum install -y xorg-x11-server-Xvfb x11vnc fluxbox
```

### Запуск с виртуальным дисплеем

```bash
# Создание виртуального дисплея
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &

# Запуск приложения
source surya_env/bin/activate
python3 gui_run.py
```

### Настройка VNC для удаленного доступа

```bash
# Запуск VNC сервера
x11vnc -display :99 -nopw -listen localhost -xkb &

# Подключение через SSH туннель
# На локальной машине:
ssh -L 5900:localhost:5900 user@your-server.com
```

## 🐳 Docker установка (альтернативный способ)

### Создание Dockerfile

```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git \
    build-essential libssl-dev libffi-dev python3-dev \
    libjpeg-dev libpng-dev libtiff-dev libopenjp2-7-dev \
    pkg-config libhdf5-dev \
    xvfb x11vnc fluxbox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN python3 -m venv surya_env && \
    . surya_env/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install surya-ocr requests pillow numpy opencv-python && \
    pip install pdf2image pymupdf python-dotenv

EXPOSE 5900

CMD ["bash", "-c", "source surya_env/bin/activate && Xvfb :99 -screen 0 1024x768x24 & export DISPLAY=:99 && python3 gui_run.py"]
```

### Сборка и запуск Docker контейнера

```bash
# Сборка образа
docker build -t superocr .

# Запуск контейнера
docker run --gpus all -p 5900:5900 -v $(pwd)/data:/app/data superocr
```

## 🔧 Настройка systemd сервиса

### Создание сервисного файла

```bash
sudo nano /etc/systemd/system/superocr.service
```

```ini
[Unit]
Description=SuperOCR Document Processing Service
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/SuperOCR
Environment=DISPLAY=:99
ExecStartPre=/usr/bin/Xvfb :99 -screen 0 1024x768x24
ExecStart=/path/to/SuperOCR/surya_env/bin/python gui_run.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Активация сервиса

```bash
sudo systemctl daemon-reload
sudo systemctl enable superocr
sudo systemctl start superocr
sudo systemctl status superocr
```

## 🌐 Настройка веб-интерфейса (опционально)

Для удаленного доступа через веб-браузер можно использовать noVNC:

```bash
# Установка noVNC
git clone https://github.com/novnc/noVNC.git
cd noVNC

# Запуск веб-сервера
./utils/launch.sh --vnc localhost:5900 --listen 6080
```

Теперь можно подключиться через браузер: `http://your-server:6080`

## 🛠️ Устранение неполадок

### Проблемы с CUDA

```bash
# Проверка драйверов NVIDIA
nvidia-smi

# Переустановка CUDA драйверов
sudo apt purge nvidia-* -y
sudo apt autoremove -y
sudo apt install nvidia-driver-520 -y
sudo reboot
```

### Проблемы с зависимостями

```bash
# Переустановка виртуального окружения
rm -rf surya_env
python3 -m venv surya_env
source surya_env/bin/activate
pip install --upgrade pip
# Повторить установку зависимостей
```

### Проблемы с GUI

```bash
# Проверка X11
echo $DISPLAY
xdpyinfo

# Перезапуск виртуального дисплея
pkill Xvfb
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

## 📊 Мониторинг производительности

```bash
# Мониторинг GPU
watch -n 1 nvidia-smi

# Мониторинг ресурсов
htop

# Логи приложения
tail -f /var/log/superocr.log
```

## 🔒 Безопасность

```bash
# Настройка файрвола
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 5900/tcp  # VNC (только для доверенных IP)
sudo ufw allow 6080/tcp  # noVNC (только для доверенных IP)
sudo ufw enable

# Создание отдельного пользователя
sudo useradd -m -s /bin/bash superocr
sudo usermod -aG sudo superocr
```

---

## 🎯 Готово!

После выполнения всех шагов SuperOCR будет готов к работе на Linux сервере. Для запуска используйте:

```bash
cd SuperOCR
source surya_env/bin/activate
python3 gui_run.py
```

Или подключитесь через VNC/noVNC для использования графического интерфейса.
