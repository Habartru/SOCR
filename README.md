# 🚀 SuperOCR - Продвинутая система OCR с искусственным интеллектом

<div align="center">

![SuperOCR Logo](https://img.shields.io/badge/SuperOCR-v2.1-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-orange?style=for-the-badge&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)

**Мощная система для массовой обработки документов с использованием Surya OCR и больших языковых моделей**

</div>

---

## 🎯 Особенности

SuperOCR - это профессиональное решение для автоматизации обработки документов, объединяющее передовые технологии распознавания текста и искусственного интеллекта.

## ✨ Основные возможности

- 📄 **Массовая обработка PDF** - параллельная обработка множества документов
- 🔍 **Высокоточное OCR** - распознавание текста с использованием Surya OCR
- 🤖 **LLM анализ** - структурированный анализ содержимого документов
- ⚡ **Высокая производительность** - многопоточная обработка
- 📊 **Мониторинг в реальном времени** - отслеживание прогресса и статистики
- 📊 **Аналитика**: Автоматическое создание отчетов и статистики
- **💾 Экспорт**: Сохранение результатов в различных форматах (JSON, CSV, TXT)
- 🛑 **Контроль процессов** - возможность остановки обработки

## 📈 Производительность

### 🔥 Тесты на NVIDIA RTX 3090

| Компонент | Время обработки | Описание |
|-----------|-------------------|------------|
| **OCR (Surya)** | **~7 секунд** | Распознавание текста на страницу |
| **LLM обработка** | **10-20 секунд** | Локальная модель (LM Studio) |
| **Общее время** | **17-27 секунд** | Полная обработка страницы |

### ☁️ Сравнение с облачными решениями

| Платформа | Оборудование | Время обработки | Преимущества |
|-----------|-------------|-------------------|-------------|
| **SuperOCR (RTX 3090)** | RTX 3090 | **17-27 сек** | Локально, без ограничений |
| **OpenAI GPT-4** | H100+ | **13-20 сек** | Высокая скорость, но платно |
| **Локальные модели** | RTX 3090 | **10-20 сек** | Полная конфиденциальность |

> 💡 **Преимущество SuperOCR**: Полностью локальное решение без ограничений по количеству запросов и с полной конфиденциальностью данных

## 🎯 Поддерживаемые LLM провайдеры

- **LM Studio** - локальные модели (рекомендуется для конфиденциальности)
  - Рекомендуемая модель: **Microsoft/phi-4** (6B параметров)
  - Также доступна облегченная версия: **Microsoft/phi-4-Q3_K_L**
  - При настройке указывайте имя модели `local-1` (для первой) и `local-2` (для второй параллельной)
- **OpenAI API** - рекомендуются модели:
  - **gpt-4o-mini** (оптимальная производительность)
  - **gpt-4.1-mini** и **gpt-4.1-nano** (отличное соотношение цена/качество)
  - Требуется указать действующий API ключ

## 🚀 Быстрый старт

### 📋 Системные требования

- **Python**: 3.10.x (обязательно)
- **GPU**: NVIDIA с CUDA 11.8+ (рекомендуется RTX 3090 или выше)
- **RAM**: минимум 8 ГБ (рекомендуется 16 ГБ)
- **Место на диске**: минимум 5 ГБ свободного места
- **ОС**: Windows 10/11 или Linux (Ubuntu 20.04+)

### 💻 Установка

#### 🪟 Windows (Автоматическая установка)
```cmd
# Клонирование репозитория
git clone https://github.com/habartru/SuperOCR.git
cd SuperOCR

# Автоматическая установка
.\install_final.bat

# Запуск приложения
.\launch.bat
```

#### Windows (Ручная установка)
```cmd
# 1. Создание виртуального окружения
python -m venv surya_env

# 2. Активация окружения
surya_env\Scripts\Activate.ps1

# 3. Установка зависимостей
pip install -r requirements.txt

# 4. Установка Surya OCR без зависимостей
pip install surya-ocr==0.14.6 --no-deps

# 5. Установка недостающих зависимостей
pip install pydantic pydantic-settings filetype pre-commit

# 6. Обновление PyTorch (если нужно)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# 7. Запуск приложения
python gui_run.py
```

#### Linux
```bash
# Клонирование репозитория
git clone https://github.com/habartru/SuperOCR.git
cd SuperOCR

# 1. Создание виртуального окружения
python3 -m venv surya_env

# 2. Активация окружения
source surya_env/bin/activate

# 3. Установка зависимостей
pip install -r requirements.txt

# 4. Установка Surya OCR без зависимостей
pip install surya-ocr==0.14.6 --no-deps

# 5. Установка недостающих зависимостей
pip install pydantic pydantic-settings filetype pre-commit

# 6. Обновление PyTorch для совместимости
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# 7. Запуск приложения
python3 gui_run.py
```

#### 🐳 Docker (Универсальное решение)

**Предварительные требования:**
- Docker с NVIDIA Container Toolkit
- NVIDIA GPU с CUDA 11.8+

```bash
# Клонирование репозитория
git clone https://github.com/Habartru/SOCR.git
cd SOCR

# Создание конфигурационного файла
cp local.env.example local.env

# Запуск с Docker Compose
docker-compose up -d

# Подключение к GUI через VNC
# Откройте VNC клиент и подключитесь к localhost:5900
```

**Открытые порты:**
- `5900` - VNC для доступа к GUI
- `6080` - noVNC (веб-интерфейс)
- `1234` - LM Studio API (для локальных LLM)

📄 **Полная инструкция по установке на Linux**: [LINUX_INSTALL.md](LINUX_INSTALL.md)

#### Docker
```bash
# Сборка и запуск
docker-compose up -d

# Подключение через VNC: localhost:5900
# Или через веб: http://localhost:6080
```

### Настройка

1. **Выберите папку с PDF файлами**
2. **Настройте количество потоков** для OCR и LLM
3. **Выберите LLM провайдера**:
   - Для LM Studio: укажите endpoint (по умолчанию http://localhost:1234)
   - Для OpenAI: введите API ключ
4. **Нажмите "Начать обработку"**

## 📁 Результаты обработки

Система создает следующие файлы:

- `ocr_result.csv` - сводная таблица всех обработанных документов
- `document_name.json` - детальные результаты для каждого документа
- Логи обработки в реальном времени

## 🔧 Системные требования

### 💻 Поддерживаемые платформы
- **Windows**: 10/11 (нативная поддержка)
- **Linux**: Ubuntu 20.04+, CentOS 8+, Debian 11+
- **Docker**: Любая платформа с Docker и NVIDIA Container Toolkit

### ⚙️ Минимальные требования
- **Python**: 3.8+
- **GPU**: NVIDIA GPU с CUDA 11.8+ (рекомендуется)
- **ОЗУ**: Минимум 8GB, рекомендуется 16GB+
- **Место на диске**: 10GB+ свободного места

## 📊 Возможности GUI

- **Выбор папки** с PDF документами
- **Настройка потоков** для OCR и LLM обработки
- **Выбор LLM провайдера** и модели
- **Автоповтор** при ошибках LLM
- **Статистика в реальном времени**:
  - Количество обработанных документов
  - Среднее время обработки
  - Скорость обработки (документов/минуту)
- **Логи обработки** с детальной информацией
- **Кнопка остановки** для прерывания процесса

## 🔒 Безопасность

- **Локальная обработка** - все данные остаются на вашем компьютере
- **Маскирование API ключей** - безопасный ввод конфиденциальной информации
- **Контроль процессов** - принудительное завершение всех процессов при закрытии

## 📖 Документация

Подробная техническая документация доступна в файле [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md)

История изменений и улучшений: [CHANGELOG.md](Docs/CHANGELOG.md)

### 🆕 Последние улучшения (v2.1)

- ✅ **Улучшенное логирование** - исправлен вывод "Неопределен" в логах
- ✅ **Нормализация ключей JSON** - унифицированный формат с подчеркиванием
- ✅ **Улучшенная валидация данных** - более точная обработка ИНН и КПП
- ✅ **Агрессивное восстановление JSON** - предотвращение ошибок при разборе ответов LLM

## 🆘 Поддержка

При возникновении проблем:

1. Проверьте логи в GUI приложения
2. Убедитесь, что все зависимости установлены корректно
3. Проверьте доступность LLM сервиса (LM Studio или OpenAI API)

---

**🎯 SuperOCR - ваш надежный помощник в автоматизации обработки документов!**
