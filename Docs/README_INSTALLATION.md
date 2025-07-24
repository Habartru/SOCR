# SuperOCR Installation Guide

## 🚀 Полное руководство по установке SuperOCR

### 📋 Системные требования

- **Python**: 3.10.x (обязательно)
- **ОС**: Windows 10/11, Linux (Ubuntu 20.04+)
- **GPU**: NVIDIA GPU с CUDA 11.8+ (рекомендуется)
- **RAM**: минимум 8 ГБ, рекомендуется 16 ГБ
- **Свободное место**: минимум 5 ГБ

---

## 🖥️ Windows - Автоматическая установка (Рекомендуется)

### Шаг 1: Клонирование репозитория
```cmd
git clone https://github.com/habartru/SuperOCR.git
cd SuperOCR
```

### Шаг 2: Запуск автоматической установки
```cmd
.\install_final.bat
```

**Что делает батник:**
- ✅ Проверяет версию Python
- ✅ Создает виртуальное окружение
- ✅ Устанавливает все зависимости в правильном порядке
- ✅ Устанавливает PyTorch 2.7.1 с CUDA поддержкой
- ✅ Устанавливает Surya OCR без конфликтов зависимостей
- ✅ Тестирует установку
- ✅ Создает launch.bat для запуска

### Шаг 3: Запуск приложения
```cmd
.\launch.bat
```

---

## 🖥️ Windows - Ручная установка

### Шаг 1: Создание виртуального окружения
```cmd
python -m venv surya_env
```

### Шаг 2: Активация окружения
```cmd
surya_env\Scripts\Activate.ps1
```
**Важно:** После активации в начале строки должно появиться `(surya_env)`

### Шаг 3: Установка основных зависимостей
```cmd
pip install -r requirements.txt
```

### Шаг 4: Установка Surya OCR (БЕЗ зависимостей)
```cmd
pip install surya-ocr==0.14.6 --no-deps
```
**Важно:** Флаг `--no-deps` предотвращает переустановку PyTorch

### Шаг 5: Установка недостающих зависимостей Surya
```cmd
pip install pydantic pydantic-settings filetype pre-commit
```

### Шаг 6: Обновление PyTorch для совместимости
```cmd
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

### Шаг 7: Запуск приложения
```cmd
python gui_run.py
```

---

## 🐧 Linux - Установка

### Шаг 1: Клонирование репозитория
```bash
git clone https://github.com/habartru/SuperOCR.git
cd SuperOCR
```

### Шаг 2: Создание виртуального окружения
```bash
python3 -m venv surya_env
```

### Шаг 3: Активация окружения
```bash
source surya_env/bin/activate
```

### Шаг 4: Установка зависимостей
```bash
pip install -r requirements.txt
```

### Шаг 5: Установка Surya OCR
```bash
pip install surya-ocr==0.14.6 --no-deps
```

### Шаг 6: Установка недостающих зависимостей
```bash
pip install pydantic pydantic-settings filetype pre-commit
```

### Шаг 7: Обновление PyTorch
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

### Шаг 8: Запуск приложения
```bash
python3 gui_run.py
```

---

## 🔧 Проверка установки

### Проверка PyTorch и CUDA
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

### Проверка Surya OCR
```python
import surya
print("Surya OCR imported successfully")
```

---

## ❗ Устранение проблем

### Проблема: "ModuleNotFoundError: No module named 'requests'"
**Решение:** Убедитесь, что виртуальное окружение активировано
```cmd
surya_env\Scripts\Activate.ps1  # Windows
source surya_env/bin/activate   # Linux
```

### Проблема: "TypeError: is_bf16_supported() got an unexpected keyword argument"
**Решение:** Обновите PyTorch до версии 2.7.1
```cmd
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

### Проблема: Конфликты зависимостей при установке Surya OCR
**Решение:** Используйте флаг `--no-deps`
```cmd
pip install surya-ocr==0.14.6 --no-deps
```

### Проблема: CUDA недоступна
**Решение:** 
1. Убедитесь, что установлены драйверы NVIDIA
2. Проверьте совместимость CUDA версии
3. При необходимости установите CPU версию PyTorch

---

## 📁 Структура проекта после установки

```
SuperOCR/
├── gui_run.py              # Основное приложение
├── launch.bat              # Скрипт запуска (Windows)
├── install_final.bat       # Скрипт установки (Windows)
├── requirements.txt        # Зависимости проекта
├── surya_env/             # Виртуальное окружение
├── local.env              # Конфигурация
└── Docs/                  # Документация
```

---

## ✅ Успешная установка завершена!

### Что было исправлено в последней версии:

1. **Проблема с кодировкой** - Убраны все кириллические символы из batch файлов
2. **Конфликт версий PyTorch** - Установлен PyTorch 2.7.1 с CUDA поддержкой
3. **Проблема с Surya OCR** - Исправлена совместимость с новой версией PyTorch
4. **Зависимости** - Все пакеты установлены в правильном порядке
5. **Автоматизация** - Создан надежный install_final.bat

### Как запустить приложение:

1. **Основной способ (Windows)**: `.\.\launch.bat`
2. **Альтернативный способ**: 
   ```cmd
   surya_env\Scripts\Activate.ps1
   python gui_run.py
   ```
3. **Linux**: 
   ```bash
   source surya_env/bin/activate
   python3 gui_run.py
   ```

- ✅ Python 3.10.11
- ✅ PyTorch 2.3.1+cu118 (CUDA поддержка)
- ✅ CUDA доступна: True
- ✅ GPU count: 1
- ✅ NumPy 1.26.4 (совместимая версия)
- ✅ Все основные зависимости установлены
- ✅ Surya OCR исправлен и работает

### Файлы установки:

- `install_final.bat` - Основной установщик
- `fix_numpy.bat` - Исправление NumPy
- `launch.bat` - Запуск приложения
- Исправлен файл: `surya_env\lib\site-packages\surya\recognition\loader.py`

### Примечания:

- Приложение использует CUDA для ускорения OCR
- Все конфликты зависимостей разрешены
- Установка протестирована и работает

Приложение готово к использованию!
