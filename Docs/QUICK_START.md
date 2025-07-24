# 🚀 SuperOCR - Быстрый старт

## Windows (Рекомендуемый способ)

```cmd
# 1. Клонирование проекта
git clone https://github.com/habartru/SuperOCR.git
cd SuperOCR

# 2. Автоматическая установка
.\install_final.bat

# 3. Запуск приложения
.\launch.bat
```

## Linux

```bash
# 1. Клонирование проекта
git clone https://github.com/habartru/SuperOCR.git
cd SuperOCR

# 2. Создание и активация виртуального окружения
python3 -m venv surya_env
source surya_env/bin/activate

# 3. Установка зависимостей (в правильном порядке!)
pip install -r requirements.txt
pip install surya-ocr==0.14.6 --no-deps
pip install pydantic pydantic-settings filetype pre-commit
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# 4. Запуск приложения
python3 gui_run.py
```

## Системные требования

- **Python**: 3.10.x (обязательно)
- **GPU**: NVIDIA с CUDA 11.8+ (рекомендуется)
- **RAM**: минимум 8 ГБ
- **Место**: минимум 5 ГБ

## Устранение проблем

- **Ошибка импорта**: Убедитесь, что виртуальное окружение активировано
- **Конфликт PyTorch**: Используйте флаг `--no-deps` при установке Surya OCR
- **CUDA недоступна**: Проверьте драйверы NVIDIA

📖 **Полная документация**: [Docs/README_INSTALLATION.md](Docs/README_INSTALLATION.md)
