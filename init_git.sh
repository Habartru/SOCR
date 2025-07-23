#!/bin/bash

echo "Инициализация Git репозитория для SuperOCR..."

# Инициализация Git
git init

# Добавление всех файлов
git add .

# Первый коммит
git commit -m "Initial commit: SuperOCR - Система автоматизированной обработки документов

- Добавлена поддержка Surya OCR для высокоточного распознавания текста
- Интеграция с LLM (LM Studio и OpenAI API) для анализа документов
- Многопоточная обработка PDF файлов
- GUI интерфейс с мониторингом в реальном времени
- Поддержка Windows и Linux
- Docker контейнеризация
- Система автоповтора при ошибках
- Подробная документация и инструкции по установке"

echo ""
echo "Git репозиторий инициализирован!"
echo ""
echo "Следующие шаги:"
echo "1. Создайте репозиторий на GitHub"
echo "2. Выполните команды:"
echo "   git remote add origin https://github.com/habartru/SuperOCR.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
