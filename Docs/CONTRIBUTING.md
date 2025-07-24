# 🤝 Руководство по внесению вклада в SuperOCR

Спасибо за интерес к проекту SuperOCR! Мы приветствуем вклад от сообщества.

## 🚀 Как начать

1. **Fork** репозитория
2. **Clone** вашего fork
3. Создайте **feature branch** (`git checkout -b feature/amazing-feature`)
4. Внесите изменения
5. **Commit** ваши изменения (`git commit -m 'Add amazing feature'`)
6. **Push** в branch (`git push origin feature/amazing-feature`)
7. Создайте **Pull Request**

## 📋 Требования к коду

### Python Code Style
- Следуйте [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Используйте `flake8` для проверки стиля
- Максимальная длина строки: 127 символов

### Тестирование
- Добавляйте тесты для новой функциональности
- Убедитесь, что все существующие тесты проходят
- Используйте `pytest` для запуска тестов

### Документация
- Обновляйте документацию при изменении API
- Используйте docstrings для функций и классов
- Обновляйте README.md при необходимости

## 🐛 Сообщение об ошибках

При сообщении об ошибке, пожалуйста, включите:

- Версию Python
- Версию CUDA (если применимо)
- Модель GPU
- Подробное описание проблемы
- Шаги для воспроизведения
- Логи ошибок

## 💡 Предложение новых функций

Перед началом работы над новой функцией:

1. Создайте Issue с описанием предлагаемой функции
2. Дождитесь обратной связи от мейнтейнеров
3. Начинайте разработку после одобрения

## 🔧 Настройка среды разработки

```bash
# Клонирование репозитория
git clone https://github.com/Habartru/SOCR.git
cd SOCR

# Создание виртуального окружения
python3 -m venv dev_env
source dev_env/bin/activate  # Linux/Mac
# или
dev_env\Scripts\activate  # Windows

# Установка зависимостей для разработки
pip install -r requirements.txt
pip install -r requirements-dev.txt  # если есть

# Установка pre-commit hooks
pre-commit install
```

## 📝 Commit Messages

Используйте понятные commit messages:

- `feat:` - новая функция
- `fix:` - исправление ошибки
- `docs:` - изменения в документации
- `style:` - форматирование кода
- `refactor:` - рефакторинг кода
- `test:` - добавление тестов
- `chore:` - обновление зависимостей, настройки

Пример: `feat: add support for new OCR model`

## 🏷️ Версионирование

Проект использует [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- MAJOR: breaking changes
- MINOR: новая функциональность (обратно совместимая)
- PATCH: исправления ошибок

## 📞 Связь

- GitHub Issues: для багов и предложений
- GitHub Discussions: для общих вопросов
- Email: [ваш-email] для приватных вопросов

## 📄 Лицензия

Внося вклад в проект, вы соглашаетесь с тем, что ваш код будет лицензирован под MIT License.
