#!/usr/bin/env python3
"""
Точный подсчетчик токенов для OCR данных
Использует tiktoken для точного подсчета токенов
"""

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("⚠️ tiktoken не установлен, используем приблизительный подсчет")

# Инициализируем энкодер токенов
if TIKTOKEN_AVAILABLE:
    try:
        # Используем cl100k_base - стандартный энкодер для GPT-4 и подобных моделей
        TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"⚠️ Ошибка инициализации tiktoken: {e}")
        TIKTOKEN_AVAILABLE = False
        TOKEN_ENCODER = None
else:
    TOKEN_ENCODER = None

def estimate_tokens(text):
    """
    Точный подсчет токенов с помощью tiktoken
    Если tiktoken недоступен - используем приблизительный подсчет
    """
    if not text:
        return 0
    
    if TIKTOKEN_AVAILABLE and TOKEN_ENCODER:
        try:
            # Точный подсчет с tiktoken
            tokens = TOKEN_ENCODER.encode(text)
            return len(tokens)
        except Exception as e:
            print(f"⚠️ Ошибка tiktoken, используем приблизительный подсчет: {e}")
    
    # Приблизительный подсчет для русского текста
    char_count = len(text)
    estimated_tokens = char_count / 4.2  # 1 токен ≈ 4.2 символа для русского
    return int(estimated_tokens)

def estimate_tokens_from_lines(lines_data):
    """
    Подсчет токенов из структурированных данных OCR
    """
    if not lines_data:
        return 0
    
    total_text = ""
    for line in lines_data:
        if isinstance(line, dict) and 'text' in line:
            total_text += line['text'] + " "
        elif isinstance(line, str):
            total_text += line + " "
    
    return estimate_tokens(total_text)

def check_context_limit(lines_data, max_tokens=16000):
    """
    Проверяет, превышает ли документ лимит контекста
    Возвращает (превышает_лимит, количество_токенов)
    """
    token_count = estimate_tokens_from_lines(lines_data)
    exceeds_limit = token_count > max_tokens
    
    return exceeds_limit, token_count

def smart_truncate_for_llm(lines_data, max_tokens=12000):
    """
    Умное усечение документа для LLM с учетом реального размера JSON структуры
    
    Стратегия:
    1. Проверяем реальный размер JSON структуры
    2. Если не помещается - итеративно уменьшаем количество строк
    3. Используем точный подсчет токенов с tiktoken
    """
    import json
    
    # Проверяем реальный размер JSON структуры
    json_data = json.dumps(lines_data, ensure_ascii=False, indent=2)
    json_tokens = estimate_tokens(json_data)
    
    if json_tokens <= max_tokens:
        return lines_data, json_tokens, False  # документ, токены, был_усечен
    
    # Документ слишком большой - применяем усечение
    print(f"⚠️ Документ превышает лимит: {json_tokens} токенов > {max_tokens}")
    
    total_lines = len(lines_data)
    best_result = None
    
    # Итеративно уменьшаем размер документа
    for attempt in range(10):  # Максимум 10 попыток
        # Адаптивное усечение на основе размера документа и попытки
        reduction_factor = 0.3 + (attempt * 0.1)  # От 30% до 120% сокращения
        
        if json_tokens > max_tokens * 3:  # Очень большой документ
            header_count = max(10, int(total_lines * (0.1 - attempt * 0.01)))
            footer_count = max(15, int(total_lines * (0.15 - attempt * 0.015)))
        elif json_tokens > max_tokens * 2:  # Большой документ
            header_count = max(20, int(total_lines * (0.15 - attempt * 0.02)))
            footer_count = max(25, int(total_lines * (0.2 - attempt * 0.02)))
        else:  # Умеренно большой документ
            header_count = max(30, int(total_lines * (0.2 - attempt * 0.03)))
            footer_count = max(40, int(total_lines * (0.25 - attempt * 0.03)))
        
        # Убеждаемся, что не берем больше строк, чем есть
        header_count = min(header_count, total_lines // 3)
        footer_count = min(footer_count, total_lines // 3)
        
        # Минимальные значения для сохранения смысла
        if header_count < 5 or footer_count < 5:
            break
        
        truncated_lines = []
        
        # Добавляем заголовок
        header_lines = lines_data[:header_count]
        for line in header_lines:
            if isinstance(line, dict):
                line_copy = line.copy()
                line_copy['chunk_type'] = 'header'
                truncated_lines.append(line_copy)
            else:
                truncated_lines.append({'text': str(line), 'chunk_type': 'header'})
        
        # Добавляем разделитель
        truncated_lines.append({
            'text': f'[... ПРОПУЩЕНО {total_lines - header_count - footer_count} СТРОК ...]',
            'chunk_type': 'separator'
        })
        
        # Добавляем подвал
        footer_start = total_lines - footer_count
        footer_lines = lines_data[footer_start:]
        for line in footer_lines:
            if isinstance(line, dict):
                line_copy = line.copy()
                line_copy['chunk_type'] = 'footer'
                truncated_lines.append(line_copy)
            else:
                truncated_lines.append({'text': str(line), 'chunk_type': 'footer'})
        
        # Проверяем реальный размер JSON усеченного документа
        truncated_json = json.dumps(truncated_lines, ensure_ascii=False, indent=2)
        final_token_count = estimate_tokens(truncated_json)
        
        print(f"✂️ Попытка {attempt + 1}: {len(truncated_lines)} строк, {final_token_count} токенов")
        
        if final_token_count <= max_tokens:
            print(f"✅ Документ успешно усечен: {json_tokens} → {final_token_count} токенов")
            return truncated_lines, final_token_count, True
        
        best_result = (truncated_lines, final_token_count)
    
    # Если не удалось уместить в лимит, возвращаем лучший результат
    if best_result:
        print(f"⚠️ Не удалось уместить в лимит, возвращаем лучший результат: {best_result[1]} токенов")
        return best_result[0], best_result[1], True
    
    # Крайний случай - возвращаем только первые 10 строк
    minimal_lines = lines_data[:10]
    minimal_json = json.dumps(minimal_lines, ensure_ascii=False, indent=2)
    minimal_tokens = estimate_tokens(minimal_json)
    print(f"🚨 Крайний случай: возвращаем только {len(minimal_lines)} строк, {minimal_tokens} токенов")
    
    return minimal_lines, minimal_tokens, True

if __name__ == "__main__":
    # Тестирование
    test_text = "Это тестовый русский текст для проверки подсчета токенов"
    tokens = estimate_tokens(test_text)
    print(f"Текст: '{test_text}'")
    print(f"Символов: {len(test_text)}")
    print(f"Токенов (оценка): {tokens}")
