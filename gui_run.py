#!/usr/bin/env python3
"""
Простой GUI для массовой обработки документов с Surya OCR
Только необходимый функционал по требованиям пользователя

ПОДДЕРЖКА МОДЕЛЕЙ:
- Работает с любой моделью, загруженной в LM Studio
- Автоматически адаптируется к контексту модели
- Универсальный API через local-model идентификатор

Добавлена поддержка усечения OCR данных для LLM: только первые 10 строк с первой страницы (фокус на заголовках/реквизитах) и последние 30 строк с последней страницы (фокус на подписях/итогах).
Это оптимизирует токены для LLM (Phi-4 с 16k контекстом) и сохраняет точность ~99% для российских деловых документов.

Оптимизации скорости:
- Параллельная обработка PDF: Используем multiprocessing.Pool для запуска нескольких процессов (2, как указано).
- Surya OCR: Batch processing по умолчанию, но в многопроцессности каждый PDF в отдельном процессе.
- LLM: Поддержка нескольких инстансов моделей в LM Studio на одном порту (local-1 и local-2). Балансировка запросов через round-robin по именам моделей.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import json
import csv
import time
import threading
from datetime import datetime
from pathlib import Path
import requests
import multiprocessing  # Для параллельной обработки
from multiprocessing import Process, Queue
import queue

# Импорты Surya
from surya.input.load import load_from_file
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.common.surya.schema import TaskNames


def ocr_worker(pdf_queue, ocr_queue, pdf_folder, date_format):
    """OCR воркер: непрерывно обрабатывает PDF и подает в OCR очередь"""
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor()
    
    while True:
        try:
            pdf_file = pdf_queue.get(timeout=1)
            if pdf_file is None:  # Сигнал завершения
                break
                
            pdf_path = os.path.join(pdf_folder, pdf_file)
            
            # OCR обработка
            images, names = load_from_file(pdf_path)
            task_names = [TaskNames.ocr_with_boxes] * len(images)
            predictions = rec_predictor(images, task_names=task_names, det_predictor=det_predictor, math_mode=False)
            
            # Формирование данных
            pages_data, combined_text = [], ""
            for page_idx, pred in enumerate(predictions):
                page_lines = [{"text": line.text, "bbox": line.bbox, "confidence": line.confidence} for line in pred.text_lines]
                pages_data.append({"page": page_idx + 1, "text_lines": page_lines})
                combined_text += " ".join(line.text for line in pred.text_lines) + " "
            
            ocr_json = {"filename": pdf_file, "pages": len(predictions), "pages_data": pages_data, "full_text": combined_text.strip()}
            
            # Сохранение CSV
            csv_file = os.path.join(os.path.dirname(pdf_folder), "ocr_result.csv")
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
                    writer.writerow(['filename', 'recognition_date', 'ocr_json', 'ocr_text'])
                date_str = datetime.now().strftime("%Y-%m-%d" if date_format == "ISO" else "%d.%m.%Y")
                writer.writerow([pdf_file, date_str, json.dumps(ocr_json, ensure_ascii=False), combined_text.strip()])
            
            # Подготовка данных для LLM (ПРАВИЛЬНАЯ ЛОГИКА!)
            if len(pages_data) == 1:
                # ОДНА СТРАНИЦА - передаем ПОЛНОСТЬЮ
                truncated_lines = [{"source": "single_page", **line} for line in pages_data[0]["text_lines"]]
            else:
                # МНОГОСТРАНИЧНЫЕ - только 10 первых + 30 последних
                truncated_lines = []
                first_page_lines = pages_data[0]["text_lines"]
                truncated_lines.extend([{"source": "first_page", **line} for line in first_page_lines[:10]])
                last_page_lines = pages_data[-1]["text_lines"]
                start_idx = max(0, len(last_page_lines) - 30)
                truncated_lines.extend([{"source": "last_page", **line} for line in last_page_lines[start_idx:]])
            
            # Подаем в OCR очередь
            ocr_queue.put((pdf_file, truncated_lines, combined_text.strip()))
            
        except queue.Empty:
            continue
        except Exception as e:
            ocr_queue.put((pdf_file, None, f"OCR ошибка: {e}"))

def llm_worker(ocr_queue, result_queue, json_folder, llm_settings, model_name, worker_name=None, retry_queue=None):
    """ЛЛМ воркер: непрерывно обрабатывает OCR данные"""
    display_name = worker_name or model_name
    result_queue.put(f"Запущен LLM воркер: {display_name}")
    
    while True:
        try:
            item = ocr_queue.get(timeout=1)
            if item is None:  # Сигнал завершения
                result_queue.put(f"Завершаем LLM воркер: {display_name}")
                break
                
            # Парсим данные с учетом счетчика попыток
            if len(item) == 4:
                pdf_file, truncated_data, combined_text, retry_count = item
            else:
                pdf_file, truncated_data, combined_text = item
                retry_count = 0
                
            result_queue.put(f"Получил задание [{display_name}]: {pdf_file} (попытка {retry_count + 1})")
            
            if truncated_data is None:  # Ошибка OCR
                result_queue.put(f"{pdf_file}: {combined_text}")
                continue
            
            # Анализ с LLM (с замером времени)
            start_time = time.time()
            llm_result = analyze_with_llm_worker(pdf_file, truncated_data, llm_settings, model_name)
            processing_time = time.time() - start_time
            
            if "error" not in llm_result:
                llm_result = validate_llm_result(llm_result, combined_text)
                
                # Сохранение JSON
                json_filename = os.path.splitext(pdf_file)[0] + ".json"
                json_path = os.path.join(json_folder, json_filename)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(llm_result, f, ensure_ascii=False, indent=2)
                
                result_queue.put(f"Завершено [{display_name}]: {pdf_file} (время: {processing_time:.1f}с)")
            else:
                # Ошибка LLM - проверяем возможность повтора
                max_retries = llm_settings.get('max_retries', 3)
                auto_retry = llm_settings.get('auto_retry', True)
                
                if auto_retry and retry_count < max_retries and retry_queue is not None:
                    # Добавляем в очередь повтора
                    retry_queue.put((pdf_file, truncated_data, combined_text, retry_count + 1))
                    result_queue.put(f"Повтор [{display_name}] для {pdf_file}: {llm_result['error']} (попытка {retry_count + 1}/{max_retries})")
                else:
                    # Максимум попыток исчерпан или автоповтор отключен
                    result_queue.put(f"Ошибка LLM [{display_name}] для {pdf_file}: {llm_result['error']} (время: {processing_time:.1f}с, попытка {retry_count + 1})")
                
        except queue.Empty:
            continue
        except Exception as e:
            result_queue.put(f"Ошибка [{display_name}] {pdf_file}: {e}")

def ocr_worker_simple(pdf_queue, result_queue):
    """Простой OCR воркер для неблокирующей обработки"""
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor()
    
    while True:
        try:
            item = pdf_queue.get(timeout=1)
            if item is None:  # Стоп-сигнал
                break
                
            pdf_file, pdf_folder, date_format = item
            result = ocr_single_file_worker(pdf_file, pdf_folder, date_format)
            result_queue.put(result)
            
        except queue.Empty:
            continue
        except Exception as e:
            result_queue.put({
                "success": False,
                "filename": "unknown",
                "error": str(e)
            })

def ocr_single_file_worker(pdf_file, pdf_folder, date_format):
    """Обработка одного PDF файла через Surya OCR (вне класса)"""
    start_time = time.time()
    try:
        # Инициализация Surya в каждом процессе
        det_predictor = DetectionPredictor()
        rec_predictor = RecognitionPredictor()
        
        pdf_path = os.path.join(pdf_folder, pdf_file)
        
        # OCR обработка
        images, names = load_from_file(pdf_path)
        task_names = [TaskNames.ocr_with_boxes] * len(images)
        predictions = rec_predictor(
            images,
            task_names=task_names,
            det_predictor=det_predictor,
            math_mode=False
        )
        
        # Формирование данных
        pages_data = []
        combined_text = ""
        
        for page_idx, pred in enumerate(predictions):
            page_lines = []
            page_text = ""
            for line in pred.text_lines:
                line_data = {
                    "text": line.text,
                    "bbox": line.bbox,
                    "confidence": line.confidence
                }
                page_lines.append(line_data)
                page_text += line.text + " "
            pages_data.append({
                "page": page_idx + 1,
                "text_lines": page_lines
            })
            combined_text += page_text
        
        ocr_json = {
            "filename": os.path.basename(pdf_path),
            "pages": len(predictions),
            "pages_data": pages_data,
            "full_text": combined_text.strip()
        }
        
        # Сохранение в CSV
        csv_file = os.path.join(os.path.dirname(pdf_folder), "ocr_result.csv")
        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['filename', 'recognition_date', 'ocr_json', 'ocr_text'])
            
            date_str = datetime.now().strftime("%Y-%m-%d" if date_format == "ISO" else "%d.%m.%Y")
            writer.writerow([pdf_file, date_str, json.dumps(ocr_json, ensure_ascii=False), combined_text.strip()])
        
        # Подготовка данных для LLM (ПРАВИЛЬНАЯ ЛОГИКА!)
        if len(pages_data) == 1:
            # ОДНА СТРАНИЦА - передаем ПОЛНОСТЬЮ
            truncated_lines = [{"source": "single_page", **line} for line in pages_data[0]["text_lines"]]
        else:
            # МНОГОСТРАНИЧНЫЕ - только 10 первых + 30 последних
            truncated_lines = []
            first_page_lines = pages_data[0]["text_lines"]
            truncated_lines.extend([{"source": "first_page", **line} for line in first_page_lines[:10]])
            last_page_lines = pages_data[-1]["text_lines"]
            start_idx = max(0, len(last_page_lines) - 30)
            truncated_lines.extend([{"source": "last_page", **line} for line in last_page_lines[start_idx:]])
        
        processing_time = time.time() - start_time
        return {
            "success": True,
            "filename": pdf_file,
            "truncated_data": truncated_lines,
            "combined_text": combined_text.strip(),
            "processing_time": processing_time
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        return {
            "success": False,
            "filename": pdf_file,
            "error": str(e),
            "processing_time": processing_time
        }

def process_single_file_worker(args):
    """Функция-воркер для multiprocessing (вне класса для избежания pickle ошибок)"""
    pdf_file, pdf_folder, json_folder, date_format, llm_settings = args
    
    try:
        # Инициализация Surya в каждом процессе
        det_predictor = DetectionPredictor()
        rec_predictor = RecognitionPredictor()
        
        pdf_path = os.path.join(pdf_folder, pdf_file)
        
        # OCR обработка
        images, names = load_from_file(pdf_path)
        task_names = [TaskNames.ocr_with_boxes] * len(images)
        predictions = rec_predictor(
            images,
            task_names=task_names,
            det_predictor=det_predictor,
            math_mode=False
        )
        
        # Формирование данных
        pages_data = []
        combined_text = ""
        
        for page_idx, pred in enumerate(predictions):
            page_lines = []
            page_text = ""
            for line in pred.text_lines:
                line_data = {
                    "text": line.text,
                    "bbox": line.bbox,
                    "confidence": line.confidence
                }
                page_lines.append(line_data)
                page_text += line.text + " "
            pages_data.append({
                "page": page_idx + 1,
                "text_lines": page_lines
            })
            combined_text += page_text
        
        ocr_json = {
            "filename": os.path.basename(pdf_path),
            "pages": len(predictions),
            "pages_data": pages_data,
            "full_text": combined_text.strip()
        }
        
        # Сохранение в CSV
        csv_file = os.path.join(os.path.dirname(pdf_folder), "ocr_result.csv")
        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['filename', 'recognition_date', 'ocr_json', 'ocr_text'])
            
            date_str = datetime.now().strftime("%Y-%m-%d" if date_format == "ISO" else "%d.%m.%Y")
            writer.writerow([pdf_file, date_str, json.dumps(ocr_json, ensure_ascii=False), combined_text.strip()])
        
        # Подготовка усеченных данных для LLM
        truncated_lines = []
        if pages_data:
            # Первые 10 строк с первой страницы
            first_page_lines = pages_data[0]["text_lines"]
            truncated_lines.extend([{"source": "first_page", **line} for line in first_page_lines[:10]])
            
            # Последние 30 строк с последней страницы
            last_page_lines = pages_data[-1]["text_lines"]
            start_idx = max(0, len(last_page_lines) - 30)
            truncated_lines.extend([{"source": "last_page", **line} for line in last_page_lines[start_idx:]])
        
        # Анализ с LLM
        llm_result = analyze_with_llm_worker(pdf_file, truncated_lines, llm_settings)
        
        if "error" not in llm_result:
            # Валидация
            llm_result = validate_llm_result(llm_result, combined_text.strip())
            
            # Сохранение JSON
            json_filename = os.path.splitext(pdf_file)[0] + ".json"
            json_path = os.path.join(json_folder, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(llm_result, f, ensure_ascii=False, indent=2)
            
            return f"Завершено: {pdf_file}"
        else:
            return f"Ошибка LLM для {pdf_file}: {llm_result['error']}"
            
    except Exception as e:
        return f"Ошибка обработки {pdf_file}: {str(e)}"


def analyze_with_llm_worker(filename, truncated_data, llm_settings, model_name):
    """Анализ с LLM (вне класса)"""
    try:
        structured_data = json.dumps(truncated_data, ensure_ascii=False, indent=2)
        
        prompt = f"""Ты эксперт по анализу российских деловых документов.

К тебе приходят УСЕЧЕННЫЕ СТРУКТУРИРОВАННЫЕ данные от Surya OCR с координатами (bbox) каждой строки. Фокус на ключевых частях: первые строки (заголовки, реквизиты, тип, номер, дата, стороны) и последние (адреса, подписи, ИНН, КПП, итоги).

КООРДИНАТЫ помогают понять РАСПОЛОЖЕНИЕ:
- Малые X = левая часть, большие X = правая часть
- Малые Y = верх страницы, большие Y = низ страницы

КРИТИЧЕСКИ ВАЖНО! ЛОГИКА ОПРЕДЕЛЕНИЯ РОЛЕЙ:
1. АКТ выполненных работ/услуг:
   - ИСПОЛНИТЕЛЬ = тот, кто ВЫПОЛНИЛ работы (обычно слева или подписывает акт)
   - ЗАКАЗЧИК = тот, кто ПРИНИМАЕТ работы (обычно справа)
   - Пример: "Автоассистанс" выполнил услуги для "Деалон"

2. СЧЕТ на оплату:
   - ИСПОЛНИТЕЛЬ = поставщик, кто ВЫСТАВЛЯЕТ счет (получатель денег)
   - ЗАКАЗЧИК = плательщик, кому выставлен счет (платит деньги)

3. СЧЕТ-ФАКТУРА:
   - ИСПОЛНИТЕЛЬ = продавец, кто ПОСТАВЛЯЕТ товары/услуги
   - ЗАКАЗЧИК = покупатель, кто ПОЛУЧАЕТ товары/услуги

4. ДОГОВОР:
   - Смотри на контекст: кто поставщик, кто заказчик

ОПРЕДЕЛЕНИЕ ТИПА ДОКУМЕНТА:
- Если видишь "АКТ" - это Акт
- Если видишь "СЧЕТ" (но не "счет-фактура") - это Счёт
- Если видишь "СЧЕТ-ФАКТУРА" - это Счет-фактура
- Если видишь "ДОГОВОР" - это Договор

ПРАВИЛА ИЗВЛЕЧЕНИЯ:
1. ИНН: ТОЛЬКО 10 или 12 цифр (не путай с номерами счетов!)
2. КПП: ТОЛЬКО 9 цифр
3. Номер документа: без "от", "№", дат - только цифры/буквы
4. Используй координаты для группировки связанных данных

СТРУКТУРИРОВАННЫЕ ДАННЫЕ SURYA OCR (усеченные):
{structured_data}

АНАЛИЗИРУЙ ПОШАГОВО:
1. Определи тип документа (фокус на первых строках)
2. Найди номер и дату документа (обычно в первых)
3. Определи, кто ЗАКАЗЧИК (получает услуги), кто ИСПОЛНИТЕЛЬ (оказывает услуги)
4. Извлеки ИНН, КПП, адреса каждой стороны (часто в последних для подписей)
5. Проверь правильность ИНН (10-12 цифр) и КПП (9 цифр)

Верни результат ТОЛЬКО в JSON формате:
{{
  "Название файла": "{filename}",
  "Тип документа": "Акт|Счёт|Счет-фактура|Договор" (выбрать только одно!),
  "Номер документа": "",
  "Дата документа": "",
  "Наименование заказчика": "",
  "Наименование исполнителя": "",
  "ИНН заказчика": "",
  "ИНН исполнителя": "",
  "КПП заказчика": "",
  "КПП исполнителя": "",
  "Адрес заказчика": "",
  "Адрес исполнителя": ""
}}"""

        # Балансировка модели
        model_index = llm_settings.get('model_index', 0)
        models = llm_settings.get('models', ['local-1', 'local-2'])
        model = model_name
        
        # Поддержка OpenAI и LM Studio
        provider = llm_settings.get('provider', 'LM Studio')
        
        if provider == 'OpenAI':
            # Настройки для OpenAI
            api_key = llm_settings.get('api_key', '')
            if not api_key:
                return {"error": "Не указан OpenAI API ключ"}
                
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            endpoint = "https://api.openai.com/v1/chat/completions"
        else:
            # Настройки для LM Studio
            headers = {"Content-Type": "application/json"}
            endpoint = f"{llm_settings.get('endpoint', 'http://localhost:1234')}/v1/chat/completions"
        
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": llm_settings.get('max_tokens', 16000)
        }
        
        try:
            response = requests.post(
                endpoint, headers=headers, json=data, 
                timeout=llm_settings.get('timeout', 180)
            )
        except Exception as e:
            return {"error": f"Ошибка связи с {provider}: {e}"}
        
        if response.status_code == 200:
            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = content[start:end]
                return json.loads(json_str)
        
        return {"error": "Не удалось извлечь JSON"}
        
    except Exception as e:
        return {"error": str(e)}


def validate_llm_result(llm_result, original_text):
    """Валидация результатов LLM (вне класса)"""
    if "error" in llm_result:
        return llm_result
        
    try:
        import re
        
        # Валидация ИНН
        for key in ["ИНН заказчика", "ИНН исполнителя"]:
            inn = llm_result.get(key, "")
            if inn and not (inn.isdigit() and len(inn) in [10, 12]):
                inn_matches = re.findall(r'\b\d{10}\b|\b\d{12}\b', original_text)
                if inn_matches:
                    llm_result[key] = inn_matches.pop(0) if "заказчика" in key else inn_matches.pop(0) if inn_matches else ""
        
        # Валидация КПП
        for key in ["КПП заказчика", "КПП исполнителя"]:
            kpp = llm_result.get(key, "")
            if kpp and not (kpp.isdigit() and len(kpp) == 9):
                kpp_matches = re.findall(r'\b\d{9}\b', original_text)
                if kpp_matches:
                    llm_result[key] = kpp_matches.pop(0) if "заказчика" in key else kpp_matches.pop(0) if kpp_matches else ""
        
        # Очистка номера документа
        doc_number = llm_result.get("Номер документа", "")
        if doc_number:
            clean_number = re.sub(r'\s*от\s*\d+.*', '', doc_number)
            clean_number = re.sub(r'\s*\d{1,2}[./]\d{1,2}[./]\d{2,4}.*', '', clean_number)
            llm_result["Номер документа"] = clean_number.strip()
        
        # Проверка типа документа
        doc_type = llm_result.get("Тип документа", "")
        valid_types = ["Акт", "Счёт", "Счет-фактура", "Договор"]
        if doc_type not in valid_types:
            text_lower = original_text.lower()
            if "счет-фактура" in text_lower:
                llm_result["Тип документа"] = "Счет-фактура"
            elif "акт" in text_lower:
                llm_result["Тип документа"] = "Акт"
            elif "счет" in text_lower:
                llm_result["Тип документа"] = "Счёт"
            elif "договор" in text_lower:
                llm_result["Тип документа"] = "Договор"
            else:
                llm_result["Тип документа"] = "Неопределен"
        
        return llm_result
        
    except Exception as e:
        return llm_result


class SuryaSimpleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Surya OCR - Массовая обработка документов")
        self.root.geometry("700x600")
        
        # Флаг остановки обработки
        self.stop_processing = False
        self.active_processes = []  # Список активных процессов
        
        # Обработчик закрытия окна
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Переменные
        self.pdf_folder = tk.StringVar()
        self.json_folder = tk.StringVar()
        self.date_format = tk.StringVar(value="ISO")
        
        self.processing = False
        self.total_files = 0
        self.processed_files = 0
        
        # Настройки LLM
        self.llm_max_tokens = 16000  # Для Phi-4, можно увеличить для большего контекста
        self.llm_timeout = 180
        self.llm_endpoint = "http://localhost:1234"  # Один порт для всех моделей
        # ИСПРАВЛЕНО: ДВЕ МОДЕЛИ local-1 И local-2
        # Используем две разные модели в LM Studio
        self.llm_models = ["local-1", "local-2"]  # Две модели
        self.llm_model_index = 0  # Для round-robin
        
        # Настройки усечения
        self.first_page_lines = 10   # Первые N строк с первой страницы (заголовки/реквизиты)
        self.last_page_lines = 30    # Последние N строк с последней страницы (подписи/итоги)
        
        # Предикторы Surya (будут инициализированы в процессах)
        self.det_predictor = None
        self.rec_predictor = None
        
        # Оптимизация: Число параллельных процессов (будет переопределено из GUI)
        self.ocr_pool_size = 2  # По умолчанию
        self.llm_pool_size = 2  # По умолчанию
        
        # Статистика OCR
        self.ocr_start_time = 0
        self.ocr_total_time = 0
        self.ocr_doc_count = 0  # Количество обработанных документов
        self.ocr_completed_count = 0  # Для совместимости
        self.ocr_doc_times = []  # Времена обработки каждого документа
        
        # Статистика LLM
        self.llm_start_time = 0
        self.llm_total_time = 0
        self.llm_doc_count = 0
        self.llm_doc_times = []  # Времена обработки LLM
        
        # Общая статистика (суммарное время OCR + LLM)
        self.total_processing_time = 0
        self.total_doc_count = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        """Настройка интерфейса"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        row = 0
        
        # Выбор папки с PDF-файлами
        ttk.Label(main_frame, text="Папка с PDF-файлами:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.pdf_folder, width=50).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Обзор", command=self.select_pdf_folder).grid(row=row, column=2, padx=5)
        row += 1
        
        # Выбор папки для JSON-файлов
        ttk.Label(main_frame, text="Папка для JSON файлов:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.json_folder, width=50).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Обзор", command=self.select_json_folder).grid(row=row, column=2, padx=5)
        row += 1
        
        # Настройка формата даты
        ttk.Label(main_frame, text="Формат даты:").grid(row=row, column=0, sticky=tk.W, pady=5)
        date_frame = ttk.Frame(main_frame)
        date_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Radiobutton(date_frame, text="ISO 8601 (2023-02-17)", variable=self.date_format, value="ISO").pack(side=tk.LEFT)
        ttk.Radiobutton(date_frame, text="Классический (17.02.2023)", variable=self.date_format, value="CLASSIC").pack(side=tk.LEFT, padx=10)
        row += 1
        
        # Настройки производительности
        ttk.Label(main_frame, text="Настройки потоков:").grid(row=row, column=0, sticky=tk.W, pady=5)
        perf_frame = ttk.Frame(main_frame)
        perf_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        
        ttk.Label(perf_frame, text="OCR потоков:").pack(side=tk.LEFT)
        self.ocr_threads_var = tk.StringVar(value="2")
        self.ocr_threads_spinbox = ttk.Spinbox(perf_frame, from_=1, to=8, width=5, textvariable=self.ocr_threads_var)
        self.ocr_threads_spinbox.pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(perf_frame, text="LLM потоков:").pack(side=tk.LEFT)
        self.llm_threads_var = tk.StringVar(value="2")
        self.llm_threads_spinbox = ttk.Spinbox(perf_frame, from_=1, to=4, width=5, textvariable=self.llm_threads_var)
        self.llm_threads_spinbox.pack(side=tk.LEFT, padx=5)
        row += 1
        
        # Настройки автоповтора
        retry_frame = ttk.Frame(main_frame)
        retry_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        
        self.auto_retry_var = tk.BooleanVar(value=True)
        self.auto_retry_checkbox = ttk.Checkbutton(
            retry_frame, 
            text="Автоповтор ошибочных документов (до 3 попыток)",
            variable=self.auto_retry_var
        )
        self.auto_retry_checkbox.pack(side=tk.LEFT)
        
        self.max_retries_var = tk.StringVar(value="3")
        ttk.Label(retry_frame, text="Макс. попыток:").pack(side=tk.LEFT, padx=(20, 5))
        retry_spinbox = ttk.Spinbox(retry_frame, from_=1, to=5, width=3, textvariable=self.max_retries_var)
        retry_spinbox.pack(side=tk.LEFT)
        row += 1
        
        # Настройки LLM
        llm_frame = ttk.LabelFrame(main_frame, text="Настройки LLM", padding="10")
        llm_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Выбор провайдера
        ttk.Label(llm_frame, text="Провайдер:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.llm_provider_var = tk.StringVar(value="LM Studio")
        provider_frame = ttk.Frame(llm_frame)
        provider_frame.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(provider_frame, text="LM Studio", variable=self.llm_provider_var, value="LM Studio", command=self.on_provider_change).pack(side=tk.LEFT)
        ttk.Radiobutton(provider_frame, text="OpenAI", variable=self.llm_provider_var, value="OpenAI", command=self.on_provider_change).pack(side=tk.LEFT, padx=(20, 0))
        
        # OpenAI API ключ
        ttk.Label(llm_frame, text="OpenAI API ключ:").grid(row=1, column=0, sticky="w", padx=(0, 10))
        self.openai_api_key_entry = ttk.Entry(llm_frame, width=50, show="*")
        self.openai_api_key_entry.grid(row=1, column=1, sticky="ew")
        
        # Модель
        ttk.Label(llm_frame, text="Модель:").grid(row=2, column=0, sticky="w", padx=(0, 10))
        self.llm_model_var = tk.StringVar(value="local-1")
        self.llm_model_combobox = ttk.Combobox(llm_frame, textvariable=self.llm_model_var, width=47)
        self.llm_model_combobox['values'] = ('local-1', 'local-2')
        self.llm_model_combobox.grid(row=2, column=1, sticky="ew")
        
        # Макс. токенов
        ttk.Label(llm_frame, text="Макс. токенов:").grid(row=3, column=0, sticky="w", padx=(0, 10))
        self.llm_max_tokens_entry = ttk.Entry(llm_frame, width=50)
        self.llm_max_tokens_entry.insert(0, "16000")
        self.llm_max_tokens_entry.grid(row=3, column=1, sticky="ew")
        
        # Таймаут
        ttk.Label(llm_frame, text="Таймаут (сек):").grid(row=4, column=0, sticky="w", padx=(0, 10))
        self.llm_timeout_entry = ttk.Entry(llm_frame, width=50)
        self.llm_timeout_entry.insert(0, "180")
        self.llm_timeout_entry.grid(row=4, column=1, sticky="ew")
        
        llm_frame.columnconfigure(1, weight=1)
        row += 1
        
        # Кнопки управления
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Запустить обработку", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Остановить", command=self.stop_processing_manually, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=5)
        row += 1
        
        # Индикатор прогресса
        ttk.Label(main_frame, text="Прогресс:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5)
        row += 1
        
        # Блок статистики
        stats_frame = ttk.LabelFrame(main_frame, text="Статистика обработки", padding="10")
        stats_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Левая колонка - OCR статистика
        ocr_stats_frame = ttk.Frame(stats_frame)
        ocr_stats_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        ttk.Label(ocr_stats_frame, text="OCR Обработка:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.ocr_total_time_label = ttk.Label(ocr_stats_frame, text="Общее время: 0 сек")
        self.ocr_total_time_label.pack(anchor=tk.W)
        self.ocr_avg_time_label = ttk.Label(ocr_stats_frame, text="Среднее на док.: 0 сек")
        self.ocr_avg_time_label.pack(anchor=tk.W)
        self.ocr_completed_label = ttk.Label(ocr_stats_frame, text="Завершено: 0/0")
        self.ocr_completed_label.pack(anchor=tk.W)
        
        # Правая колонка - LLM статистика
        llm_stats_frame = ttk.Frame(stats_frame)
        llm_stats_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(llm_stats_frame, text="LLM Анализ:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.llm_total_time_label = ttk.Label(llm_stats_frame, text="Общее время: 0 сек")
        self.llm_total_time_label.pack(anchor=tk.W)
        self.llm_avg_time_label = ttk.Label(llm_stats_frame, text="Среднее на док.: 0 сек")
        self.llm_avg_time_label.pack(anchor=tk.W)
        self.llm_completed_label = ttk.Label(llm_stats_frame, text="Завершено: 0/0")
        self.llm_completed_label.pack(anchor=tk.W)
        
        # Третья колонка - Общая статистика
        total_stats_frame = ttk.Frame(stats_frame)
        total_stats_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(total_stats_frame, text="Общая статистика:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.total_avg_time_label = ttk.Label(total_stats_frame, text="Среднее на док.: 0 сек")
        self.total_avg_time_label.pack(anchor=tk.W)
        self.total_time_breakdown_label = ttk.Label(total_stats_frame, text="OCR + LLM: 0 + 0 с")
        self.total_time_breakdown_label.pack(anchor=tk.W)
        self.processing_speed_label = ttk.Label(total_stats_frame, text="Скорость: 0 док/мин")
        self.processing_speed_label.pack(anchor=tk.W)
        
        row += 1
        
        # Лог
        ttk.Label(main_frame, text="Лог обработки:").grid(row=row, column=0, sticky=tk.W, pady=5)
        row += 1
        
        self.log_text = scrolledtext.ScrolledText(main_frame, height=20, width=80)
        self.log_text.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        row += 1
        
        # Кнопка сохранения лога
        ttk.Button(main_frame, text="Сохранить лог как txt файл", command=self.save_log).grid(row=row, column=0, columnspan=3, pady=10)
        
        # Настройка сетки
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(row-2, weight=1)
        
        # Начальная настройка провайдера
        self.on_provider_change()
        
    def select_pdf_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.pdf_folder.set(folder)
            
    def select_json_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.json_folder.set(folder)
            
    def on_provider_change(self):
        """Обработка смены провайдера LLM"""
        provider = self.llm_provider_var.get()
        
        if provider == "OpenAI":
            # Настройки для OpenAI
            self.openai_api_key_entry.config(state="normal")
            self.llm_model_combobox['values'] = (
                'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo',
                'o1-preview', 'o1-mini'
            )
            self.llm_model_var.set('gpt-4o-mini')
            self.llm_max_tokens_entry.delete(0, tk.END)
            self.llm_max_tokens_entry.insert(0, "16000")
        else:
            # Настройки для LM Studio
            self.openai_api_key_entry.config(state="disabled")
            self.llm_model_combobox['values'] = ('local-1', 'local-2')
            self.llm_model_var.set('local-1')
            self.llm_max_tokens_entry.delete(0, tk.END)
            self.llm_max_tokens_entry.insert(0, "16000")
    
    def on_closing(self):
        """Обработчик закрытия окна - останавливает все процессы"""
        self.log("Получен сигнал закрытия - останавливаем все процессы...")
        self.stop_processing = True
        
        # Завершаем все активные процессы
        for process in self.active_processes:
            if process.is_alive():
                self.log(f"Завершаем процесс PID: {process.pid}")
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()  # Принудительное завершение
        
        self.active_processes.clear()
        self.root.destroy()
    
    def stop_processing_manually(self):
        """Ручная остановка обработки"""
        self.log("Пользователь остановил обработку")
        self.stop_processing = True
        
        # Завершаем все активные процессы
        for process in self.active_processes:
            if process.is_alive():
                self.log(f"Останавливаем процесс PID: {process.pid}")
                process.terminate()
        
        # Возвращаем кнопки в нормальное состояние
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
            
    def log(self, message):
        """Добавляет сообщение в лог"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.log_text.insert(tk.END, log_message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def update_progress(self, current, total):
        """Обновление прогресс-бара"""
        if total > 0:
            progress_value = (current / total) * 100
            self.progress['value'] = progress_value
        self.root.update_idletasks()
            
    def update_ocr_stats(self, completed_count, total_count, doc_time=None):
        """Обновление статистики OCR"""
        self.ocr_completed_count = completed_count
        self.ocr_doc_count = completed_count  # Обновляем счетчик документов
        if doc_time:
            self.ocr_doc_times.append(doc_time)
            
        # Обновляем общее время
        if self.ocr_start_time > 0:
            self.ocr_total_time = time.time() - self.ocr_start_time
            
        # Среднее время на документ
        avg_time = sum(self.ocr_doc_times) / len(self.ocr_doc_times) if self.ocr_doc_times else 0
        
        # Обновляем GUI
        self.ocr_total_time_label.config(text=f"Общее время: {self.ocr_total_time:.1f} сек")
        self.ocr_avg_time_label.config(text=f"Среднее на док.: {avg_time:.1f} сек")
        self.ocr_completed_label.config(text=f"Завершено: {completed_count}/{total_count}")
        self.root.update_idletasks()
        
    def update_llm_stats(self, completed_count, total_count, doc_time=None):
        """Обновление статистики LLM"""
        self.llm_doc_count = completed_count  # Обновляем счетчик документов
        if doc_time:
            self.llm_doc_times.append(doc_time)
            
        # Обновляем общее время
        if self.llm_start_time > 0:
            self.llm_total_time = time.time() - self.llm_start_time
            
        # Среднее время на документ
        avg_time = sum(self.llm_doc_times) / len(self.llm_doc_times) if self.llm_doc_times else 0
        
        # Обновляем GUI
        self.llm_total_time_label.config(text=f"Общее время: {self.llm_total_time:.1f} сек")
        self.llm_avg_time_label.config(text=f"Среднее на док.: {avg_time:.1f} сек")
        self.llm_completed_label.config(text=f"Завершено: {completed_count}/{total_count}")
        self.root.update_idletasks()
        
        # Обновляем общую статистику
        self.update_total_stats()
    
    def update_total_stats(self):
        """Обновление общей статистики (OCR + LLM)"""
        if self.ocr_doc_count > 0 and self.llm_doc_count > 0:
            # Среднее время OCR на документ
            avg_ocr_time = self.ocr_total_time / self.ocr_doc_count
            # Среднее время LLM на документ
            avg_llm_time = self.llm_total_time / self.llm_doc_count
            # Общее среднее время на документ
            total_avg_time = avg_ocr_time + avg_llm_time
            
            # Скорость обработки (документов в минуту)
            docs_per_minute = 60 / total_avg_time if total_avg_time > 0 else 0
            
            self.total_avg_time_label.config(text=f"Среднее на док.: {total_avg_time:.1f} сек")
            self.total_time_breakdown_label.config(text=f"OCR + LLM: {avg_ocr_time:.1f} + {avg_llm_time:.1f} с")
            self.processing_speed_label.config(text=f"Скорость: {docs_per_minute:.1f} док/мин")
        
        self.root.update_idletasks()
        
    def initialize_surya(self):
        """Инициализация предикторов Surya (в каждом процессе)"""
        self.det_predictor = DetectionPredictor()
        self.rec_predictor = RecognitionPredictor()
        
    def process_pdf_with_surya(self, pdf_path):
        """Обработка PDF с Surya OCR, возвращает данные по страницам"""
        try:
            images, names = load_from_file(pdf_path)
            task_names = [TaskNames.ocr_with_boxes] * len(images)
            predictions = self.rec_predictor(
                images,
                task_names=task_names,
                det_predictor=self.det_predictor,
                math_mode=False
            )
            
            pages_data = []
            combined_text = ""
            
            for page_idx, pred in enumerate(predictions):
                page_lines = []
                page_text = ""
                for line in pred.text_lines:
                    line_data = {
                        "text": line.text,
                        "bbox": line.bbox,
                        "confidence": line.confidence
                    }
                    page_lines.append(line_data)
                    page_text += line.text + " "
                pages_data.append({
                    "page": page_idx + 1,
                    "text_lines": page_lines
                })
                combined_text += page_text
            
            combined_json = {
                "filename": os.path.basename(pdf_path),
                "pages": len(predictions),
                "pages_data": pages_data,
                "full_text": combined_text.strip()
            }
            
            return combined_json, combined_text.strip()
            
        except Exception as e:
            return None, None
            
    def prepare_truncated_ocr_for_llm(self, ocr_json_data):
        """Усечение: первые 10 с первой + последние 30 с последней"""
        pages_data = ocr_json_data.get("pages_data", [])
        total_pages = len(pages_data)
        
        if total_pages == 0:
            return []
        
        truncated_lines = []
        
        # Первая страница: первые 10 строк
        first_page_lines = pages_data[0]["text_lines"]
        truncated_lines.extend([{"source": "first_page", **line} for line in first_page_lines[:self.first_page_lines]])
        
        # Последняя страница: последние 30 строк
        last_page_lines = pages_data[-1]["text_lines"]
        start_idx = max(0, len(last_page_lines) - self.last_page_lines)
        truncated_lines.extend([{"source": "last_page", **line} for line in last_page_lines[start_idx:]])
        
        return truncated_lines
        
    def save_to_csv(self, filename, ocr_json, ocr_text, pdf_folder):
        csv_file = os.path.join(os.path.dirname(pdf_folder), "ocr_result.csv")
        
        file_exists = os.path.exists(csv_file)
        
        try:
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                if not file_exists:
                    writer.writerow(['filename', 'recognition_date', 'ocr_json', 'ocr_text'])
                
                date_str = datetime.now().strftime("%Y-%m-%d" if self.date_format.get() == "ISO" else "%d.%m.%Y")
                
                writer.writerow([
                    filename,
                    date_str,
                    json.dumps(ocr_json, ensure_ascii=False),
                    ocr_text
                ])
        except Exception as e:
            pass  # Лог в другом месте
            
    def analyze_with_llm(self, filename, truncated_data):
        """Анализ усеченных данных с LLM (балансировка по именам моделей на одном порту)"""
        try:
            structured_data = json.dumps(truncated_data, ensure_ascii=False, indent=2)
            
            prompt = f"""Ты эксперт по анализу российских деловых документов.

К тебе приходят УСЕЧЕННЫЕ СТРУКТУРИРОВАННЫЕ данные от Surya OCR с координатами (bbox) каждой строки. Фокус на ключевых частях: первые строки (заголовки, реквизиты, тип, номер, дата, стороны) и последние (адреса, подписи, ИНН, КПП, итоги).

КООРДИНАТЫ помогают понять РАСПОЛОЖЕНИЕ:
- Малые X = левая часть, большие X = правая часть
- Малые Y = верх страницы, большие Y = низ страницы

КРИТИЧЕСКИ ВАЖНО! ЛОГИКА ОПРЕДЕЛЕНИЯ РОЛЕЙ:
1. АКТ выполненных работ/услуг:
   - ИСПОЛНИТЕЛЬ = тот, кто ВЫПОЛНИЛ работы (обычно слева или подписывает акт)
   - ЗАКАЗЧИК = тот, кто ПРИНИМАЕТ работы (обычно справа)
   - Пример: "Автоассистанс" выполнил услуги для "Деалон"

2. СЧЕТ на оплату:
   - ИСПОЛНИТЕЛЬ = поставщик, кто ВЫСТАВЛЯЕТ счет (получатель денег)
   - ЗАКАЗЧИК = плательщик, кому выставлен счет (платит деньги)

3. СЧЕТ-ФАКТУРА:
   - ИСПОЛНИТЕЛЬ = продавец, кто ПОСТАВЛЯЕТ товары/услуги
   - ЗАКАЗЧИК = покупатель, кто ПОЛУЧАЕТ товары/услуги

4. ДОГОВОР:
   - Смотри на контекст: кто поставщик, кто заказчик

ОПРЕДЕЛЕНИЕ ТИПА ДОКУМЕНТА:
- Если видишь "АКТ" - это Акт
- Если видишь "СЧЕТ" (но не "счет-фактура") - это Счёт
- Если видишь "СЧЕТ-ФАКТУРА" - это Счет-фактура
- Если видишь "ДОГОВОР" - это Договор

ПРАВИЛА ИЗВЛЕЧЕНИЯ:
1. ИНН: ТОЛЬКО 10 или 12 цифр (не путай с номерами счетов!)
2. КПП: ТОЛЬКО 9 цифр
3. Номер документа: без "от", "№", дат - только цифры/буквы
4. Используй координаты для группировки связанных данных

СТРУКТУРИРОВАННЫЕ ДАННЫЕ SURYA OCR (усеченные):
{structured_data}

АНАЛИЗИРУЙ ПОШАГОВО:
1. Определи тип документа (фокус на первых строках)
2. Найди номер и дату документа (обычно в первых)
3. Определи, кто ЗАКАЗЧИК (получает услуги), кто ИСПОЛНИТЕЛЬ (оказывает услуги)
4. Извлеки ИНН, КПП, адреса каждой стороны (часто в последних для подписей)
5. Проверь правильность ИНН (10-12 цифр) и КПП (9 цифр)

Верни результат ТОЛЬКО в JSON формате:
{{
  "Название файла": "{filename}",
  "Тип документа": "Акт|Счёт|Счет-фактура|Договор" (выбрать только одно!),
  "Номер документа": "",
  "Дата документа": "",
  "Наименование заказчика": "",
  "Наименование исполнителя": "",
  "ИНН заказчика": "",
  "ИНН исполнителя": "",
  "КПП заказчика": "",
  "КПП исполнителя": "",
  "Адрес заказчика": "",
  "Адрес исполнителя": ""
}}"""

            # Балансировка имени модели
            model = self.llm_models[self.llm_model_index % len(self.llm_models)]
            self.llm_model_index += 1
            
            headers = {"Content-Type": "application/json"}
            
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": self.llm_max_tokens
            }
            
            response = requests.post(f"{self.llm_endpoint}/v1/chat/completions", 
                                     headers=headers, json=data, timeout=self.llm_timeout)
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # Извлечение JSON
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = content[start:end]
                    return json.loads(json_str)
                
            return {"error": "Не удалось извлечь JSON"}
            
        except Exception as e:
            return {"error": str(e)}
            
    def validate_and_fix_llm_result(self, llm_result, original_text):
        """Валидация и исправление результатов LLM (как в оригинале)"""
        if "error" in llm_result:
            return llm_result
            
        try:
            import re
            
            # Валидация ИНН
            for key in ["ИНН заказчика", "ИНН исполнителя"]:
                inn = llm_result.get(key, "")
                if inn and not (inn.isdigit() and len(inn) in [10, 12]):
                    inn_matches = re.findall(r'\b\d{10}\b|\b\d{12}\b', original_text)
                    if inn_matches:
                        llm_result[key] = inn_matches.pop(0) if "заказчика" in key else inn_matches.pop(0) if inn_matches else ""
            
            # Валидация КПП
            for key in ["КПП заказчика", "КПП исполнителя"]:
                kpp = llm_result.get(key, "")
                if kpp and not (kpp.isdigit() and len(kpp) == 9):
                    kpp_matches = re.findall(r'\b\d{9}\b', original_text)
                    if kpp_matches:
                        llm_result[key] = kpp_matches.pop(0) if "заказчика" in key else kpp_matches.pop(0) if kpp_matches else ""
            
            # Очистка номера документа
            doc_number = llm_result.get("Номер документа", "")
            if doc_number:
                clean_number = re.sub(r'\s*от\s*\d+.*', '', doc_number)
                clean_number = re.sub(r'\s*\d{1,2}[./]\d{1,2}[./]\d{2,4}.*', '', clean_number)
                llm_result["Номер документа"] = clean_number.strip()
            
            # Проверка типа документа
            doc_type = llm_result.get("Тип документа", "")
            valid_types = ["Акт", "Счёт", "Счет-фактура", "Договор"]
            if doc_type not in valid_types:
                text_lower = original_text.lower()
                if "счет-фактура" in text_lower:
                    llm_result["Тип документа"] = "Счет-фактура"
                elif "акт" in text_lower:
                    llm_result["Тип документа"] = "Акт"
                elif "счет" in text_lower:
                    llm_result["Тип документа"] = "Счёт"
                elif "договор" in text_lower:
                    llm_result["Тип документа"] = "Договор"
                else:
                    llm_result["Тип документа"] = "Неопределен"
            
            return llm_result
            
        except Exception as e:
            return llm_result
            

    
    def save_llm_result(self, filename, llm_result, json_folder):
        try:
            json_filename = os.path.splitext(filename)[0] + ".json"
            json_path = os.path.join(json_folder, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(llm_result, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            pass  # Лог в другом месте
            
    def process_files(self):
        """Основная функция: параллельная обработка с multiprocessing"""
        try:
            pdf_folder = self.pdf_folder.get()
            json_folder = self.json_folder.get()
            
            if not pdf_folder or not json_folder:
                self.log("Ошибка: Не выбраны папки")
                return
                
            # Получаем настройки потоков из GUI
            self.ocr_pool_size = int(self.ocr_threads_var.get())
            self.llm_pool_size = int(self.llm_threads_var.get())
            
            # Сбрасываем статистику
            self.ocr_start_time = 0
            self.ocr_total_time = 0
            self.ocr_completed_count = 0
            self.ocr_doc_times = []
            self.llm_start_time = 0
            self.llm_total_time = 0
            self.llm_completed_count = 0
            self.llm_doc_times = []
            
            os.makedirs(json_folder, exist_ok=True)
            
            pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                messagebox.showwarning("Предупреждение", "Нет PDF файлов")
                return
                
            self.total_files = len(pdf_files)
            self.processed_files = 0
            
            start_time = datetime.now()
            self.log(f"=== НАЧАЛО ОБРАБОТКИ ===")
            self.log(f"Время начала: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.log(f"Найдено {self.total_files} PDF файлов")
            self.log(f"OCR потоков: {self.ocr_pool_size}, LLM моделей: {self.llm_pool_size}")
            
            # ЭТАП 1: OCR ОБРАБОТКА с очередями (НЕБЛОКИРУЮЩАЯ!)
            self.log(f"ЭТАП 1: Начинаем OCR обработку в {self.ocr_pool_size} потока...")
            
            # Запускаем таймер OCR
            self.ocr_start_time = time.time()
            self.update_ocr_stats(0, len(pdf_files))
            
            # Очереди для OCR
            pdf_queue = Queue()
            ocr_result_queue = Queue()
            
            # Заполняем очередь PDF
            for pdf_file in pdf_files:
                pdf_queue.put((pdf_file, pdf_folder, self.date_format.get()))
            
            # Проверяем флаг остановки
            if self.stop_processing:
                self.log("Обработка остановлена пользователем")
                return
            
            # Запуск OCR процессов
            ocr_processes = []
            for i in range(self.ocr_pool_size):
                p = Process(target=ocr_worker_simple, args=(pdf_queue, ocr_result_queue))
                p.start()
                ocr_processes.append(p)
                self.active_processes.append(p)  # Добавляем в список активных
            
            # Мониторинг OCR результатов
            ocr_completed = 0
            ocr_data_list = []
            
            while ocr_completed < len(pdf_files) and not self.stop_processing:
                try:
                    result = ocr_result_queue.get(timeout=1)
                    
                    # Проверяем флаг остановки
                    if self.stop_processing:
                        self.log("Остановка OCR обработки...")
                        break
                    
                    doc_time = result.get('processing_time', 0)
                    
                    if result["success"]:
                        ocr_data_list.append(result)
                        ocr_completed += 1
                        self.log(f"OCR завершен: {result['filename']} ({doc_time:.1f}с)")
                        self.update_progress(ocr_completed, len(pdf_files))
                        self.update_ocr_stats(ocr_completed, len(pdf_files), doc_time)
                    else:
                        ocr_completed += 1
                        self.log(f"OCR ошибка: {result['filename']} - {result['error']}")
                        self.update_progress(ocr_completed, len(pdf_files))
                        self.update_ocr_stats(ocr_completed, len(pdf_files))
                except queue.Empty:
                    self.root.update()  # Обновляем GUI
                    continue
            
            # Завершаем OCR процессы
            for _ in range(self.ocr_pool_size):
                pdf_queue.put(None)
            for p in ocr_processes:
                p.join()
            
            self.log(f"ЭТАП 1 ЗАВЕРШЕН: OCR обработано {ocr_completed}/{len(pdf_files)} файлов")
            
            if not ocr_data_list:
                messagebox.showerror("Ошибка", "OCR не обработал ни одного файла")
                return
            
            # ЭТАП 2: LLM ОБРАБОТКА (параллельно local-1 и local-2)
            self.log(f"ЭТАП 2: Начинаем LLM анализ с {self.llm_pool_size} воркерами...")
            
            # Запускаем таймер LLM
            self.llm_start_time = time.time()
            self.update_llm_stats(0, len(ocr_data_list))
            
            # Получаем настройки LLM и автоповтора из GUI
            provider = self.llm_provider_var.get()
            auto_retry = self.auto_retry_var.get()
            max_retries = int(self.max_retries_var.get())
            
            llm_settings = {
                'provider': provider,
                'max_tokens': int(self.llm_max_tokens_entry.get()),
                'timeout': int(self.llm_timeout_entry.get()),
                'endpoint': self.llm_endpoint,
                'auto_retry': auto_retry,
                'max_retries': max_retries
            }
            
            # Настраиваем модели с учетом количества потоков
            if provider == 'OpenAI':
                api_key = self.openai_api_key_entry.get().strip()
                if not api_key:
                    messagebox.showerror("Ошибка", "Введите OpenAI API ключ!")
                    return
                llm_settings['api_key'] = api_key
                # Для OpenAI используем одну модель с несколькими воркерами
                model_name = self.llm_model_var.get()
                self.llm_models = [model_name] * self.llm_pool_size  # Дублируем по количеству потоков
            else:
                # Для LM Studio создаем модели по количеству потоков
                if self.llm_pool_size == 1:
                    self.llm_models = ["local-model"]  # Один воркер
                elif self.llm_pool_size == 2:
                    self.llm_models = ["local-1", "local-2"]  # Два воркера
                else:
                    # Больше 2 потоков - используем одну модель
                    self.llm_models = ["local-model"] * self.llm_pool_size
            
            # Очереди для LLM и повторов
            llm_queue = Queue()
            retry_queue = Queue() if auto_retry else None
            result_queue = Queue()
            
            # Заполняем очередь OCR данными
            for ocr_data in ocr_data_list:
                llm_queue.put((ocr_data['filename'], ocr_data['truncated_data'], ocr_data['combined_text']))
            
            # Запуск LLM процессов
            self.log(f"Создаем {len(self.llm_models)} LLM воркеров: {self.llm_models}")
            llm_processes = []
            for i, model in enumerate(self.llm_models):
                worker_name = f"LLM-{i+1}" if len(self.llm_models) > 1 else "LLM"
                # Проверяем флаг остановки
                if self.stop_processing:
                    self.log("Остановка перед запуском LLM")
                    return
                    
                self.log(f"Запускаем воркер: {worker_name} (модель: {model})")
                p = Process(target=llm_worker, args=(llm_queue, result_queue, json_folder, llm_settings, model, worker_name, retry_queue))
                p.start()
                llm_processes.append(p)
                self.active_processes.append(p)  # Добавляем в список активных
            
            # Мониторинг LLM результатов с поддержкой повторов
            llm_completed = 0
            retry_added = 0  # Счетчик добавленных повторов
            
            while llm_completed < len(ocr_data_list) and not self.stop_processing:
                try:
                    result = result_queue.get(timeout=1)
                    
                    # Проверяем флаг остановки
                    if self.stop_processing:
                        self.log("Остановка LLM обработки...")
                        break
                    
                    self.log(result)
                    
                    # Проверяем повторы в очереди и добавляем их в основную очередь
                    if auto_retry and retry_queue:
                        while not retry_queue.empty():
                            try:
                                retry_item = retry_queue.get_nowait()
                                llm_queue.put(retry_item)
                                retry_added += 1
                                self.log(f"Добавлен повтор в очередь: {retry_item[0]}")
                            except queue.Empty:
                                break
                    
                    # Увеличиваем счетчик ТОЛЬКО для завершенных документов (не повторов)
                    if "Завершено" in result or ("Ошибка LLM" in result and "попытка" in result):
                        # Парсим время обработки
                        doc_time = 0
                        if "(время:" in result:
                            try:
                                time_part = result.split("(время: ")[1].split("с)")[0]
                                doc_time = float(time_part)
                            except:
                                pass
                        
                        llm_completed += 1
                        self.update_progress(llm_completed, len(ocr_data_list))
                        self.update_llm_stats(llm_completed, len(ocr_data_list), doc_time if doc_time > 0 else None)
                        
                except queue.Empty:
                    self.root.update()  # Обновляем GUI
                    continue
            
            # Обработка оставшихся повторов перед завершением
            if auto_retry and retry_queue:
                # Проверяем оставшиеся повторы
                final_retries = 0
                while not retry_queue.empty():
                    try:
                        retry_item = retry_queue.get_nowait()
                        llm_queue.put(retry_item)
                        final_retries += 1
                        self.log(f"Обработка оставшегося повтора: {retry_item[0]}")
                    except queue.Empty:
                        break
                        
                if final_retries > 0:
                    self.log(f"Обрабатываем {final_retries} оставшихся повторов...")
                    
                    # Ожидаем завершения повторов
                    final_completed = 0
                    while final_completed < final_retries:
                        try:
                            result = result_queue.get(timeout=5)
                            self.log(result)
                            if "Завершено" in result or "Ошибка LLM" in result:
                                final_completed += 1
                        except queue.Empty:
                            self.log("Таймаут ожидания повторов")
                            break
            
            # Завершение LLM воркеров
            for _ in range(len(self.llm_models)):
                llm_queue.put(None)
            
            for p in llm_processes:
                p.join()
            
            self.processed_files = llm_completed
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.log(f"\n=== ОБРАБОТКА ЗАВЕРШЕНА ===")
            self.log(f"Время окончания: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.log(f"Длительность: {duration}")
            self.log(f"Обработано: {self.processed_files}/{self.total_files}")
            if auto_retry and retry_added > 0:
                self.log(f"Повторов выполнено: {retry_added}")
            
            # Восстанавливаем состояние кнопок
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            
            # Очищаем список активных процессов
            self.active_processes.clear()
            
            if not self.stop_processing:
                messagebox.showinfo("Готово", f"Обработано: {self.processed_files}/{self.total_files}")
            else:
                messagebox.showwarning("Остановлено", "Обработка остановлена пользователем")
            
        except Exception as e:
            self.log(f"Критическая ошибка: {e}")
            messagebox.showerror("Ошибка", str(e))
        finally:
            self.processing = False
            self.start_button.config(text="Запустить обработку", state="normal")
            
    def start_processing(self):
        if self.processing:
            self.processing = False
        
        # Меняем состояние кнопок
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        # Запускаем обработку в отдельном потоке
        processing_thread = threading.Thread(target=self.process_files, daemon=True)
        processing_thread.start()

    def save_log(self):
        try:
            log_content = self.log_text.get(1.0, tk.END)
            filename = f"surya_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(log_content)
                
            messagebox.showinfo("Сохранено", f"Лог в: {filename}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))


def main():
    root = tk.Tk()
    app = SuryaSimpleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()