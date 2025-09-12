#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperOCR - Advanced OCR Application with Surya OCR 0.16.7 + CUDA
Optimized for maximum speed and GPU utilization
"""

import os
import sys
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading
import json
import platform
import queue
import csv  # Добавляем импорт модуля csv
from datetime import datetime
from multiprocessing import Process, Queue

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables BEFORE any other imports
os.environ['RECOGNITION_BATCH_SIZE'] = '64'
os.environ['DETECTOR_BATCH_SIZE'] = '16'
os.environ['TORCH_DEVICE'] = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Try to import pydantic, install if missing
def ensure_pydantic():
    """Ensure pydantic is available, install if missing"""
    try:
        import pydantic
        print(f"✅ Pydantic version: {pydantic.__version__}")
        return True
    except ImportError:
        print("⚠️ Pydantic not found, attempting to install...")
        try:
            import subprocess
            import sys
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "pydantic==2.11.7", "pydantic-settings==2.10.1"
            ])
            import pydantic
            print(f"✅ Pydantic installed and imported, version: {pydantic.__version__}")
            return True
        except Exception as e:
            print(f"❌ Failed to install pydantic: {e}")
            print("⚠️ Application may not work correctly without pydantic")
            return False

# Ensure pydantic is available
pydantic_available = ensure_pydantic()

# Check CUDA availability
try:
    import torch
    if torch.cuda.is_available():
        print(f"🚀 GPU Settings Applied: {torch.cuda.get_device_name(0)}")
        print(f"🚀 RECOGNITION_BATCH_SIZE={os.environ.get('RECOGNITION_BATCH_SIZE', '64')}")
        print(f"🚀 DETECTOR_BATCH_SIZE={os.environ.get('DETECTOR_BATCH_SIZE', '16')}")
        print(f"🚀 TORCH_DEVICE={os.environ.get('TORCH_DEVICE', 'cuda')}")
    else:
        print("⚠️ CUDA not available, but forcing GPU mode...")
except Exception as e:
    print(f"⚠️ Error checking CUDA availability: {e}")
    print("⚠️ Forcing GPU mode...")

# Дополнительные оптимизации
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Импорты Surya with error handling
SURYA_AVAILABLE = False
try:
    if pydantic_available:
        from surya.input.load import load_from_file
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
        from surya.foundation import FoundationPredictor
        from surya.common.surya.schema import TaskNames
        SURYA_AVAILABLE = True
        print("✅ Surya OCR imports successful")
    else:
        print("❌ Skipping Surya OCR imports due to missing pydantic")
except ImportError as e:
    print(f"❌ Failed to import Surya OCR: {e}")
    SURYA_AVAILABLE = False
except Exception as e:
    print(f"❌ Unexpected error importing Surya OCR: {e}")
    SURYA_AVAILABLE = False

# Подавляем предупреждения torch_dtype deprecated
import warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.* is deprecated.*")

# Импорт для подсчета токенов
try:
    from token_counter import smart_truncate_for_llm, check_context_limit
    print("✅ Token counter imports successful")
except ImportError as e:
    print(f"❌ Failed to import token_counter: {e}")
    # Create dummy functions to avoid crashes
    def smart_truncate_for_llm(*args, **kwargs):
        return [], 0, False
    
    def check_context_limit(*args, **kwargs):
        return False, 0

def handle_surya_error_optimized(pdf_path, det_predictor, rec_predictor, max_retries=2):
    """
    ОПТИМИЗИРОВАННАЯ обработка ошибок Surya OCR - БЕЗ пересоздания предикторов!
    Используем ГОТОВЫЕ предикторы для максимальной скорости
    """
    import torch
    
    filename = os.path.basename(pdf_path)
    
    # Обычные повторы (меньше попыток)
    for attempt in range(max_retries + 1):
        try:
            images, names = load_from_file(pdf_path)
            task_names = [TaskNames.ocr_with_boxes] * len(images)
            predictions = rec_predictor(
                images,
                task_names=task_names,
                det_predictor=det_predictor,
                math_mode=False
            )
            
            if attempt > 0:
                print(f"✅ Surya ошибка устранена для {filename} на попытке {attempt + 1}")
            
            return predictions  # Успешно
            
        except Exception as ocr_error:
            error_msg = str(ocr_error)
            
            # Легкая очистка GPU кэша (без пересоздания предикторов)
            if attempt < max_retries:
                print(f"⚠️ Surya ошибка для {filename} (попытка {attempt + 1}/{max_retries + 1}): {error_msg}")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Короткая пауза
                time.sleep(0.5)
                continue
            else:
                # Окончательная ошибка
                print(f"❌ Не удалось обработать {filename} после {max_retries + 1} попыток: {error_msg}")
                raise ocr_error

# Глобальный кеш предикторов для ускорения (критическая оптимизация)
_GLOBAL_PREDICTORS_CACHE = {
    'det_predictor': None,
    'foundation_predictor': None, 
    'rec_predictor': None,
    'initialized': False
}

def get_cached_predictors():
    """
    Получение кешированных предикторов Surya OCR
    КРИТИЧЕСКАЯ ОПТИМИЗАЦИЯ: создаем предикторы только один раз для всего приложения
    """
    global _GLOBAL_PREDICTORS_CACHE
    
    if not _GLOBAL_PREDICTORS_CACHE['initialized']:
        print("🚀 Инициализация Surya OCR предикторов (один раз)...")
        import time
        start_time = time.time()
        
        _GLOBAL_PREDICTORS_CACHE['det_predictor'] = DetectionPredictor()
        print(f"   ✅ DetectionPredictor: {time.time() - start_time:.1f}с")
        
        foundation_start = time.time()
        _GLOBAL_PREDICTORS_CACHE['foundation_predictor'] = FoundationPredictor()
        print(f"   ✅ FoundationPredictor: {time.time() - foundation_start:.1f}с")
        
        rec_start = time.time()
        _GLOBAL_PREDICTORS_CACHE['rec_predictor'] = RecognitionPredictor(_GLOBAL_PREDICTORS_CACHE['foundation_predictor'])
        print(f"   ✅ RecognitionPredictor: {time.time() - rec_start:.1f}с")
        
        _GLOBAL_PREDICTORS_CACHE['initialized'] = True
        print(f"🎯 Инициализация завершена за {time.time() - start_time:.1f}с")
    
    return (
        _GLOBAL_PREDICTORS_CACHE['det_predictor'],
        _GLOBAL_PREDICTORS_CACHE['rec_predictor']
    )

def handle_surya_error_and_retry(pdf_path, det_predictor, rec_predictor, max_retries=3):
    """
    Обработка ошибок Surya OCR с повторными попытками + Fallback по страницам
    Решает проблемы с '_seen_tokens' и 'ContinuousBatchingCache'
    """
    import torch
    
    filename = os.path.basename(pdf_path)
    
    # ЭТАП 1: Обычные повторы
    for attempt in range(max_retries + 1):
        try:
            images, names = load_from_file(pdf_path)
            task_names = [TaskNames.ocr_with_boxes] * len(images)
            predictions = rec_predictor(
                images,
                task_names=task_names,
                det_predictor=det_predictor,
                math_mode=False
            )
            
            # Образуем лог о результате
            if attempt > 0:
                print(f"✅ Surya ошибка устранена для {filename} на попытке {attempt + 1}")
            
            return predictions  # Успешно
            
        except Exception as ocr_error:
            error_msg = str(ocr_error)
            
            # Проверяем специфичные ошибки Surya
            if ("_seen_tokens" in error_msg or 
                "ContinuousBatchingCache" in error_msg or
                "AttributeError" in error_msg) and attempt < max_retries:
                
                print(f"⚠️ Surya ошибка для {filename} (попытка {attempt + 1}/{max_retries + 1}): {error_msg}")
                
                # Очистка GPU кэша
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Пересоздаем предикторы
                det_predictor = DetectionPredictor()
                foundation_predictor = FoundationPredictor()
                rec_predictor = RecognitionPredictor(foundation_predictor)
                
                # Пауза перед повтором (увеличиваем с каждой попыткой)
                pause_time = 1 + attempt * 0.5
                time.sleep(pause_time)
                continue
                
            else:
                # Не наша ошибка или исчерпаны попытки
                if attempt >= max_retries:
                    # ЭТАП 2: FALLBACK - попробуем обработать по страницам
                    print(f"🔄 Fallback для {filename}: попытка обработки по страницам...")
                    try:
                        return process_pdf_page_by_page(pdf_path, det_predictor, rec_predictor)
                    except Exception as fallback_error:
                        print(f"❌ Fallback тоже неудачен для {filename}: {str(fallback_error)}")
                        print(f"❌ Surya ошибка для {filename}: исчерпаны все методы - {error_msg}")
                raise ocr_error
    
    raise Exception(f"Не удалось обработать {filename} после {max_retries + 1} попыток")


def process_pdf_page_by_page(pdf_path, det_predictor, rec_predictor):
    """
    Fallback-метод: обработка PDF по одной странице для проблемных документов
    """
    import torch
    
    filename = os.path.basename(pdf_path)
    images, names = load_from_file(pdf_path)
    
    print(f"📄 Обрабатываем {filename} по страницам ({len(images)} страниц)...")
    
    all_predictions = []
    
    for page_idx, image in enumerate(images):
        page_attempts = 0
        max_page_attempts = 2
        
        while page_attempts <= max_page_attempts:
            try:
                # Очищаем кэш перед каждой страницей
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Обрабатываем одну страницу
                task_names = [TaskNames.ocr_with_boxes]
                page_predictions = rec_predictor(
                    [image],  # Одна страница
                    task_names=task_names,
                    det_predictor=det_predictor,
                    math_mode=False
                )
                
                all_predictions.extend(page_predictions)
                print(f"✅ Страница {page_idx + 1}/{len(images)} обработана")
                break  # Успешно
                
            except Exception as page_error:
                page_attempts += 1
                if "_seen_tokens" in str(page_error) or "ContinuousBatchingCache" in str(page_error):
                    if page_attempts <= max_page_attempts:
                        print(f"⚠️ Ошибка на странице {page_idx + 1} (попытка {page_attempts}/{max_page_attempts + 1})")
                        
                        # Пересоздаем предикторы для страницы
                        det_predictor = DetectionPredictor()
                        foundation_predictor = FoundationPredictor()
                        rec_predictor = RecognitionPredictor(foundation_predictor)
                        time.sleep(0.5)
                        continue
                    else:
                        print(f"❌ Пропускаем страницу {page_idx + 1}: {str(page_error)}")
                        # Создаем пустой результат для пропущенной страницы
                        # Создаем минимальный объект с пустыми строками
                        class EmptyResult:
                            def __init__(self):
                                self.text_lines = []
                                self.languages = []
                                self.image_bbox = [0, 0, 100, 100]
                        
                        all_predictions.append(EmptyResult())
                        break
                else:
                    raise page_error  # Неизвестная ошибка
    
    print(f"🎯 Fallback завершен для {filename}: {len(all_predictions)} страниц")
    return all_predictions


def ocr_worker(pdf_queue, ocr_queue, pdf_folder, date_format):
    """OCR воркер: непрерывно обрабатывает PDF и подает в OCR очередь"""
    det_predictor = DetectionPredictor()
    foundation_predictor = FoundationPredictor()
    rec_predictor = RecognitionPredictor(foundation_predictor)
    
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
            ocr_queue.put(("unknown_file", None, f"OCR ошибка: {e}"))

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
                
                # Передаем тип документа для статистики
                doc_type = llm_result.get("Тип_документа", llm_result.get("Тип документа", "не указан"))
                result_queue.put(f"Завершено [{display_name}]: {pdf_file} (время: {processing_time:.1f}с) - {doc_type}")
            else:
                # Ошибка LLM - проверяем возможность повтора
                max_retries = llm_settings.get('max_retries', 3)
                auto_retry = llm_settings.get('auto_retry', True)
                
                # Правильная проверка: retry_count начинается с 0, максимум max_retries попыток
                if auto_retry and retry_count < max_retries and retry_queue is not None:
                    # Добавляем в очередь повтора
                    retry_queue.put((pdf_file, truncated_data, combined_text, retry_count + 1))
                    result_queue.put(f"Повтор [{display_name}] для {pdf_file}: {llm_result['error']} (попытка {retry_count + 2}/{max_retries + 1})")
                else:
                    # Максимум попыток исчерпан или автоповтор отключен
                    if "prediction-error" in llm_result.get('error', '').lower():
                        result_queue.put(f"Ошибка LLM [{display_name}] для {pdf_file}: Превышен контекст модели - документ слишком большой (время: {processing_time:.1f}с, попытка {retry_count + 1}/{max_retries + 1})")
                    else:
                        result_queue.put(f"Ошибка LLM [{display_name}] для {pdf_file}: {llm_result['error']} (время: {processing_time:.1f}с, попытка {retry_count + 1}/{max_retries + 1})")
                
        except queue.Empty:
            continue
        except Exception as e:
            result_queue.put(f"Ошибка [{display_name}] unknown: {e}")

def ocr_worker_simple(pdf_queue, result_queue):
    """Простой OCR воркер для неблокирующей обработки - МАКСИМАЛЬНО ОПТИМИЗИРОВАННАЯ ВЕРСИЯ"""
    # КРИТИЧЕСКОЕ УЛУЧШЕНИЕ: Используем глобальный кеш!
    det_predictor, rec_predictor = get_cached_predictors()
    print("✅ Surya OCR воркер готов (используем кеш)")
    
    while True:
        try:
            item = pdf_queue.get(timeout=1)
            if item is None:  # Стоп-сигнал
                break
                
            pdf_file, pdf_folder, date_format = item
            # Передаем предикторы в функцию (не создаем новые!)
            result = ocr_single_file_worker_optimized(pdf_file, pdf_folder, date_format, det_predictor, rec_predictor)
            result_queue.put(result)
            
        except queue.Empty:
            continue
        except Exception as e:
            result_queue.put({
                "success": False,
                "filename": "unknown",
                "error": str(e)
            })

def ocr_single_file_worker_optimized(pdf_file, pdf_folder, date_format, det_predictor, rec_predictor):
    """ОПТИМИЗИРОВАННАЯ обработка одного PDF файла - ИСПОЛЬЗУЕМ ГОТОВЫЕ ПРЕДИКТОРЫ"""
    start_time = time.time()
    try:
        # Очистка GPU кэша (легкая операция)
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        pdf_path = os.path.join(pdf_folder, pdf_file)
        
        # OCR обработка с улучшенной обработкой ошибок Surya (БЕЗ пересоздания!)
        try:
            predictions = handle_surya_error_optimized(pdf_path, det_predictor, rec_predictor, max_retries=2)  # Меньше повторов
            
            # Проверяем, что predictions получены корректно
            if predictions is None:
                raise Exception("OCR вернул пустой результат")
                
        except Exception as ocr_error:
            raise ocr_error
        
        # Остальное остается так же...
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
        
        # Смарт-усечение для LLM (оптимизированное)
        all_lines = []
        for page_data in pages_data:
            all_lines.extend([{"page": page_data["page"], **line} for line in page_data["text_lines"]])
        
        truncated_lines, was_truncated, token_count = smart_truncate_for_llm(all_lines, max_tokens=15000)
        
        if was_truncated:
            print(f"✂️ Документ {pdf_file} усечен: {token_count} токенов")
        else:
            print(f"✅ Документ {pdf_file} помещается: {token_count} токенов")
        
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

# УСТАРЕВШАЯ ФУНКЦИЯ УДАЛЕНА - использовала неоптимизированное создание предикторов
# Заменена на ocr_single_file_worker_optimized() который использует готовые предикторы
# Эта функция создавала DetectionPredictor, FoundationPredictor, RecognitionPredictor для каждого документа
# что занимало 2+ секунды на инициализацию вместо использования глобального кеша

def process_single_file_worker(args):
    """Функция-воркер для multiprocessing (вне класса для избежания pickle ошибок)"""
    pdf_file, pdf_folder, json_folder, date_format, llm_settings = args
    
    try:
        # Инициализация Surya в каждом процессе
        det_predictor = DetectionPredictor()
        foundation_predictor = FoundationPredictor()
        rec_predictor = RecognitionPredictor(foundation_predictor)
        
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
        
        # Подготовка данных для LLM с умным усечением
        all_lines = []
        for page_data in pages_data:
            all_lines.extend(page_data["text_lines"])
        
        # Проверяем размер и применяем умное усечение
        max_tokens = 15000  # Оставляем 1500 токенов для промпта
        truncated_lines, token_count, was_truncated = smart_truncate_for_llm(all_lines, max_tokens)
        
        if was_truncated:
            print(f"✂️ Документ {pdf_file} усечен: {token_count} токенов")
        else:
            print(f"✅ Документ {pdf_file} помещается: {token_count} токенов")
        
        # Анализ с LLM
        model_name = llm_settings.get('models', ['local-1'])[0]  # Берем первую модель
        llm_result = analyze_with_llm_worker(pdf_file, truncated_lines, llm_settings, model_name)
        
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


def generate_llm_prompt(filename, truncated_data, structured_data):
    """Генерация улучшенного промпта для LLM с четким определением ролей и предотвращением дублирования типа документа"""
    prompt = f"""Ты эксперт по извлечению данных из российских деловых документов. Анализируй текст и извлеки структурированную информацию строго по правилам.

ВАЖНО: В JSON ответе должно быть ТОЛЬКО ОДНО поле с типом документа - "Тип_документа" - без вариаций, дублирования или альтернатив!

ОПРЕДЕЛЕНИЕ РОЛЕЙ (САМОЕ ВАЖНОЕ) В каждом документе всегда 2 стороны и в ответе должны быть 2 стороны:
1. ИСПОЛНИТЕЛЬ (кто оказывает услуги/продает товары):
   - В договоре: "Исполнитель", "Продавец", "Поставщик", изучай стороны договора чтобы решить.
   - В акте: "Исполнитель", тот кто выполнил работы/оказал услуги
   - В счете: "Поставщик", "ИП" или организация в верхней части счета

2. ЗАКАЗЧИК (кто платит/получает услуги/товары):
   - В договоре: "Заказчик", "Покупатель", "Получатель", сторона документа, изучи чтобы решить кто есть кто.
   - В акте: "Заказчик", тот кому оказаны услуги
   - В счете: "Покупатель", "Плательщик", получатель счета

ОПРЕДЕЛЕНИЕ ТИПА ДОКУМЕНТА (ТОЛЬКО ОДИН ТИП - БЕЗ ДУБЛИРОВАНИЯ!):
- "договор" - если в заголовке есть слова "договор", "соглашение", "купли-продажи"
- "акт" - если в заголовке есть слова "акт", "выполненных работ", "оказанных услуг"
- "счет" - если в заголовке есть "счет", "счет на оплату"
- "счет-фактура" - если в заголовке есть "счет-фактура"

ТИПЫ ОРГАНИЗАЦИЙ (определи точно):
- "юрлицо" - ООО, АО, ПАО, ГУП, МУП, ФГУП (ИНН 10 цифр, КПП 9 цифр)
- "ип" - ИП, Индивидуальный предприниматель (ИНН 12 цифр, без КПП)
- "физлицо" - Ф.И.О. без указания ООО/ИП (возможно ИНН 12 цифр)

ПРАВИЛА ИЗВЛЕЧЕНИЯ ДАННЫХ:
- Название организации: только официальное название (ООО "Название", ИП Иванов А.А., либо физ лицо - Иванов А.А.)
- ИНН: только числа длиной 10 цифр (юрлица) или 12 цифр (ИП/физлица)
- КПП: только 9 цифр (только для юрлиц!)
- Адрес: полный юридический адрес без названия организации
- Дата документа: ОБЯЗАТЕЛЬНО приведи к формату ДД.ММ.ГГГГ (например: 17.02.2023), независимо от исходного формата в документе

ПРИМЕРЫ ОПРЕДЕЛЕНИЯ РОЛЕЙ:
- В договоре купли-продажи автомобиля: Продавец = ИСПОЛНИТЕЛЬ, Покупатель = ЗАКАЗЧИК
- В акте выполненных работ: кто выполнил работы = ИСПОЛНИТЕЛЬ, кто принял = ЗАКАЗЧИК
- В счете: кто выставил счет = ИСПОЛНИТЕЛЬ, кому выставлен счет = ЗАКАЗЧИК

ДОКУМЕНТ:
{structured_data}

ВЕРНИ СТРОГО ФОРМАТИРОВАННЫЙ JSON В ТОЧНОМ СООТВЕТСТВИИ С ШАБЛОНОМ НИЖЕ. ТОЧНО СОБЛЮДАЙ ИМЕНА ПОЛЕЙ (с подчеркиванием, не с пробелами):
{{
  "Название_файла": "{filename}",
  "Тип_документа": "договор",  // ТОЛЬКО ОДНО ПОЛЕ С ТИПОМ! Используй "договор", "акт", "счет" или "счет-фактура" в нижнем регистре
  "Номер_документа": "",
  "Дата_документа": "",
  "Наименование_заказчика": "",
  "Наименование_исполнителя": "",
  "ИНН_заказчика": "",  // 10 или 12 цифр, не путай с КПП!
  "КПП_заказчика": "",  // 9 цифр, только для юрлиц
  "Адрес_заказчика": "",
  "ИНН_исполнителя": "",  // 10 или 12 цифр, не путай с КПП!
  "КПП_исполнителя": "",  // 9 цифр, только для юрлиц
  "Адрес_исполнителя": "",
  "Тип_заказчика": "юрлицо",  // строго одно из: юрлицо, ип, физлицо
  "Тип_исполнителя": "юрлицо"  // строго одно из: юрлицо, ип, физлицо
}}"""
    return prompt


def analyze_document(filename, truncated_data, llm_settings, model_name):
    """Обработка документа с LLM"""
    try:
        # Проверка наличия данных
        if not truncated_data or len(truncated_data) == 0:
            return {"error": "Пустые данные OCR"}
        
        structured_data = json.dumps(truncated_data, ensure_ascii=False, indent=2)
        
        # Логируем размер данных для отладки
        data_size = len(structured_data)
        print(f"📊 Отправляем в LLM: {filename}, размер {data_size} байт")
        
        # Генерируем промпт
        prompt = generate_llm_prompt(filename, truncated_data, structured_data)
        
        return send_to_llm(prompt, llm_settings, model_name)
        
    except Exception as e:
        return {"error": str(e)}


def analyze_with_llm_worker(filename, truncated_data, llm_settings, model_name):
    """Основная функция анализа документа с LLM"""
    return analyze_document(filename, truncated_data, llm_settings, model_name)


# ОПТИМИЗИРОВАННЫЙ HTTP КЛИЕНТ ДЛЯ LLM
# Кеширование HTTP сессий для повышения производительности
_HTTP_SESSIONS_CACHE = {}

def get_http_session(endpoint):
    """Получение кешированной HTTP сессии для повторного использования соединений"""
    global _HTTP_SESSIONS_CACHE
    
    if endpoint not in _HTTP_SESSIONS_CACHE:
        import requests
        from requests.adapters import HTTPAdapter
        session = requests.Session()
        # Оптимизация connection pooling
        session.mount('http://', HTTPAdapter(
            pool_connections=2, pool_maxsize=5, max_retries=1))
        session.mount('https://', HTTPAdapter(
            pool_connections=2, pool_maxsize=5, max_retries=1))
        _HTTP_SESSIONS_CACHE[endpoint] = session
        print(f"🔗 Создана HTTP сессия для {endpoint}")
    
    return _HTTP_SESSIONS_CACHE[endpoint]

def send_to_llm(prompt, llm_settings, model_name):
    """ОПТИМИЗИРОВАННАЯ отправка промпта в LLM с кешированием HTTP сессий"""
    try:
        # Поддержка OpenAI и LM Studio
        provider = llm_settings.get('provider', 'LM Studio')
        
        if provider == 'OpenAI':
            api_key = llm_settings.get('api_key', '')
            if not api_key:
                return {"error": "Не указан OpenAI API ключ"}
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            endpoint = "https://api.openai.com/v1/chat/completions"
        else:
            headers = {"Content-Type": "application/json"}
            endpoint = f"{llm_settings.get('endpoint', 'http://localhost:1234')}/v1/chat/completions"
        
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": llm_settings.get('max_tokens', 16000)
        }
        
        # Используем кешированную HTTP сессию для ускорения
        session = get_http_session(endpoint)
        
        try:
            response = session.post(
                endpoint, headers=headers, json=data, 
                timeout=llm_settings.get('timeout', 180)
            )
        except requests.exceptions.Timeout:
            return {"error": f"Таймаут соединения с {provider} (более 180с)"}
        except requests.exceptions.ConnectionError:
            return {"error": f"Не удалось подключиться к {provider} - проверьте LM Studio"}
        except Exception as e:
            return {"error": f"Ошибка связи с {provider}: {e}"}
        
        if response.status_code == 200:
            try:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                if not content:
                    return {"error": "Пустой ответ от модели"}
                
                # Поиск и извлечение JSON-блока
                start = content.find('{')
                end = content.rfind('}') + 1
                
                if start != -1 and end != -1:
                    json_str = content[start:end]
                    
                    # Предварительная обработка для исправления часто встречающихся ошибок JSON
                    json_str = fix_json_format(json_str)
                    
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        # Дополнительная попытка восстановления с более агрессивной обработкой
                        json_str = aggressive_json_repair(json_str)
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            # Если снова не удалось, возвращаем детали ошибки
                            return {"error": f"Не удалось распарсить JSON: {e}\nФрагмент ответа: {json_str[:100]}..."}
                else:
                    return {"error": f"Не найден JSON в ответе: {content[:200]}..."}
                
            except json.JSONDecodeError as e:
                return {"error": f"Ошибка парсинга JSON: {e}"}
            except Exception as e:
                return {"error": f"Неизвестная ошибка при обработке ответа: {e}"}
                
        elif response.status_code == 400:
            # Ошибка превышения контекста (LM Studio)
            try:
                error_detail = response.json().get('error', '')
                if 'context' in error_detail.lower() or 'token' in error_detail.lower():
                    return {"error": "prediction-error: Превышен контекст модели"}
                else:
                    return {"error": f"prediction-error: {error_detail}"}
            except:
                return {"error": "prediction-error: Превышен контекст модели"}
        elif response.status_code == 422:
            # Ошибка превышения контекста (OpenAI)
            try:
                error_detail = response.json().get('detail', '')
                if 'context' in error_detail.lower() or 'token' in error_detail.lower():
                    return {"error": "prediction-error: Превышен контекст модели"}
                else:
                    return {"error": f"prediction-error: {error_detail}"}
            except:
                return {"error": "prediction-error: Превышен контекст модели"}
        else:
            try:
                error_detail = response.json()
                return {"error": f"HTTP {response.status_code}: {error_detail}"}
            except:
                return {"error": f"HTTP {response.status_code}: {response.text[:200]}"}
        
    except Exception as e:
        return {"error": str(e)}


def fix_json_format(json_str):
    """Исправляет наиболее частые ошибки в JSON-строке"""
    import re
    
    # Заменяем одинарные кавычки на двойные вокруг ключей и строковых значений
    json_str = re.sub(r"([\{\s,]+)'([^']+)'\s*:", r'\1"\2":', json_str)
    json_str = re.sub(r":\s*'([^']+)'([\s,\}]+)", r':"\1"\2', json_str)
    
    # Исправляем ключи без кавычек (наиболее частая ошибка)
    json_str = re.sub(r"([\{\s,]+)([A-Za-zА-Яа-я0-9_]+)\s*:", r'\1"\2":', json_str)
    
    # Убираем комментарии в стиле JavaScript
    json_str = re.sub(r"//[^\n]*", "", json_str)
    
    # Удаляем последние запятые перед закрывающими скобками (trailing commas)
    json_str = re.sub(r',\s*\}', '}', json_str)
    
    # Преобразуем \n в пробелы внутри строковых значений
    in_string = False
    result = []
    for char in json_str:
        if char == '"' and (not result or result[-1] != '\\'):
            in_string = not in_string
        if in_string and char == '\n':
            result.append(' ')
        else:
            result.append(char)
    
    return ''.join(result)


def aggressive_json_repair(json_str):
    """Агрессивное восстановление JSON при серьезных ошибках формата"""
    import re
    
    # Основная обработка
    json_str = fix_json_format(json_str)
    
    # Дополнительные агрессивные исправления
    
    # Исправляем пропущенные запятые между объектами
    json_str = re.sub(r'("[^"]*")\s*("[^"]*"\s*:)', r'\1,\2', json_str)
    
    # Заменяем неэкранированные кавычки в строковых значениях
    in_string = False
    quote_start = -1
    result = []
    
    for i, char in enumerate(json_str):
        if char == '"' and (i == 0 or json_str[i-1] != '\\'):
            if not in_string:
                in_string = True
                quote_start = i
            else:
                in_string = False
                
        # Если внутри строки и нашли неэкранированную кавычку - экранируем её
        if in_string and char == '"' and i != quote_start and json_str[i-1] != '\\':
            result.append('\\')
            
        result.append(char)
    
    return ''.join(result)


def validate_llm_result(llm_result, original_text):
    """Валидация результатов LLM и исправление форматирования ключей и значений"""
    if "error" in llm_result:
        return llm_result
        
    try:
        import re
        normalized_result = {}
        
        # Нормализация ключей: преобразование всех ключей в формат с подчеркиванием
        key_mapping = {
            # Ключи с пробелами -> ключи с подчеркиванием
            "Тип документа": "Тип_документа",
            "Номер документа": "Номер_документа",
            "Дата документа": "Дата_документа",
            "Название файла": "Название_файла",
            "Наименование заказчика": "Наименование_заказчика",
            "Наименование исполнителя": "Наименование_исполнителя",
            "ИНН заказчика": "ИНН_заказчика",
            "КПП заказчика": "КПП_заказчика",
            "Адрес заказчика": "Адрес_заказчика",
            "ИНН исполнителя": "ИНН_исполнителя",
            "КПП исполнителя": "КПП_исполнителя",
            "Адрес исполнителя": "Адрес_исполнителя",
            "Тип заказчика": "Тип_заказчика",
            "Тип исполнителя": "Тип_исполнителя"
        }
        
        # Перенос всех значений с нормализацией ключей
        for key, value in llm_result.items():
            normalized_key = key_mapping.get(key, key)
            normalized_result[normalized_key] = value
        
        # Если был дублирующий тип документа, исправляем
        doc_type_keys = ["Тип_документа", "Тип документа"]
        found_doc_type = None
        for key in doc_type_keys:
            if key in normalized_result:
                found_doc_type = normalized_result[key].lower()
                # Удаляем ключ с пробелом, оставляем только с подчеркиванием
                if key == "Тип документа" and "Тип_документа" not in normalized_result:
                    normalized_result["Тип_документа"] = found_doc_type
                    del normalized_result[key]
        
        # Валидация и нормализация ИНН
        for key in ["ИНН_заказчика", "ИНН_исполнителя"]:
            inn = normalized_result.get(key, "")
            if inn and not (inn.isdigit() and len(inn) in [10, 12]):
                inn_matches = re.findall(r'\b\d{10}\b|\b\d{12}\b', original_text)
                if inn_matches:
                    normalized_result[key] = inn_matches.pop(0) if "заказчика" in key else inn_matches.pop(0) if inn_matches else ""
        
        # Валидация и нормализация КПП
        for key in ["КПП_заказчика", "КПП_исполнителя"]:
            kpp = normalized_result.get(key, "")
            if kpp and not (kpp.isdigit() and len(kpp) == 9):
                kpp_matches = re.findall(r'\b\d{9}\b', original_text)
                if kpp_matches:
                    normalized_result[key] = kpp_matches.pop(0) if "заказчика" in key else kpp_matches.pop(0) if kpp_matches else ""
        
        # Очистка номера документа
        doc_number = normalized_result.get("Номер_документа", "")
        if doc_number:
            clean_number = re.sub(r'\s*от\s*\d+.*', '', doc_number)
            clean_number = re.sub(r'\s*\d{1,2}[./]\d{1,2}[./]\d{2,4}.*', '', clean_number)
            normalized_result["Номер_документа"] = clean_number.strip()
        
        # Проверка и нормализация типа документа в нижнем регистре
        doc_type = normalized_result.get("Тип_документа", "").lower()
        valid_types = ["договор", "акт", "счет", "счет-фактура"]
        
        if doc_type not in valid_types:
            text_lower = original_text.lower()
            if "счет-фактура" in text_lower:
                normalized_result["Тип_документа"] = "счет-фактура"
            elif "акт" in text_lower:
                normalized_result["Тип_документа"] = "акт"
            elif "счет" in text_lower:
                normalized_result["Тип_документа"] = "счет"
            elif "договор" in text_lower:
                normalized_result["Тип_документа"] = "договор"
            else:
                normalized_result["Тип_документа"] = "неопределен"
        else:
            # Убеждаемся, что тип документа в нижнем регистре
            normalized_result["Тип_документа"] = doc_type.lower()
            
        return normalized_result
        
    except Exception as e:
        print(f"Ошибка валидации: {e}")
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
        
        # Новые переменные для расширенного выбора файлов
        self.selected_files = []  # Список выбранных отдельных файлов
        self.recursive_search = tk.BooleanVar(value=False)  # Рекурсивный поиск в подпапках
        self.file_selection_mode = tk.StringVar(value="folder")  # "folder" или "files"
        
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
        self.foundation_predictor = None
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
        
        # Счетчики типов документов
        self.acts_count = 0
        self.invoices_count = 0
        self.bills_count = 0
        self.contracts_count = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        """Настройка интерфейса"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        row = 0
        
        # Выбор режима выбора файлов
        ttk.Label(main_frame, text="Режим выбора файлов:").grid(row=row, column=0, sticky=tk.W, pady=5)
        mode_frame = ttk.Frame(main_frame)
        mode_frame.grid(row=row, column=1, sticky="ew", padx=5)
        
        ttk.Radiobutton(mode_frame, text="Папка с файлами", 
                       variable=self.file_selection_mode, value="folder",
                       command=self.on_mode_change).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Отдельные файлы", 
                       variable=self.file_selection_mode, value="files",
                       command=self.on_mode_change).pack(side=tk.LEFT, padx=(20, 0))
        row += 1
        
        # Выбор папки с PDF-файлами (для режима папки)
        self.folder_label = ttk.Label(main_frame, text="Папка с PDF-файлами:")
        self.folder_label.grid(row=row, column=0, sticky=tk.W, pady=5)
        self.folder_entry = ttk.Entry(main_frame, textvariable=self.pdf_folder, width=50)
        self.folder_entry.grid(row=row, column=1, sticky="ew", padx=5)
        self.folder_button = ttk.Button(main_frame, text="Обзор", command=self.select_pdf_folder)
        self.folder_button.grid(row=row, column=2, padx=5)
        row += 1
        
        # Опция рекурсивного поиска (для режима папки)
        self.recursive_frame = ttk.Frame(main_frame)
        self.recursive_frame.grid(row=row, column=1, sticky="ew", padx=5)
        self.recursive_checkbox = ttk.Checkbutton(
            self.recursive_frame,
            text="Искать в подпапках (рекурсивно)",
            variable=self.recursive_search
        )
        self.recursive_checkbox.pack(side=tk.LEFT)
        row += 1
        
        # Выбор отдельных файлов (для режима файлов)
        self.files_label = ttk.Label(main_frame, text="Выбранные файлы:")
        self.files_label.grid(row=row, column=0, sticky=tk.W, pady=5)
        
        files_button_frame = ttk.Frame(main_frame)
        files_button_frame.grid(row=row, column=1, sticky="ew", padx=5)
        self.select_files_button = ttk.Button(files_button_frame, text="Выбрать файлы", command=self.select_pdf_files)
        self.select_files_button.pack(side=tk.LEFT)
        self.clear_files_button = ttk.Button(files_button_frame, text="Очистить", command=self.clear_selected_files)
        self.clear_files_button.pack(side=tk.LEFT, padx=(10, 0))
        self.files_count_label = ttk.Label(files_button_frame, text="Файлов: 0")
        self.files_count_label.pack(side=tk.LEFT, padx=(10, 0))
        row += 1
        
        # Выбор папки для JSON-файлов
        ttk.Label(main_frame, text="Папка для JSON файлов:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.json_folder, width=50).grid(row=row, column=1, sticky="ew", padx=5)
        ttk.Button(main_frame, text="Обзор", command=self.select_json_folder).grid(row=row, column=2, padx=5)
        row += 1
        
        # Настройка формата даты
        ttk.Label(main_frame, text="Формат даты:").grid(row=row, column=0, sticky=tk.W, pady=5)
        date_frame = ttk.Frame(main_frame)
        date_frame.grid(row=row, column=1, sticky="ew", padx=5)
        ttk.Radiobutton(date_frame, text="ISO 8601 (2023-02-17)", variable=self.date_format, value="ISO").pack(side=tk.LEFT)
        ttk.Radiobutton(date_frame, text="Классический (17.02.2023)", variable=self.date_format, value="CLASSIC").pack(side=tk.LEFT, padx=10)
        row += 1
        
        # Настройки производительности
        ttk.Label(main_frame, text="Настройки потоков:").grid(row=row, column=0, sticky=tk.W, pady=5)
        perf_frame = ttk.Frame(main_frame)
        perf_frame.grid(row=row, column=1, sticky="ew", padx=5)
        
        ttk.Label(perf_frame, text="OCR потоков:").pack(side=tk.LEFT)
        self.ocr_threads_var = tk.StringVar(value="1")
        self.ocr_threads_spinbox = ttk.Spinbox(perf_frame, from_=1, to=8, width=5, textvariable=self.ocr_threads_var)
        self.ocr_threads_spinbox.pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(perf_frame, text="LLM потоков:").pack(side=tk.LEFT)
        self.llm_threads_var = tk.StringVar(value="1")
        self.llm_threads_spinbox = ttk.Spinbox(perf_frame, from_=1, to=4, width=5, textvariable=self.llm_threads_var)
        self.llm_threads_spinbox.pack(side=tk.LEFT, padx=5)
        row += 1
        
        # Настройки автоповтора
        retry_frame = ttk.Frame(main_frame)
        retry_frame.grid(row=row, column=1, sticky="ew", padx=5)
        
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
        llm_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)
        
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
        self.progress.grid(row=row, column=1, columnspan=2, sticky="ew", padx=5)
        row += 1
        
        # Блок статистики
        stats_frame = ttk.LabelFrame(main_frame, text="Статистика обработки", padding="10")
        stats_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)
        
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
        total_stats_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        ttk.Label(total_stats_frame, text="Общая статистика:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.total_avg_time_label = ttk.Label(total_stats_frame, text="Среднее на док.: 0 сек")
        self.total_avg_time_label.pack(anchor=tk.W)
        self.total_time_breakdown_label = ttk.Label(total_stats_frame, text="OCR + LLM: 0 + 0 с")
        self.total_time_breakdown_label.pack(anchor=tk.W)
        self.processing_speed_label = ttk.Label(total_stats_frame, text="Скорость: 0 док/мин")
        self.processing_speed_label.pack(anchor=tk.W)
        
        # Четвертая колонка - Типы документов
        doc_types_frame = ttk.Frame(stats_frame)
        doc_types_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        ttk.Label(doc_types_frame, text="Типы документов:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.acts_count_label = ttk.Label(doc_types_frame, text="Акты: 0")
        self.acts_count_label.pack(anchor=tk.W)
        self.invoices_count_label = ttk.Label(doc_types_frame, text="Счета: 0")
        self.invoices_count_label.pack(anchor=tk.W)
        self.bills_count_label = ttk.Label(doc_types_frame, text="Счет-фактуры: 0")
        self.bills_count_label.pack(anchor=tk.W)
        self.contracts_count_label = ttk.Label(doc_types_frame, text="Договоры: 0")
        self.contracts_count_label.pack(anchor=tk.W)
        
        row += 1
        
        # Лог
        ttk.Label(main_frame, text="Лог обработки:").grid(row=row, column=0, sticky=tk.W, pady=5)
        row += 1
        
        self.log_text = scrolledtext.ScrolledText(main_frame, height=20, width=80)
        self.log_text.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=5)
        row += 1
        
        # Кнопка сохранения лога
        ttk.Button(main_frame, text="Сохранить лог как txt файл", command=self.save_log).grid(row=row, column=0, columnspan=3, pady=10)
        
        # Настройка сетки
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(row-2, weight=1)
        
        # Начальная настройка провайдера
        self.on_provider_change()
        
        # Начальная настройка режима выбора файлов
        self.on_mode_change()
        
    def select_pdf_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку с PDF файлами")
        if folder:
            self.pdf_folder.set(folder)
            
    def select_pdf_files(self):
        """Выбор отдельных PDF файлов"""
        files = filedialog.askopenfilenames(
            title="Выберите PDF файлы",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if files:
            self.selected_files.extend(files)
            # Удаляем дубликаты
            self.selected_files = list(set(self.selected_files))
            self.update_files_count()
            
    def clear_selected_files(self):
        """Очистка списка выбранных файлов"""
        self.selected_files = []
        self.update_files_count()
        
    def update_files_count(self):
        """Обновление счетчика выбранных файлов"""
        count = len(self.selected_files)
        self.files_count_label.config(text=f"Файлов: {count}")
        
    def on_mode_change(self):
        """Обработка смены режима выбора файлов"""
        mode = self.file_selection_mode.get()
        
        if mode == "folder":
            # Показываем элементы для работы с папкой
            self.folder_label.grid()
            self.folder_entry.grid()
            self.folder_button.grid()
            self.recursive_frame.grid()
            self.recursive_checkbox.grid()
            
            # Скрываем элементы для работы с файлами
            files_button_frame = getattr(self, 'files_button_frame', None)
            if files_button_frame:
                files_button_frame.grid_remove()
            
        else:  # "files"
            # Скрываем элементы для работы с папкой
            self.folder_label.grid_remove()
            self.folder_entry.grid_remove()
            self.folder_button.grid_remove()
            self.recursive_frame.grid_remove()
            
            # Показываем элементы для работы с файлами
            files_button_frame = getattr(self, 'files_button_frame', None)
            if files_button_frame:
                files_button_frame.grid()
                self.select_files_button.pack(side=tk.LEFT)
                self.clear_files_button.pack(side=tk.LEFT, padx=(10, 0))
                self.files_count_label.pack(side=tk.LEFT, padx=(10, 0))
            
    def get_pdf_files_list(self):
        """Получение списка PDF файлов: КОМБИНИРУЕТ папку И отдельные файлы"""
        mode = self.file_selection_mode.get()
        
        # Инициализируем списки для комбинирования
        all_file_names = []
        all_file_paths = {}
        
        # ЧАСТЬ 1: Обработка папки (если указана)
        pdf_folder = self.pdf_folder.get()
        if pdf_folder and os.path.exists(pdf_folder):
            folder_files = []
            
            if self.recursive_search.get():
                # Рекурсивный поиск
                for root, dirs, files in os.walk(pdf_folder):
                    for file in files:
                        if file.lower().endswith('.pdf'):
                            folder_files.append(os.path.join(root, file))
                
                # Добавляем файлы из рекурсивного поиска
                for file_path in folder_files:
                    file_name = os.path.basename(file_path)
                    if file_name not in all_file_names:  # Избегаем дубликатов
                        all_file_names.append(file_name)
                        all_file_paths[file_name] = os.path.dirname(file_path)
            else:
                # Обычный поиск только в указанной папке
                folder_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
                
                # Добавляем файлы из папки
                for file_name in folder_files:
                    if file_name not in all_file_names:  # Избегаем дубликатов
                        all_file_names.append(file_name)
                        all_file_paths[file_name] = pdf_folder
        
        # ЧАСТЬ 2: Обработка отдельных файлов (если выбраны)
        if self.selected_files:
            for file_path in self.selected_files:
                file_name = os.path.basename(file_path)
                if file_name not in all_file_names:  # Избегаем дубликатов
                    all_file_names.append(file_name)
                    all_file_paths[file_name] = os.path.dirname(file_path)
        
        # РЕЗУЛЬТАТ: Возвращаем объединенный список
        if not all_file_names:
            return [], None
        
        # Если есть специальные пути, возвращаем их
        if all_file_paths:
            return all_file_names, all_file_paths
        else:
            return all_file_names, None
            
    def select_json_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку для JSON файлов")
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
    
    def update_document_type_count(self, doc_type):
        """Обновление счетчиков типов документов"""
        # Увеличиваем счетчик только если тип не пустой
        if doc_type:
            # Приводим к нижнему регистру для сравнения
            doc_type_lower = doc_type.lower()
            
            if doc_type_lower == "акт":
                self.acts_count += 1
            elif doc_type_lower in ["счёт", "счет"]:
                self.invoices_count += 1
            elif doc_type_lower == "счет-фактура":
                self.bills_count += 1
            elif doc_type_lower == "договор":
                self.contracts_count += 1
        
        # Обновляем GUI
        self.acts_count_label.config(text=f"Акты: {self.acts_count}")
        self.invoices_count_label.config(text=f"Счета: {self.invoices_count}")
        self.bills_count_label.config(text=f"Счет-фактуры: {self.bills_count}")
        self.contracts_count_label.config(text=f"Договоры: {self.contracts_count}")
        self.root.update_idletasks()
    
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
        self.foundation_predictor = FoundationPredictor()
        self.rec_predictor = RecognitionPredictor(self.foundation_predictor)
        
    def process_pdf_with_surya(self, pdf_path):
        """Обработка PDF с Surya OCR, возвращает данные по страницам"""
        try:
            images, names = load_from_file(pdf_path)
            task_names = [TaskNames.ocr_with_boxes] * len(images)
            # Инициализируем предикторы, если не инициализированы
            if self.rec_predictor is None or self.det_predictor is None:
                self.initialize_surya()
            
            if self.rec_predictor is not None:
                predictions = self.rec_predictor(
                    images,
                    task_names=task_names,
                    det_predictor=self.det_predictor,
                    math_mode=False
                )
            else:
                raise Exception("Не удалось инициализировать Surya OCR")
            
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
        """Умное усечение OCR данных для LLM с точным подсчетом токенов"""
        pages_data = ocr_json_data.get("pages_data", [])
        total_pages = len(pages_data)
        
        if total_pages == 0:
            return []
        
        # Собираем все строки со всех страниц
        all_lines = []
        for page_idx, page in enumerate(pages_data):
            for line in page["text_lines"]:
                all_lines.append({
                    "page": page_idx + 1,
                    "text": line["text"],
                    "bbox": line["bbox"],
                    "confidence": line["confidence"]
                })
        
        # Используем новую логику усечения с точным подсчетом токенов
        max_tokens = 12000  # Оставляем место для промпта (4000 токенов)
        truncated_lines, token_count, was_truncated = smart_truncate_for_llm(all_lines, max_tokens)
        
        if was_truncated:
            self.log(f"✂️ Документ усечен: {token_count} токенов")
        else:
            self.log(f"✅ Документ помещается: {token_count} токенов")
        
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
            json_folder = self.json_folder.get()
            
            if not json_folder:
                self.log("Ошибка: Не выбрана папка для JSON файлов")
                return
            
            # Получаем список файлов в зависимости от режима
            pdf_files, file_paths = self.get_pdf_files_list()
            
            if not pdf_files:
                mode = self.file_selection_mode.get()
                if mode == "files":
                    self.log("Ошибка: Не выбраны файлы для обработки")
                else:
                    self.log("Ошибка: Нет PDF файлов в выбранной папке")
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
            
            # Сбрасываем счетчики типов документов
            self.acts_count = 0
            self.invoices_count = 0
            self.bills_count = 0
            self.contracts_count = 0
            self.update_document_type_count("")  # Обновляем GUI
            
            os.makedirs(json_folder, exist_ok=True)
                
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
                if file_paths:  # Если есть специальные пути (отдельные файлы или рекурсивный поиск)
                    pdf_folder_for_file = file_paths[pdf_file]
                else:  # Обычная папка
                    pdf_folder_for_file = self.pdf_folder.get()
                pdf_queue.put((pdf_file, pdf_folder_for_file, self.date_format.get()))
            
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
                        
                        # Парсим тип документа для статистики
                        if "Завершено" in result and " - " in result:
                            try:
                                doc_type = result.split(" - ")[-1].strip()
                                if doc_type and doc_type != "Неопределен":
                                    self.update_document_type_count(doc_type)
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
            
            # Используем папку JSON, если она указана, иначе текущую директорию
            json_folder = self.json_folder.get()
            if json_folder and os.path.exists(json_folder):
                log_dir = json_folder
            else:
                log_dir = "."  # Текущая директория как fallback
            
            filename = f"surya_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            log_path = os.path.join(log_dir, filename)
            
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(log_content)
                
            messagebox.showinfo("Сохранено", f"Лог сохранен в: {log_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить лог: {str(e)}")


def main():
    root = tk.Tk()
    app = SuryaSimpleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()