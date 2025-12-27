"""
Модуль для управления различными моделями генерации изображений
Поддерживает Z-Image-Turbo и Qwen-Image-Edit-2511
"""

import os
import torch
import yaml
from pathlib import Path
from typing import Optional, Union, List
from PIL import Image
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Импорт мониторинга (опционально, если модуль доступен)
try:
    from monitor import ProcessMonitor, ProgressTracker
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False
    ProcessMonitor = None
    ProgressTracker = None

# Импорт проверки полноты модели
try:
    from model_checker import ModelCompletenessChecker
    MODEL_CHECKER_AVAILABLE = True
except ImportError:
    MODEL_CHECKER_AVAILABLE = False
    ModelCompletenessChecker = None


class BaseImageGenerator:
    """Базовый класс для генераторов изображений"""
    
    def __init__(self, config: dict, model_config: dict):
        """
        Инициализация базового генератора
        
        Args:
            config: Общая конфигурация
            model_config: Конфигурация конкретной модели
        """
        self.config = config
        self.model_config = model_config
        self.device = self._get_device()
        self.pipe = None
    
    def _get_device(self) -> str:
        """Определение устройства для вычислений"""
        device_config = self.config.get('device', {})
        device_type = device_config.get('type', 'auto')
        
        if device_type == 'auto':
            if torch.cuda.is_available():
                try:
                    # Проверка, что CUDA действительно работает
                    torch.cuda.current_device()
                    device_name = torch.cuda.get_device_name(0)
                    logger.info(f"Автоопределение: CUDA доступна. Устройство: {device_name}")
                    return 'cuda'
                except Exception as e:
                    logger.warning(f"CUDA обнаружена, но недоступна: {e}. Используется CPU.")
                    return 'cpu'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                logger.info("Автоопределение: CUDA недоступна, используется CPU")
                return 'cpu'
        elif device_type == 'cuda':
            if not torch.cuda.is_available():
                logger.warning("CUDA недоступна, используется CPU")
                return 'cpu'
            else:
                try:
                    # Проверка, что CUDA действительно работает
                    torch.cuda.current_device()
                    device_name = torch.cuda.get_device_name(0)
                    logger.info(f"CUDA выбрана явно. Устройство: {device_name}")
                    return 'cuda'
                except Exception as e:
                    logger.warning(f"CUDA выбрана, но недоступна: {e}. Используется CPU.")
                    return 'cpu'
        else:
            return device_type
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Получение типа данных PyTorch"""
        dtype_str = self.model_config.get('torch_dtype', 'bfloat16')
        
        # На CPU используем float32 (float16/bfloat16 могут не поддерживаться)
        if self.device == 'cpu':
            return torch.float32
        
        if dtype_str == 'bfloat16':
            return torch.bfloat16
        elif dtype_str == 'float16':
            return torch.float16
        else:
            return torch.float32
    
    def _save_image(self, image: Image.Image, save_path: str):
        """Сохранение изображения"""
        output_dir = Path(save_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_format = self.config.get('output', {}).get('format', 'png')
        quality = self.config.get('output', {}).get('quality', 95)
        
        if output_format.lower() == 'jpg' or output_format.lower() == 'jpeg':
            image.save(save_path, format='JPEG', quality=quality)
        else:
            image.save(save_path, format='PNG')
        
        logger.info(f"Изображение сохранено: {save_path}")


class ZImageGenerator(BaseImageGenerator):
    """Генератор для Z-Image-Turbo (генерация изображений)"""
    
    def __init__(self, config: dict, model_config: dict):
        super().__init__(config, model_config)
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Загрузка модели Z-Image-Turbo"""
        from diffusers import ZImagePipeline
        
        # Запуск мониторинга процесса
        monitor = None
        progress = None
        if MONITOR_AVAILABLE:
            monitor = ProcessMonitor(timeout=600.0)  # 10 минут таймаут
            monitor.start()
            progress = ProgressTracker(total_steps=5, description="Загрузка Z-Image-Turbo")
        
        try:
            logger.info("Загрузка модели Z-Image-Turbo...")
            
            # Шаг 1: Проверка CUDA
            if progress:
                progress.update(1, "Проверка CUDA...")
            if monitor:
                monitor.update_activity()
            
            # Проверка доступности CUDA перед загрузкой
            if self.device == 'cuda':
                if not torch.cuda.is_available():
                    logger.warning("CUDA недоступна, переключаюсь на CPU")
                    self.device = 'cpu'
                else:
                    logger.info(f"CUDA доступна. Устройство: {torch.cuda.get_device_name(0)}")
                    # Очистка кэша перед загрузкой
                    torch.cuda.empty_cache()
            
            # Шаг 2: Определение источника модели
            if progress:
                progress.update(2, "Определение источника модели...")
            if monitor:
                monitor.update_activity()
            
            # Проверка локального пути
            local_path = self.model_config.get('local_path')
            model_name = local_path if local_path and os.path.exists(local_path) else self.model_config.get('name', 'Tongyi-MAI/Z-Image-Turbo')
            
            if local_path and os.path.exists(local_path):
                logger.info(f"Использование локальной модели: {local_path}")
            else:
                logger.info(f"Загрузка модели из Hugging Face: {model_name}")
            
            torch_dtype = self._get_torch_dtype()
            low_cpu_mem_usage = self.model_config.get('low_cpu_mem_usage', False)
            
            # Шаг 3: Загрузка модели
            if progress:
                progress.update(3, "Загрузка модели из Hugging Face...")
            if monitor:
                monitor.update_activity()
            # Согласно документации Z-Image-Turbo, можно передать torch_dtype при загрузке
            # Попробуем загрузить с torch_dtype, если не получится - загрузим без него
            try:
                self.pipe = ZImagePipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    resume_download=True,  # Продолжаем загрузку частично скачанных файлов
                )
                logger.info(f"Модель загружена с torch_dtype={torch_dtype}")
            except TypeError:
                # Если не поддерживается, загружаем без torch_dtype
                self.pipe = ZImagePipeline.from_pretrained(
                    model_name,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    resume_download=True,  # Продолжаем загрузку
                )
                logger.info("Модель загружена без torch_dtype (будет использован автоматический dtype)")
            
            if monitor:
                monitor.update_activity()
            
            # Шаг 4: Настройка CPU offload
            if progress:
                progress.update(4, "Настройка CPU offload...")
            if monitor:
                monitor.update_activity()
            
            # CPU offload работает только с CUDA, на CPU просто загружаем модель
            if self.config['device'].get('enable_cpu_offload', False) and self.device == 'cuda':
                try:
                    use_sequential = self.config['device'].get('sequential_offload', False)
                    
                    if use_sequential:
                        # Sequential CPU offload - загружает компоненты по одному (медленнее, но экономит память)
                        if hasattr(self.pipe, 'enable_sequential_cpu_offload'):
                            self.pipe.enable_sequential_cpu_offload()
                            logger.info("Sequential CPU offload включен (экономия памяти, но медленнее)")
                        else:
                            self.pipe.enable_model_cpu_offload()
                            logger.info("CPU offload включен (sequential не поддерживается)")
                    else:
                        # Обычный CPU offload - загружает несколько компонентов одновременно (быстрее)
                        self.pipe.enable_model_cpu_offload()
                        logger.info("CPU offload включен (обычный режим - быстрее, но требует больше VRAM)")
                    
                    # Синхронизация после включения CPU offload
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except Exception as e:
                    logger.warning(f"Не удалось включить CPU offload: {e}. Загружаю модель на GPU.")
                    device_obj = torch.device(self.device)
                    self.pipe.to(device_obj)
            else:
                device_obj = torch.device(self.device)
                self.pipe.to(device_obj)
            
            # Шаг 5: Финальная настройка
            if progress:
                progress.update(5, "Финальная настройка...")
            if monitor:
                monitor.update_activity()
            
            # Синхронизация и очистка кэша CUDA после загрузки
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # При CPU offload память может не показываться через memory_allocated
                # Проверяем общую память GPU
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                memory_free = memory_total - memory_reserved
                logger.info(f"VRAM после загрузки: {memory_allocated:.2f}GB выделено, {memory_reserved:.2f}GB зарезервировано")
                logger.info(f"VRAM общая: {memory_total:.2f}GB, свободно: {memory_free:.2f}GB")
            
            logger.info(f"Модель Z-Image-Turbo загружена на устройство: {self.device}")
            self._apply_optimizations()
            
            if progress:
                progress.finish("Модель Z-Image-Turbo успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            # Очистка кэша в случае ошибки
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        finally:
            if monitor:
                monitor.stop()
    
    def _apply_optimizations(self):
        """Применение оптимизаций"""
        opt_config = self.config.get('optimization', {})
        
        if opt_config.get('enable_flash_attention', False):
            attention_backend = opt_config.get('attention_backend', 'flash')
            if hasattr(self.pipe, 'transformer'):
                try:
                    if attention_backend == 'flash':
                        self.pipe.transformer.set_attention_backend("flash")
                        logger.info("Flash Attention 2 включен")
                    elif attention_backend == '_flash_3':
                        self.pipe.transformer.set_attention_backend("_flash_3")
                        logger.info("Flash Attention 3 включен")
                except Exception as e:
                    logger.warning(f"Не удалось установить attention backend: {e}")
        
        if opt_config.get('compile_model', False):
            if hasattr(self.pipe, 'transformer'):
                try:
                    self.pipe.transformer.compile()
                    logger.info("Модель скомпилирована")
                except Exception as e:
                    logger.warning(f"Не удалось скомпилировать модель: {e}")
    
    def generate(
        self,
        prompt: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> Image.Image:
        """Генерация изображения по текстовому промпту"""
        if self.pipe is None:
            raise RuntimeError("Модель не загружена")
        
        gen_config = self.config.get('generation', {})
        height = height or gen_config.get('default_height', 1024)
        width = width or gen_config.get('default_width', 1024)
        num_inference_steps = num_inference_steps or gen_config.get('default_num_inference_steps', 9)
        guidance_scale = guidance_scale if guidance_scale is not None else gen_config.get('default_guidance_scale', 0.0)
        seed = seed if seed is not None else gen_config.get('default_seed')
        
        # Проверка доступности CUDA перед генерацией
        if self.device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA недоступна, но устройство установлено как 'cuda'")
            
            # Проверка, что GPU не занят
            try:
                torch.cuda.synchronize()
                logger.info(f"CUDA синхронизирована. Устройство: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                logger.warning(f"Предупреждение при синхронизации CUDA: {e}")
            
            # Очистка кэша CUDA перед генерацией
            torch.cuda.empty_cache()
            
            # Проверка доступной памяти
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                memory_free = memory_total - memory_reserved
                logger.info(f"VRAM перед генерацией: {memory_allocated:.2f}GB выделено, {memory_reserved:.2f}GB зарезервировано")
                logger.info(f"VRAM общая: {memory_total:.2f}GB, свободно: {memory_free:.2f}GB")
        
        # Создание генератора с правильным устройством
        if seed is not None:
            device_obj = torch.device(self.device)
            generator = torch.Generator(device_obj).manual_seed(seed)
        else:
            generator = None
        
        logger.info(f"Генерация изображения: {prompt[:50]}...")
        logger.info(f"Параметры: {height}x{width}, шагов: {num_inference_steps}, device: {self.device}")
        
        try:
            result = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            
            # Синхронизация CUDA после генерации
            if self.device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            image = result.images[0]
            
            if save_path:
                self._save_image(image, save_path)
            
            logger.info("Изображение успешно сгенерировано")
            return image
            
        except RuntimeError as e:
            error_msg = str(e)
            if "CUDA" in error_msg or "cuda" in error_msg:
                logger.error(f"CUDA ошибка при генерации: {e}")
                if self.device == 'cuda':
                    # Попытка восстановления: очистка кэша и повторная синхронизация
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        logger.info("Попытка восстановления CUDA выполнена")
                    except:
                        pass
            raise
        except Exception as e:
            logger.error(f"Ошибка при генерации изображения: {e}")
            raise


class QwenImageEditGenerator(BaseImageGenerator):
    """Генератор для Qwen-Image-Edit-2511 (генерация и редактирование изображений)"""
    
    def __init__(self, config: dict, model_config: dict):
        super().__init__(config, model_config)
        self._load_pipeline()
    
    def generate(
        self,
        prompt: str,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> Image.Image:
        """
        Генерация изображения по текстовому промпту
        
        Args:
            prompt: Текстовое описание изображения
            height: Высота изображения
            width: Ширина изображения
            num_inference_steps: Количество шагов инференса
            guidance_scale: Масштаб guidance
            seed: Случайное зерно для воспроизводимости
            save_path: Путь для сохранения изображения (опционально)
        
        Returns:
            PIL.Image: Сгенерированное изображение
        """
        if self.pipe is None:
            raise RuntimeError("Модель не загружена")
        
        gen_config = self.config.get('generation', {})
        edit_config = self.model_config.get('edit', {})
        
        height = height or gen_config.get('default_height', 512)
        width = width or gen_config.get('default_width', 512)
        num_inference_steps = num_inference_steps or edit_config.get('default_num_inference_steps', 40)
        guidance_scale = guidance_scale if guidance_scale is not None else edit_config.get('default_guidance_scale', 1.0)
        true_cfg_scale = edit_config.get('default_true_cfg_scale', 4.0)
        seed = seed if seed is not None else gen_config.get('default_seed')
        
        # Проверка доступности CUDA перед генерацией
        if self.device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA недоступна, но устройство установлено как 'cuda'")
            
            try:
                torch.cuda.synchronize()
                logger.info(f"CUDA синхронизирована. Устройство: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                logger.warning(f"Предупреждение при синхронизации CUDA: {e}")
            
            torch.cuda.empty_cache()
            
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                memory_free = memory_total - memory_reserved
                logger.info(f"VRAM перед генерацией: {memory_allocated:.2f}GB выделено, {memory_reserved:.2f}GB зарезервировано")
                logger.info(f"VRAM общая: {memory_total:.2f}GB, свободно: {memory_free:.2f}GB")
        
        # Создание генератора с правильным устройством
        if seed is not None:
            device_obj = torch.device(self.device)
            generator = torch.Generator(device_obj).manual_seed(seed)
        else:
            generator = None
        
        logger.info(f"Генерация изображения: {prompt[:50]}...")
        logger.info(f"Параметры: {height}x{width}, шагов: {num_inference_steps}, device: {self.device}")
        
        try:
            # Qwen может генерировать изображения, передав пустой список для image
            # Это означает генерацию с нуля без входных изображений
            inputs = {
                "image": [],  # Пустой список для генерации с нуля
                "prompt": prompt,
                "generator": generator,
                "true_cfg_scale": true_cfg_scale,
                "negative_prompt": " ",
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": 1,
            }
            
            # Мониторинг памяти перед генерацией
            if self.device == 'cuda' and torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated(0) / 1024**3
                logger.debug(f"VRAM перед инференсом: {memory_before:.2f}GB")
            
            with torch.inference_mode():
                output = self.pipe(**inputs)
                image = output.images[0]
            
            # Синхронизация CUDA после генерации
            if self.device == 'cuda':
                torch.cuda.synchronize()
                
                # Проверка памяти после генерации
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    memory_free = memory_total - memory_reserved
                    usage_percent = (memory_reserved / memory_total) * 100
                    
                    logger.info(f"VRAM после генерации: {memory_allocated:.2f}GB выделено, {memory_reserved:.2f}GB зарезервировано")
                    logger.info(f"VRAM общая: {memory_total:.2f}GB, свободно: {memory_free:.2f}GB, использование: {usage_percent:.1f}%")
                    
                    if usage_percent > 95:
                        logger.warning("⚠️ Использование VRAM близко к максимуму!")
                    elif usage_percent > 85:
                        logger.info("ℹ️ Использование VRAM высокое, но в допустимых пределах")
                    else:
                        logger.info("✅ Использование VRAM оптимально")
                
                torch.cuda.empty_cache()
            
            if save_path:
                self._save_image(image, save_path)
            
            logger.info("Изображение успешно сгенерировано")
            return image
            
        except RuntimeError as e:
            error_msg = str(e)
            if "CUDA" in error_msg or "cuda" in error_msg:
                logger.error(f"CUDA ошибка при генерации: {e}")
                if self.device == 'cuda':
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        logger.info("Попытка восстановления CUDA выполнена")
                    except:
                        pass
            raise
        except Exception as e:
            logger.error(f"Ошибка при генерации: {e}")
            raise
    
    def _load_pipeline(self):
        """Загрузка модели Qwen-Image-Edit-2511 (GGUF или оригинальная)"""
        from diffusers import QwenImageEditPlusPipeline
        
        # Запуск мониторинга процесса
        monitor = None
        progress = None
        if MONITOR_AVAILABLE:
            monitor = ProcessMonitor(timeout=900.0)  # 15 минут таймаут для большой модели
            monitor.start()
            progress = ProgressTracker(total_steps=6, description="Загрузка Qwen-Image-Edit-2511")
        
        try:
            # Шаг 1: Проверка CUDA
            if progress:
                progress.update(1, "Проверка CUDA...")
            if monitor:
                monitor.update_activity()
            
            # Проверка доступности CUDA перед загрузкой
            if self.device == 'cuda':
                if not torch.cuda.is_available():
                    logger.warning("CUDA недоступна, переключаюсь на CPU")
                    self.device = 'cpu'
                else:
                    logger.info(f"CUDA доступна. Устройство: {torch.cuda.get_device_name(0)}")
                    # Очистка кэша перед загрузкой
                    torch.cuda.empty_cache()
        
            # Шаг 2: Детальная проверка полноты модели и докачка недостающих файлов
            if progress:
                progress.update(2, "Детальная проверка полноты модели...")
            if monitor:
                monitor.update_activity()
            
            # Проверка локального пути и полноты модели
            local_path = self.model_config.get('local_path')
            use_gguf = self.model_config.get('use_gguf', False)
            model_name = self.model_config.get('name', 'Qwen/Qwen-Image-Edit-2511')
            
            # Детальная проверка полноты модели
            use_local = False
            if MODEL_CHECKER_AVAILABLE and (local_path or True):  # Проверяем всегда, даже если local_path не указан (кэш HF)
                try:
                    checker = ModelCompletenessChecker(model_name, local_path)
                    is_complete, report = checker.check_model_completeness()
                    
                    if not is_complete:
                        logger.warning("Модель неполная, начинаю докачку недостающих файлов...")
                        if monitor:
                            monitor.update_activity()
                        
                        # Докачиваем недостающие файлы
                        download_success = checker.download_missing_files(report, max_retries=3)
                        
                        if download_success:
                            # Повторная проверка после докачки
                            is_complete, report = checker.check_model_completeness()
                            if is_complete:
                                logger.info("✅ Модель теперь полная после докачки")
                                use_local = local_path and os.path.exists(local_path)
                            else:
                                logger.warning("⚠️ После докачки модель все еще неполная, будет загружена из Hugging Face")
                        else:
                            logger.warning("⚠️ Не удалось докачать все файлы, будет загружена из Hugging Face")
                    else:
                        logger.info("✅ Модель полная, можно использовать локально")
                        use_local = local_path and os.path.exists(local_path)
                        
                except Exception as e:
                    logger.warning(f"Ошибка при проверке полноты модели: {e}")
                    logger.info("Продолжаю загрузку из Hugging Face...")
            else:
                # Упрощенная проверка, если ModelCompletenessChecker недоступен
                if local_path and os.path.exists(local_path):
                    model_index = os.path.join(local_path, 'model_index.json')
                    if os.path.exists(model_index):
                        use_local = True
                        logger.info(f"Использование локальной модели: {local_path}")
                    else:
                        logger.warning(f"Локальная модель неполная: отсутствует model_index.json")
                else:
                    logger.info("Локальный путь не указан или не существует")
            
            if not use_local:
                if use_gguf:
                    logger.info(f"Попытка загрузки GGUF модели из Hugging Face: {model_name}")
                else:
                    logger.info(f"Загрузка модели из Hugging Face: {model_name}")
                model_name = self.model_config.get('name', 'Qwen/Qwen-Image-Edit-2511')
            
            # Шаг 3: Проверка кэша Hugging Face
            if progress:
                progress.update(3, "Проверка кэша Hugging Face...")
            if monitor:
                monitor.update_activity()
            
            torch_dtype = self._get_torch_dtype()
            low_cpu_mem_usage = self.model_config.get('low_cpu_mem_usage', False)
            
            # Попытка загрузки GGUF модели (может не сработать, т.к. diffusers не поддерживает GGUF напрямую)
            if use_gguf:
                logger.info("Попытка загрузки GGUF версии модели...")
                try:
                    # Попытка загрузить через diffusers (скорее всего не сработает для GGUF)
                    self.pipe = QwenImageEditPlusPipeline.from_pretrained(
                        model_name,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=low_cpu_mem_usage,
                    )
                    logger.info(f"✅ GGUF модель успешно загружена с torch_dtype={torch_dtype}")
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось загрузить GGUF модель через diffusers: {e}")
                    logger.info("🔄 Переключение на оригинальную модель Qwen/Qwen-Image-Edit-2511 с максимальными оптимизациями для 8GB VRAM...")
                    # Fallback на оригинальную модель с максимальными оптимизациями
                    model_name = "Qwen/Qwen-Image-Edit-2511"
                    use_gguf = False
                    # Принудительно используем float16 для максимальной экономии памяти
                    if torch_dtype != torch.float16:
                        logger.info("Использование float16 для максимальной экономии памяти на 8GB VRAM")
                        torch_dtype = torch.float16
            else:
                logger.info("Загрузка оригинальной модели Qwen-Image-Edit-2511...")
                
                # Проверка кэша Hugging Face на наличие частично скачанных файлов
                try:
                    from huggingface_hub import scan_cache_dir
                    cache_info = scan_cache_dir()
                    # Ищем модель в кэше
                    for repo in cache_info.repos:
                        if 'Qwen-Image-Edit-2511' in str(repo.repo_id) or 'qwen-image-edit' in str(repo.repo_id).lower():
                            logger.info(f"Найдена частично скачанная модель в кэше: {repo.repo_id}")
                            logger.info(f"Размер в кэше: {repo.size_on_disk_str}")
                            logger.info("Продолжаем загрузку недостающих файлов...")
                            break
                except Exception as e:
                    logger.debug(f"Не удалось проверить кэш: {e}")
            
            # Шаг 4: Загрузка модели
            if progress:
                progress.update(4, "Загрузка модели из Hugging Face...")
            if monitor:
                monitor.update_activity()
            
            try:
                # QwenImageEditPlusPipeline автоматически продолжит загрузку частично скачанных файлов
                # from_pretrained использует resume_download=True по умолчанию
                logger.info("Проверка и докачка недостающих файлов модели...")
                try:
                    self.pipe = QwenImageEditPlusPipeline.from_pretrained(
                        model_name,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=low_cpu_mem_usage,
                        resume_download=True,  # Явно указываем продолжение загрузки
                    )
                    logger.info(f"Модель загружена с torch_dtype={torch_dtype}")
                    if monitor:
                        monitor.update_activity()
                except TypeError:
                    # Если не поддерживается, загружаем без него
                    self.pipe = QwenImageEditPlusPipeline.from_pretrained(
                        model_name,
                        low_cpu_mem_usage=low_cpu_mem_usage,
                        resume_download=True,  # Продолжаем загрузку
                    )
                    logger.info("Модель загружена без torch_dtype (будет использован автоматический dtype)")
                    if monitor:
                        monitor.update_activity()
                except FileNotFoundError as e:
                    # Если локальная модель неполная, пробуем загрузить из Hugging Face
                    if use_local and local_path:
                        logger.warning(f"Локальная модель неполная: {e}")
                        logger.info("Попытка загрузки из Hugging Face...")
                        model_name = self.model_config.get('name', 'Qwen/Qwen-Image-Edit-2511')
                        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
                            model_name,
                            torch_dtype=torch_dtype,
                            low_cpu_mem_usage=low_cpu_mem_usage,
                            resume_download=True,  # Продолжаем загрузку
                        )
                        logger.info(f"Модель загружена из Hugging Face с torch_dtype={torch_dtype}")
                        if monitor:
                            monitor.update_activity()
                    else:
                        raise
                
                # Шаг 5: Настройка CPU offload
                if progress:
                    progress.update(5, "Настройка CPU offload...")
                if monitor:
                    monitor.update_activity()
                
                # CPU offload работает только с CUDA, на CPU просто загружаем модель
                if self.config['device'].get('enable_cpu_offload', False) and self.device == 'cuda':
                    try:
                        use_sequential = self.config['device'].get('sequential_offload', False)
                        
                        if use_sequential:
                            # Sequential CPU offload - загружает компоненты по одному (медленнее, но экономит память)
                            if hasattr(self.pipe, 'enable_sequential_cpu_offload'):
                                self.pipe.enable_sequential_cpu_offload()
                                logger.info("Sequential CPU offload включен (экономия памяти, но медленнее)")
                            else:
                                self.pipe.enable_model_cpu_offload()
                                logger.info("CPU offload включен (sequential не поддерживается)")
                        else:
                            # Обычный CPU offload - загружает несколько компонентов одновременно (быстрее)
                            self.pipe.enable_model_cpu_offload()
                            logger.info("CPU offload включен (обычный режим - быстрее, но требует больше VRAM)")
                        
                        # Синхронизация после включения CPU offload
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    except Exception as e:
                        logger.warning(f"Не удалось включить CPU offload: {e}. Загружаю модель на GPU.")
                        device_obj = torch.device(self.device)
                        self.pipe.to(device_obj)
                else:
                    device_obj = torch.device(self.device)
                    self.pipe.to(device_obj)
            
                # Шаг 6: Финальная настройка
                if progress:
                    progress.update(6, "Финальная настройка...")
                if monitor:
                    monitor.update_activity()
                
                # Синхронизация и очистка кэша CUDA после загрузки
                if self.device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    memory_free = memory_total - memory_reserved
                    logger.info(f"VRAM после загрузки: {memory_allocated:.2f}GB выделено, {memory_reserved:.2f}GB зарезервировано")
                    logger.info(f"VRAM общая: {memory_total:.2f}GB, свободно: {memory_free:.2f}GB")
                
                # Отключение progress bar по умолчанию
                if hasattr(self.pipe, 'set_progress_bar_config'):
                    self.pipe.set_progress_bar_config(disable=None)
                
                logger.info(f"Модель Qwen-Image-Edit-2511 загружена на устройство: {self.device}")
                
                if progress:
                    progress.finish("Модель Qwen-Image-Edit-2511 успешно загружена")
            except Exception as e:
                logger.error(f"Ошибка при загрузке модели: {e}")
                # Очистка кэша в случае ошибки
                if self.device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
        except Exception as e:
            logger.error(f"Критическая ошибка при загрузке модели: {e}")
            # Очистка кэша в случае ошибки
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        finally:
            if monitor:
                monitor.stop()
    
    def edit(
        self,
        images: Union[Image.Image, List[Image.Image]],
        prompt: str,
        negative_prompt: str = " ",
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        true_cfg_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> Image.Image:
        """
        Редактирование изображения(ий) по текстовому промпту
        
        Args:
            images: Одно или несколько входных изображений
            prompt: Текстовое описание желаемого результата
            negative_prompt: Негативный промпт
            num_inference_steps: Количество шагов инференса
            guidance_scale: Масштаб guidance
            true_cfg_scale: Масштаб CFG для Qwen
            num_images_per_prompt: Количество изображений на промпт
            seed: Случайное зерно
            save_path: Путь для сохранения
        
        Returns:
            PIL.Image: Отредактированное изображение
        """
        if self.pipe is None:
            raise RuntimeError("Модель не загружена")
        
        # Преобразование одного изображения в список
        if isinstance(images, Image.Image):
            images = [images]
        
        # Получение значений по умолчанию
        edit_config = self.model_config.get('edit', {})
        num_inference_steps = num_inference_steps or edit_config.get('default_num_inference_steps', 40)
        guidance_scale = guidance_scale if guidance_scale is not None else edit_config.get('default_guidance_scale', 1.0)
        true_cfg_scale = true_cfg_scale if true_cfg_scale is not None else edit_config.get('default_true_cfg_scale', 4.0)
        
        # Проверка доступности CUDA перед редактированием
        if self.device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA недоступна, но устройство установлено как 'cuda'")
            
            # Проверка, что GPU не занят
            try:
                torch.cuda.synchronize()
                logger.info(f"CUDA синхронизирована. Устройство: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                logger.warning(f"Предупреждение при синхронизации CUDA: {e}")
            
            # Очистка кэша CUDA перед редактированием
            torch.cuda.empty_cache()
            
            # Проверка доступной памяти
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                memory_free = memory_total - memory_reserved
                logger.info(f"VRAM перед редактированием: {memory_allocated:.2f}GB выделено, {memory_reserved:.2f}GB зарезервировано")
                logger.info(f"VRAM общая: {memory_total:.2f}GB, свободно: {memory_free:.2f}GB")
        
        # Создание генератора с правильным устройством
        if seed is not None:
            device_obj = torch.device(self.device)
            generator = torch.Generator(device_obj).manual_seed(seed)
        else:
            generator = None
        
        logger.info(f"Редактирование изображения: {prompt[:50]}...")
        logger.info(f"Параметры: шагов: {num_inference_steps}, device: {self.device}")
        
        try:
            inputs = {
                "image": images,
                "prompt": prompt,
                "generator": generator,
                "true_cfg_scale": true_cfg_scale,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": num_images_per_prompt,
            }
            
            # Мониторинг памяти перед редактированием
            if self.device == 'cuda' and torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated(0) / 1024**3
                logger.debug(f"VRAM перед инференсом: {memory_before:.2f}GB")
            
            with torch.inference_mode():
                output = self.pipe(**inputs)
                image = output.images[0]
            
            # Синхронизация CUDA после редактирования
            if self.device == 'cuda':
                torch.cuda.synchronize()
                
                # Проверка памяти после редактирования
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    memory_free = memory_total - memory_reserved
                    usage_percent = (memory_reserved / memory_total) * 100
                    
                    logger.info(f"VRAM после редактирования: {memory_allocated:.2f}GB выделено, {memory_reserved:.2f}GB зарезервировано")
                    logger.info(f"VRAM общая: {memory_total:.2f}GB, свободно: {memory_free:.2f}GB, использование: {usage_percent:.1f}%")
                    
                    # Предупреждение при высоком использовании памяти
                    if usage_percent > 95:
                        logger.warning("⚠️ Использование VRAM близко к максимуму! Рекомендуется:")
                        logger.warning("   - Убедитесь, что enable_cpu_offload: true")
                        logger.warning("   - Убедитесь, что sequential_offload: true")
                        logger.warning("   - Закройте другие приложения, использующие GPU")
                    elif usage_percent > 85:
                        logger.info("ℹ️ Использование VRAM высокое, но в допустимых пределах")
                    else:
                        logger.info("✅ Использование VRAM оптимально")
                
                # Очистка кэша после редактирования (важно для CPU offload)
                torch.cuda.empty_cache()
            
            if save_path:
                self._save_image(image, save_path)
            
            logger.info("Изображение успешно отредактировано")
            return image
            
        except RuntimeError as e:
            error_msg = str(e)
            if "CUDA" in error_msg or "cuda" in error_msg:
                logger.error(f"CUDA ошибка при редактировании: {e}")
                if self.device == 'cuda':
                    # Попытка восстановления: очистка кэша и повторная синхронизация
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        logger.info("Попытка восстановления CUDA выполнена")
                    except:
                        pass
            raise
        except Exception as e:
            logger.error(f"Ошибка при редактировании изображения: {e}")
            raise


class ModelFactory:
    """Фабрика для создания генераторов различных моделей"""
    
    MODEL_TYPES = {
        'z-image-turbo': ZImageGenerator,
        'qwen-image-edit': QwenImageEditGenerator,
    }
    
    @staticmethod
    def create_generator(model_type: str, config_path: str = "config.yaml") -> BaseImageGenerator:
        """
        Создание генератора указанного типа
        
        Args:
            model_type: Тип модели ('z-image-turbo' или 'qwen-image-edit')
            config_path: Путь к файлу конфигурации
        
        Returns:
            BaseImageGenerator: Экземпляр генератора
        """
        # Загрузка конфигурации
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Файл конфигурации {config_path} не найден")
        
        # Получение конфигурации модели
        models_config = config.get('models', {})
        if model_type not in models_config:
            raise ValueError(f"Модель '{model_type}' не найдена в конфигурации. Доступные модели: {list(models_config.keys())}")
        
        model_config = models_config[model_type]
        model_class = ModelFactory.MODEL_TYPES.get(model_type)
        
        if model_class is None:
            raise ValueError(f"Неизвестный тип модели: {model_type}. Доступные типы: {list(ModelFactory.MODEL_TYPES.keys())}")
        
        return model_class(config, model_config)
    
    @staticmethod
    def get_available_models(config_path: str = "config.yaml") -> List[str]:
        """Получение списка доступных моделей"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return list(config.get('models', {}).keys())
        return []

