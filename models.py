"""
Модуль для управления различными моделями генерации изображений
Поддерживает Z-Image-Turbo
"""

import os
import torch
import yaml
from pathlib import Path
from typing import Optional, Union, List
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        
        logger.info("Загрузка модели Z-Image-Turbo...")
        
        # Проверка доступности CUDA перед загрузкой
        if self.device == 'cuda':
            if not torch.cuda.is_available():
                logger.warning("CUDA недоступна, переключаюсь на CPU")
                self.device = 'cpu'
            else:
                logger.info(f"CUDA доступна. Устройство: {torch.cuda.get_device_name(0)}")
                # Очистка кэша перед загрузкой
                torch.cuda.empty_cache()
        
        # Проверка локального пути
        local_path = self.model_config.get('local_path')
        model_name = local_path if local_path and os.path.exists(local_path) else self.model_config.get('name', 'Tongyi-MAI/Z-Image-Turbo')
        
        if local_path and os.path.exists(local_path):
            logger.info(f"Использование локальной модели: {local_path}")
        else:
            logger.info(f"Загрузка модели из Hugging Face: {model_name}")
        
        torch_dtype = self._get_torch_dtype()
        low_cpu_mem_usage = self.model_config.get('low_cpu_mem_usage', False)
        
        try:
            # Согласно документации Z-Image-Turbo, можно передать torch_dtype при загрузке
            # Попробуем загрузить с torch_dtype, если не получится - загрузим без него
            try:
                self.pipe = ZImagePipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
                logger.info(f"Модель загружена с torch_dtype={torch_dtype}")
            except TypeError:
                # Если не поддерживается, загружаем без torch_dtype
                self.pipe = ZImagePipeline.from_pretrained(
                    model_name,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
                logger.info("Модель загружена без torch_dtype (будет использован автоматический dtype)")
            
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
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            # Очистка кэша в случае ошибки
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    
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


class ModelFactory:
    """Фабрика для создания генераторов различных моделей"""
    
    MODEL_TYPES = {
        'z-image-turbo': ZImageGenerator,
    }
    
    @staticmethod
    def create_generator(model_type: str, config_path: str = "config.yaml") -> BaseImageGenerator:
        """
        Создание генератора указанного типа
        
        Args:
            model_type: Тип модели ('z-image-turbo')
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

