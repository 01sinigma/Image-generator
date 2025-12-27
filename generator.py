"""
Z-Image Generator Module
Основной модуль для генерации изображений с использованием Z-Image-Turbo
"""

import os
import torch
import yaml
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import logging

from models import ModelFactory, BaseImageGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZImageGenerator:
    """
    Класс для генерации изображений с использованием различных моделей
    Поддерживает выбор модели через конфигурацию
    """
    
    def __init__(self, config_path: str = "config.yaml", model_type: Optional[str] = None):
        """
        Инициализация генератора
        
        Args:
            config_path: Путь к файлу конфигурации
            model_type: Тип модели ('z-image-turbo'). 
                       Если None, используется модель из конфигурации
        """
        self.config = self._load_config(config_path)
        self.model_type = model_type or self.config.get('default_model', 'z-image-turbo')
        self.generator = ModelFactory.create_generator(self.model_type, config_path)
        self.pipe = self.generator.pipe  # Для обратной совместимости
        self.device = self.generator.device
    
    def _load_config(self, config_path: str) -> dict:
        """Загрузка конфигурации из YAML файла"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Конфигурация загружена из {config_path}")
            return config
        else:
            logger.warning(f"Файл конфигурации {config_path} не найден, используются значения по умолчанию")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Возвращает конфигурацию по умолчанию"""
        return {
            'default_model': 'z-image-turbo',
            'models': {
                'z-image-turbo': {
                    'name': 'Tongyi-MAI/Z-Image-Turbo',
                    'torch_dtype': 'bfloat16',
                    'low_cpu_mem_usage': False
                }
            },
            'device': {
                'type': 'cuda',
                'enable_cpu_offload': False
            },
            'optimization': {
                'attention_backend': 'sdpa',
                'compile_model': False,
                'enable_flash_attention': False
            },
            'generation': {
                'default_height': 1024,
                'default_width': 1024,
                'default_num_inference_steps': 9,
                'default_guidance_scale': 0.0,
                'default_seed': None
            },
            'output': {
                'directory': 'outputs',
                'format': 'png',
                'quality': 95
            }
        }
    
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
            guidance_scale: Масштаб guidance (должен быть 0 для Turbo моделей)
            seed: Случайное зерно для воспроизводимости
            save_path: Путь для сохранения изображения (опционально)
        
        Returns:
            PIL.Image: Сгенерированное изображение
        """
        if isinstance(self.generator, BaseImageGenerator) and hasattr(self.generator, 'generate'):
            return self.generator.generate(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                save_path=save_path
            )
        else:
            raise RuntimeError(f"Модель {self.model_type} не поддерживает генерацию изображений")
    
    def generate_batch(
        self,
        prompts: list[str],
        output_dir: Optional[str] = None,
        **kwargs
    ) -> list[Image.Image]:
        """
        Генерация нескольких изображений
        
        Args:
            prompts: Список текстовых промптов
            output_dir: Директория для сохранения изображений
            **kwargs: Дополнительные параметры для generate()
        
        Returns:
            list[Image.Image]: Список сгенерированных изображений
        """
        images = []
        output_dir = output_dir or self.config.get('output', {}).get('directory', 'outputs')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Генерация {i+1}/{len(prompts)}")
            save_path = os.path.join(output_dir, f"generated_{i+1}.png")
            image = self.generate(prompt, save_path=save_path, **kwargs)
            images.append(image)
        
        return images

