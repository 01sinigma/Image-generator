"""
GGUF Model Loader
Загрузка GGUF моделей для работы на 8GB VRAM через llama-cpp-python

ВАЖНО: Для полной работы с GGUF моделями Qwen-Image-Edit требуется ComfyUI-GGUF
или специализированный загрузчик. Этот модуль предоставляет базовую интеграцию.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch

logger = logging.getLogger(__name__)

# Попытка импорта llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None
    logger.warning("llama-cpp-python не установлен. Установите через: pip install llama-cpp-python")


class GGUFModelLoader:
    """
    Загрузчик GGUF моделей с оптимизацией для 8GB VRAM
    
    Использует llama-cpp-python для работы с квантизированными моделями
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация загрузчика
        
        Args:
            config: Конфигурация модели из config.yaml
        """
        self.config = config
        self.model = None
        self.gguf_config = config.get('gguf', {})
        
        # Проверка доступности GGUF
        if not LLAMA_CPP_AVAILABLE:
            logger.error("llama-cpp-python не установлен!")
            logger.error("Установите через: pip install llama-cpp-python")
            logger.error("Для CUDA поддержки: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python")
    
    def is_available(self) -> bool:
        """Проверка доступности GGUF загрузки"""
        if not LLAMA_CPP_AVAILABLE:
            return False
        
        model_path = self.gguf_config.get('model_path', '')
        return os.path.exists(model_path)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о GGUF модели"""
        model_path = self.gguf_config.get('model_path', '')
        info = {
            'available': self.is_available(),
            'path': model_path,
            'exists': os.path.exists(model_path),
            'quantization': self.gguf_config.get('quantization', 'unknown'),
            'llama_cpp_available': LLAMA_CPP_AVAILABLE,
        }
        
        if os.path.exists(model_path):
            info['size_gb'] = os.path.getsize(model_path) / (1024**3)
        
        return info
    
    def load_model(self) -> Optional[Any]:
        """
        Загрузка GGUF модели
        
        Returns:
            Загруженная модель или None при ошибке
        """
        if not self.is_available():
            logger.error("GGUF модель недоступна")
            return None
        
        model_path = self.gguf_config.get('model_path')
        
        try:
            logger.info(f"Загрузка GGUF модели: {model_path}")
            logger.info(f"Квантизация: {self.gguf_config.get('quantization', 'Q4_K_M')}")
            
            # Определение количества слоев на GPU
            n_gpu_layers = self.gguf_config.get('n_gpu_layers', -1)
            
            # Проверка доступности CUDA
            if torch.cuda.is_available():
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"CUDA доступна. VRAM: {vram_total:.2f}GB")
                
                # Для 8GB VRAM рекомендуется ограничить слои
                if vram_total <= 8 and n_gpu_layers == -1:
                    # Примерная оценка: каждый слой ~0.2GB для Q4_K_M
                    n_gpu_layers = 35  # Оставляем ~1GB для других операций
                    logger.info(f"Ограничение GPU слоев до {n_gpu_layers} для 8GB VRAM")
            else:
                n_gpu_layers = 0
                logger.warning("CUDA недоступна, модель будет работать на CPU")
            
            # Загрузка модели
            self.model = Llama(
                model_path=model_path,
                n_ctx=self.gguf_config.get('n_ctx', 2048),
                n_gpu_layers=n_gpu_layers,
                n_batch=self.gguf_config.get('n_batch', 512),
                verbose=False,
            )
            
            logger.info("GGUF модель успешно загружена")
            return self.model
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке GGUF модели: {e}")
            return None
    
    def unload_model(self):
        """Выгрузка модели для освобождения памяти"""
        if self.model is not None:
            del self.model
            self.model = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("GGUF модель выгружена")


class GGUFDownloader:
    """Загрузка GGUF моделей с Hugging Face"""
    
    RECOMMENDED_MODELS = {
        'qwen-image-edit': {
            'repo': 'unsloth/Qwen-Image-Edit-2511-GGUF',
            'files': {
                'Q4_K_M': 'qwen-image-edit-2511-Q4_K_M.gguf',
                'Q4_K_S': 'qwen-image-edit-2511-Q4_K_S.gguf',
                'Q5_K_M': 'qwen-image-edit-2511-Q5_K_M.gguf',
                'Q6_K': 'qwen-image-edit-2511-Q6_K.gguf',
                'Q8_0': 'qwen-image-edit-2511-Q8_0.gguf',
            },
            'recommended_8gb': 'Q4_K_M',  # Рекомендуется для 8GB VRAM
            'recommended_12gb': 'Q5_K_M',  # Рекомендуется для 12GB VRAM
        }
    }
    
    @staticmethod
    def download_model(
        model_type: str,
        quantization: str = 'Q4_K_M',
        output_dir: str = './models/qwen-image-edit-gguf',
        resume: bool = True
    ) -> Optional[str]:
        """
        Скачивание GGUF модели
        
        Args:
            model_type: Тип модели ('qwen-image-edit')
            quantization: Уровень квантизации (Q4_K_M, Q4_K_S, Q5_K_M, Q6_K, Q8_0)
            output_dir: Директория для сохранения
            resume: Продолжить загрузку если файл частично скачан
        
        Returns:
            Путь к скачанному файлу или None при ошибке
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            logger.error("huggingface_hub не установлен")
            return None
        
        if model_type not in GGUFDownloader.RECOMMENDED_MODELS:
            logger.error(f"Неизвестный тип модели: {model_type}")
            return None
        
        model_info = GGUFDownloader.RECOMMENDED_MODELS[model_type]
        
        if quantization not in model_info['files']:
            logger.error(f"Неизвестная квантизация: {quantization}")
            logger.info(f"Доступные: {list(model_info['files'].keys())}")
            return None
        
        filename = model_info['files'][quantization]
        repo_id = model_info['repo']
        
        # Создание директории
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        try:
            logger.info(f"Скачивание {filename} из {repo_id}...")
            logger.info(f"Квантизация: {quantization}")
            
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=output_dir,
                resume_download=resume,
            )
            
            logger.info(f"Модель скачана: {downloaded_path}")
            return downloaded_path
            
        except Exception as e:
            logger.error(f"Ошибка при скачивании: {e}")
            return None
    
    @staticmethod
    def get_recommended_quantization(vram_gb: float) -> str:
        """
        Получение рекомендуемой квантизации для заданного объема VRAM
        
        Args:
            vram_gb: Объем VRAM в гигабайтах
        
        Returns:
            Рекомендуемый уровень квантизации
        """
        if vram_gb <= 6:
            return 'Q4_K_S'  # Минимальный размер
        elif vram_gb <= 8:
            return 'Q4_K_M'  # Оптимальный баланс для 8GB
        elif vram_gb <= 12:
            return 'Q5_K_M'  # Лучшее качество для 12GB
        elif vram_gb <= 16:
            return 'Q6_K'  # Высокое качество
        else:
            return 'Q8_0'  # Максимальное качество


def check_gguf_compatibility() -> Dict[str, Any]:
    """
    Проверка совместимости системы с GGUF
    
    Returns:
        Словарь с информацией о совместимости
    """
    result = {
        'llama_cpp_available': LLAMA_CPP_AVAILABLE,
        'cuda_available': torch.cuda.is_available(),
        'recommended_quantization': 'Q4_K_M',
        'warnings': [],
        'recommendations': [],
    }
    
    if not LLAMA_CPP_AVAILABLE:
        result['warnings'].append("llama-cpp-python не установлен")
        result['recommendations'].append(
            "Установите: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
        )
    
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        result['vram_gb'] = vram_gb
        result['gpu_name'] = torch.cuda.get_device_name(0)
        result['recommended_quantization'] = GGUFDownloader.get_recommended_quantization(vram_gb)
        
        if vram_gb < 8:
            result['warnings'].append(f"VRAM {vram_gb:.1f}GB может быть недостаточно")
            result['recommendations'].append("Используйте Q4_K_S квантизацию")
    else:
        result['warnings'].append("CUDA недоступна, модель будет работать на CPU (медленно)")
        result['recommendations'].append("Установите CUDA для ускорения")
    
    return result


if __name__ == "__main__":
    # Тестирование
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("GGUF Compatibility Check")
    print("=" * 60)
    
    compat = check_gguf_compatibility()
    
    for key, value in compat.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("Рекомендации:")
    for rec in compat.get('recommendations', []):
        print(f"  - {rec}")

