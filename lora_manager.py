"""
LoRA Manager
Управление LoRA адаптерами для моделей генерации изображений

Поддерживаемые LoRA:
- Lightning (4-step): Ускорение генерации с 40 шагов до 4
- Uncensored: Снятие ограничений на контент
- Style LoRAs: Стилизация изображений
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
import torch

logger = logging.getLogger(__name__)

# Попытка импорта peft для LoRA
try:
    from peft import PeftModel, get_peft_model, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    PeftModel = None
    logger.warning("peft не установлен. Установите через: pip install peft")

# Попытка импорта safetensors
try:
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    logger.warning("safetensors не установлен. Установите через: pip install safetensors")


class LoRAInfo:
    """Информация о LoRA адаптере"""
    
    def __init__(
        self,
        name: str,
        path: str,
        strength: float = 1.0,
        enabled: bool = True,
        description: str = "",
        category: str = "custom"
    ):
        self.name = name
        self.path = path
        self.strength = strength
        self.enabled = enabled
        self.description = description
        self.category = category  # "lightning", "style", "uncensored", "custom"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'path': self.path,
            'strength': self.strength,
            'enabled': self.enabled,
            'description': self.description,
            'category': self.category,
        }


class LoRAManager:
    """
    Менеджер LoRA адаптеров
    
    Позволяет загружать, применять и управлять LoRA для моделей Qwen и других
    """
    
    # Известные LoRA с Hugging Face/Civitai
    KNOWN_LORAS = {
        'qwen-lightning-4step': {
            'description': 'Ускорение генерации до 4 шагов',
            'category': 'lightning',
            'recommended_strength': 0.75,
            'recommended_steps': 4,
            'huggingface_repo': None,  # Будет добавлено когда LoRA станет доступна
        },
        'lighting-enhancement': {
            'description': 'Улучшение освещения',
            'category': 'style',
            'recommended_strength': 0.6,
        },
        'realistic-vision': {
            'description': 'Реалистичная стилизация',
            'category': 'style',
            'recommended_strength': 0.7,
        },
    }
    
    def __init__(self, lora_dir: str = "./models/lora"):
        """
        Инициализация менеджера
        
        Args:
            lora_dir: Директория с LoRA файлами
        """
        self.lora_dir = Path(lora_dir)
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        
        self.loaded_loras: Dict[str, LoRAInfo] = {}
        self.active_loras: List[str] = []
    
    def scan_loras(self) -> List[LoRAInfo]:
        """
        Сканирование директории на наличие LoRA файлов
        
        Returns:
            Список найденных LoRA
        """
        loras = []
        
        for ext in ['*.safetensors', '*.pt', '*.bin']:
            for path in self.lora_dir.glob(ext):
                name = path.stem
                
                # Определение категории по имени файла
                category = 'custom'
                if 'lightning' in name.lower() or '4step' in name.lower():
                    category = 'lightning'
                elif 'uncensored' in name.lower() or 'nsfw' in name.lower():
                    category = 'uncensored'
                elif 'style' in name.lower() or 'realistic' in name.lower():
                    category = 'style'
                
                lora_info = LoRAInfo(
                    name=name,
                    path=str(path),
                    strength=0.75,
                    enabled=True,
                    description=f"LoRA файл: {path.name}",
                    category=category
                )
                loras.append(lora_info)
        
        logger.info(f"Найдено {len(loras)} LoRA файлов")
        return loras
    
    def load_lora(
        self,
        model: Any,
        lora_path: str,
        strength: float = 1.0,
        adapter_name: str = "default"
    ) -> Any:
        """
        Загрузка LoRA адаптера в модель
        
        Args:
            model: Модель для применения LoRA
            lora_path: Путь к LoRA файлу
            strength: Сила применения (0.0 - 1.0)
            adapter_name: Имя адаптера
        
        Returns:
            Модель с примененным LoRA
        """
        if not os.path.exists(lora_path):
            logger.error(f"LoRA файл не найден: {lora_path}")
            return model
        
        try:
            # Загрузка весов LoRA
            if lora_path.endswith('.safetensors') and SAFETENSORS_AVAILABLE:
                lora_weights = load_safetensors(lora_path)
            else:
                lora_weights = torch.load(lora_path, map_location='cpu')
            
            # Применение весов к модели
            # Метод зависит от типа модели
            if hasattr(model, 'load_lora_weights'):
                # Для diffusers моделей
                model.load_lora_weights(lora_path, adapter_name=adapter_name)
                if strength != 1.0:
                    model.set_adapters_for_inference(
                        adapter_names=[adapter_name],
                        adapter_weights=[strength]
                    )
                logger.info(f"LoRA загружена через diffusers: {lora_path}")
                
            elif PEFT_AVAILABLE and hasattr(model, 'add_adapter'):
                # Для моделей с поддержкой peft
                model.add_adapter(lora_weights, adapter_name)
                logger.info(f"LoRA загружена через peft: {lora_path}")
                
            else:
                # Ручное применение весов
                self._apply_lora_weights(model, lora_weights, strength)
                logger.info(f"LoRA применена вручную: {lora_path}")
            
            # Сохранение информации о загруженной LoRA
            lora_name = Path(lora_path).stem
            self.loaded_loras[lora_name] = LoRAInfo(
                name=lora_name,
                path=lora_path,
                strength=strength,
                enabled=True
            )
            self.active_loras.append(lora_name)
            
            return model
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке LoRA: {e}")
            return model
    
    def _apply_lora_weights(
        self,
        model: Any,
        lora_weights: Dict[str, torch.Tensor],
        strength: float = 1.0
    ):
        """
        Применение LoRA весов напрямую к модели
        
        Args:
            model: Модель
            lora_weights: Словарь с весами LoRA
            strength: Сила применения
        """
        model_state_dict = model.state_dict()
        
        applied_count = 0
        for key, lora_tensor in lora_weights.items():
            # Преобразование ключа LoRA в ключ модели
            model_key = self._convert_lora_key(key)
            
            if model_key in model_state_dict:
                original = model_state_dict[model_key]
                
                # Применение с учетом силы
                if original.shape == lora_tensor.shape:
                    model_state_dict[model_key] = original + (lora_tensor * strength)
                    applied_count += 1
        
        if applied_count > 0:
            model.load_state_dict(model_state_dict)
            logger.info(f"Применено {applied_count} LoRA слоев")
        else:
            logger.warning("Не удалось применить LoRA слои (несовместимость ключей)")
    
    def _convert_lora_key(self, lora_key: str) -> str:
        """
        Преобразование ключа LoRA в ключ модели
        
        Args:
            lora_key: Ключ из LoRA файла
        
        Returns:
            Ключ для модели
        """
        # Убираем типичные префиксы LoRA
        key = lora_key
        for prefix in ['lora_', 'lora.', 'base_model.model.']:
            if key.startswith(prefix):
                key = key[len(prefix):]
        
        # Убираем суффиксы LoRA
        for suffix in ['.lora_A', '.lora_B', '.lora_down', '.lora_up']:
            if key.endswith(suffix):
                key = key[:-len(suffix)]
        
        return key
    
    def unload_lora(self, model: Any, adapter_name: str = "default") -> Any:
        """
        Выгрузка LoRA адаптера
        
        Args:
            model: Модель
            adapter_name: Имя адаптера для выгрузки
        
        Returns:
            Модель без LoRA
        """
        try:
            if hasattr(model, 'unload_lora_weights'):
                model.unload_lora_weights()
                logger.info(f"LoRA выгружена: {adapter_name}")
            
            if adapter_name in self.active_loras:
                self.active_loras.remove(adapter_name)
            
        except Exception as e:
            logger.error(f"Ошибка при выгрузке LoRA: {e}")
        
        return model
    
    def get_lightning_settings(self) -> Dict[str, Any]:
        """
        Получение рекомендуемых настроек для Lightning LoRA
        
        Returns:
            Словарь с рекомендуемыми настройками
        """
        return {
            'num_inference_steps': 4,  # 4 шага вместо 40
            'guidance_scale': 1.0,
            'lora_strength': 0.75,
            'description': 'Lightning LoRA позволяет генерировать за 4 шага вместо 40, ускоряя работу в ~10 раз'
        }
    
    def download_lora(
        self,
        lora_name: str,
        source: str = 'huggingface'
    ) -> Optional[str]:
        """
        Скачивание LoRA с Hugging Face или Civitai
        
        Args:
            lora_name: Имя LoRA из KNOWN_LORAS
            source: Источник ('huggingface' или 'civitai')
        
        Returns:
            Путь к скачанному файлу или None
        """
        if lora_name not in self.KNOWN_LORAS:
            logger.error(f"Неизвестная LoRA: {lora_name}")
            logger.info(f"Доступные: {list(self.KNOWN_LORAS.keys())}")
            return None
        
        lora_info = self.KNOWN_LORAS[lora_name]
        
        if source == 'huggingface' and lora_info.get('huggingface_repo'):
            try:
                from huggingface_hub import hf_hub_download
                
                repo_id = lora_info['huggingface_repo']
                filename = f"{lora_name}.safetensors"
                
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(self.lora_dir),
                    resume_download=True,
                )
                
                logger.info(f"LoRA скачана: {downloaded}")
                return downloaded
                
            except Exception as e:
                logger.error(f"Ошибка при скачивании LoRA: {e}")
                return None
        else:
            logger.warning(f"LoRA {lora_name} недоступна для автоматического скачивания")
            logger.info("Скачайте вручную с Civitai или Hugging Face")
            return None


def apply_lightning_lora_to_qwen(
    pipeline: Any,
    lora_path: Optional[str] = None,
    strength: float = 0.75
) -> Any:
    """
    Применение Lightning LoRA к Qwen модели
    
    Эта функция оптимизирована для работы с Qwen-Image-Edit
    и позволяет генерировать за 4 шага вместо 40
    
    Args:
        pipeline: QwenImageEditPlusPipeline
        lora_path: Путь к LoRA файлу (опционально)
        strength: Сила применения (0.0 - 1.0)
    
    Returns:
        Pipeline с примененной LoRA
    """
    manager = LoRAManager()
    
    if lora_path and os.path.exists(lora_path):
        logger.info("Применение Lightning LoRA к Qwen...")
        
        # Для Qwen нужно применить LoRA к Vision-башне и Language-блоку
        if hasattr(pipeline, 'transformer'):
            # Применение к transformer
            pipeline.transformer = manager.load_lora(
                pipeline.transformer,
                lora_path,
                strength=strength,
                adapter_name="lightning"
            )
        
        if hasattr(pipeline, 'text_encoder'):
            # Применение к text_encoder
            pipeline.text_encoder = manager.load_lora(
                pipeline.text_encoder,
                lora_path,
                strength=strength,
                adapter_name="lightning"
            )
        
        logger.info("Lightning LoRA применена")
        logger.info("Рекомендуемые настройки: num_inference_steps=4, guidance_scale=1.0")
    else:
        logger.warning(f"Lightning LoRA не найдена: {lora_path}")
        logger.info("Скачайте Lightning LoRA или используйте стандартные 40 шагов")
    
    return pipeline


if __name__ == "__main__":
    # Тестирование
    logging.basicConfig(level=logging.INFO)
    
    manager = LoRAManager()
    
    print("=" * 60)
    print("LoRA Manager Test")
    print("=" * 60)
    
    # Сканирование директории
    loras = manager.scan_loras()
    
    print(f"\nНайденные LoRA:")
    for lora in loras:
        print(f"  - {lora.name} ({lora.category}): {lora.path}")
    
    print(f"\nИзвестные LoRA:")
    for name, info in manager.KNOWN_LORAS.items():
        print(f"  - {name}: {info['description']}")
    
    print(f"\nНастройки Lightning LoRA:")
    settings = manager.get_lightning_settings()
    for key, value in settings.items():
        print(f"  {key}: {value}")

