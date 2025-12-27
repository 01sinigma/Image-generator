"""
Скрипт для скачивания всех необходимых моделей

Скачивает:
1. Qwen-Image-Edit-2511 GGUF (Q4_K_M) - для редактирования изображений
2. RMBG-2.0 - для удаления фона
3. Lightning LoRA (если доступна) - для ускорения генерации

Оптимизировано для 8GB VRAM (RTX 4060)
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.error("huggingface_hub не установлен!")
    logger.error("Установите через: pip install huggingface-hub")


def download_qwen_gguf(
    quantization: str = "Q4_K_M",
    output_dir: str = "./models/qwen-image-edit-gguf"
):
    """
    Скачивание Qwen-Image-Edit-2511 GGUF
    
    Args:
        quantization: Уровень квантизации (Q4_K_S, Q4_K_M, Q5_K_M, Q6_K, Q8_0)
        output_dir: Директория для сохранения
    """
    if not HF_AVAILABLE:
        return False
    
    repo_id = "unsloth/Qwen-Image-Edit-2511-GGUF"
    
    # Доступные файлы
    files = {
        'Q4_K_S': 'qwen-image-edit-2511-Q4_K_S.gguf',
        'Q4_K_M': 'qwen-image-edit-2511-Q4_K_M.gguf',
        'Q5_K_M': 'qwen-image-edit-2511-Q5_K_M.gguf',
        'Q6_K': 'qwen-image-edit-2511-Q6_K.gguf',
        'Q8_0': 'qwen-image-edit-2511-Q8_0.gguf',
    }
    
    if quantization not in files:
        logger.error(f"Неизвестная квантизация: {quantization}")
        logger.info(f"Доступные: {list(files.keys())}")
        return False
    
    filename = files[quantization]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info(f"Скачивание Qwen-Image-Edit-2511 GGUF ({quantization})")
    logger.info("="*60)
    logger.info(f"Репозиторий: {repo_id}")
    logger.info(f"Файл: {filename}")
    logger.info(f"Директория: {output_dir}")
    logger.info("")
    logger.info("⚠️  Размер файла: ~12-15GB")
    logger.info("⏱️  Загрузка может занять 10-30 минут")
    logger.info("")
    
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir,
            resume_download=True,
        )
        
        logger.info(f"✅ Модель скачана: {downloaded}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при скачивании: {e}")
        return False


def download_rmbg(output_dir: str = "./models/rmbg"):
    """
    Скачивание RMBG-2.0 для удаления фона
    
    Args:
        output_dir: Директория для сохранения
    """
    if not HF_AVAILABLE:
        return False
    
    repo_id = "briaai/RMBG-2.0"
    
    logger.info("="*60)
    logger.info("Скачивание RMBG-2.0 (удаление фона)")
    logger.info("="*60)
    logger.info(f"Репозиторий: {repo_id}")
    logger.info(f"Директория: {output_dir}")
    logger.info("")
    logger.info("⚠️  Размер: ~500MB")
    logger.info("")
    
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=output_dir,
            resume_download=True,
        )
        
        logger.info(f"✅ RMBG-2.0 скачана в {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при скачивании RMBG-2.0: {e}")
        return False


def download_lora(output_dir: str = "./models/lora"):
    """
    Информация о скачивании Lightning LoRA
    
    Args:
        output_dir: Директория для сохранения
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Lightning LoRA для ускорения генерации")
    logger.info("="*60)
    logger.info("")
    logger.info("⚠️  Lightning LoRA для Qwen-Image-Edit не доступна автоматически.")
    logger.info("")
    logger.info("Для скачивания:")
    logger.info("1. Посетите Civitai или Hugging Face")
    logger.info("2. Найдите 'Qwen Lightning' или '4-step LoRA'")
    logger.info("3. Скачайте файл .safetensors")
    logger.info(f"4. Поместите в: {output_dir}")
    logger.info("")
    logger.info("Рекомендуемые LoRA:")
    logger.info("  - Lighting Enhancement LoRA")
    logger.info("  - Realistic Vision LoRA")
    logger.info("  - Style LoRAs")
    logger.info("")
    
    # Создаем README в директории LoRA
    readme_path = Path(output_dir) / "README.md"
    readme_content = """# LoRA Directory

Поместите сюда ваши LoRA файлы (.safetensors или .pt)

## Рекомендуемые LoRA

### Lightning LoRA (для ускорения)
- Позволяет генерировать за 4 шага вместо 40
- Ускоряет работу в ~10 раз

### Style LoRAs
- Realistic Vision
- Anime Style
- Cyberpunk
- и другие

## Где скачать

- [Civitai](https://civitai.com/) - большая коллекция LoRA
- [Hugging Face](https://huggingface.co/) - официальные модели

## Использование

1. Скачайте LoRA файл
2. Поместите в эту директорию
3. В интерфейсе выберите нужную LoRA
4. Настройте силу применения (0.5-1.0)
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    logger.info(f"✅ README создан в {readme_path}")
    return True


def check_vram():
    """Проверка доступной VRAM"""
    try:
        import torch
        
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)
            
            logger.info("="*60)
            logger.info("Информация о GPU")
            logger.info("="*60)
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"VRAM: {vram:.2f}GB")
            logger.info("")
            
            if vram <= 6:
                logger.warning("⚠️  Мало VRAM! Рекомендуется Q4_K_S квантизация")
                return 'Q4_K_S'
            elif vram <= 8:
                logger.info("✅ 8GB VRAM - рекомендуется Q4_K_M квантизация")
                return 'Q4_K_M'
            elif vram <= 12:
                logger.info("✅ 12GB VRAM - рекомендуется Q5_K_M квантизация")
                return 'Q5_K_M'
            else:
                logger.info("✅ Много VRAM - можно использовать Q8_0 квантизацию")
                return 'Q8_0'
        else:
            logger.warning("CUDA недоступна!")
            return 'Q4_K_M'
            
    except ImportError:
        logger.warning("PyTorch не установлен")
        return 'Q4_K_M'


def main():
    parser = argparse.ArgumentParser(
        description='Скачивание моделей для Image Generator'
    )
    parser.add_argument(
        '--model',
        choices=['all', 'qwen', 'rmbg', 'lora'],
        default='all',
        help='Какую модель скачать'
    )
    parser.add_argument(
        '--quantization',
        choices=['Q4_K_S', 'Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0', 'auto'],
        default='auto',
        help='Уровень квантизации для GGUF (auto = автоопределение по VRAM)'
    )
    parser.add_argument(
        '--output-dir',
        default='./models',
        help='Базовая директория для моделей'
    )
    
    args = parser.parse_args()
    
    if not HF_AVAILABLE:
        logger.error("Установите huggingface_hub: pip install huggingface-hub")
        sys.exit(1)
    
    # Определение квантизации
    if args.quantization == 'auto':
        quantization = check_vram()
    else:
        quantization = args.quantization
    
    base_dir = Path(args.output_dir)
    
    success = True
    
    # Скачивание моделей
    if args.model in ['all', 'qwen']:
        if not download_qwen_gguf(quantization, str(base_dir / 'qwen-image-edit-gguf')):
            success = False
    
    if args.model in ['all', 'rmbg']:
        if not download_rmbg(str(base_dir / 'rmbg')):
            success = False
    
    if args.model in ['all', 'lora']:
        download_lora(str(base_dir / 'lora'))
    
    # Итог
    logger.info("")
    logger.info("="*60)
    logger.info("ИТОГ")
    logger.info("="*60)
    
    if success:
        logger.info("✅ Все модели успешно скачаны!")
        logger.info("")
        logger.info("Для запуска:")
        logger.info("  python app.py")
        logger.info("")
        logger.info("Или с оптимизацией памяти:")
        logger.info("  set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        logger.info("  python app.py")
    else:
        logger.warning("⚠️  Некоторые модели не удалось скачать")
        logger.info("Проверьте подключение к интернету и повторите попытку")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

