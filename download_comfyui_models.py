"""
Скачивание дополнительных моделей для ComfyUI + GGUF

Скачивает:
- CLIP encoder (clip_l.safetensors)
- T5 encoder FP8 (t5xxl_fp8_e4m3fn.safetensors) - экономит память
- VAE decoder (ae.safetensors)
"""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.error("huggingface_hub не установлен!")
    logger.error("Установите: pip install huggingface-hub")

BASE_DIR = Path(__file__).parent
COMFYUI_DIR = BASE_DIR / "ComfyUI"


def download_clip_models():
    """Скачивание CLIP и T5 encoders"""
    clip_dir = COMFYUI_DIR / "models" / "clip"
    clip_dir.mkdir(parents=True, exist_ok=True)
    
    models = [
        {
            "repo": "comfyanonymous/flux_text_encoders",
            "file": "clip_l.safetensors",
            "desc": "CLIP-L encoder",
            "size": "~250MB"
        },
        {
            "repo": "comfyanonymous/flux_text_encoders",
            "file": "t5xxl_fp8_e4m3fn.safetensors",
            "desc": "T5-XXL FP8 encoder (экономит память)",
            "size": "~5GB"
        },
    ]
    
    for model in models:
        target = clip_dir / model["file"]
        
        if target.exists():
            size_gb = target.stat().st_size / (1024**3)
            logger.info(f"✅ {model['desc']} уже скачан ({size_gb:.2f} GB)")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Скачивание: {model['desc']}")
        logger.info(f"{'='*60}")
        logger.info(f"Репозиторий: {model['repo']}")
        logger.info(f"Файл: {model['file']}")
        logger.info(f"Размер: {model['size']}")
        logger.info("")
        
        try:
            downloaded = hf_hub_download(
                repo_id=model["repo"],
                filename=model["file"],
                local_dir=str(clip_dir),
                resume_download=True,
            )
            logger.info(f"✅ Скачано: {downloaded}")
        except Exception as e:
            logger.error(f"❌ Ошибка: {e}")
            return False
    
    return True


def download_vae():
    """Скачивание VAE decoder"""
    vae_dir = COMFYUI_DIR / "models" / "vae"
    vae_dir.mkdir(parents=True, exist_ok=True)
    
    target = vae_dir / "ae.safetensors"
    
    if target.exists():
        size_gb = target.stat().st_size / (1024**3)
        logger.info(f"✅ VAE decoder уже скачан ({size_gb:.2f} GB)")
        return True
    
    logger.info(f"\n{'='*60}")
    logger.info("Скачивание: VAE decoder (ae.safetensors)")
    logger.info(f"{'='*60}")
    logger.info("Репозиторий: black-forest-labs/FLUX.1-schnell")
    logger.info("Размер: ~350MB")
    logger.info("")
    
    try:
        downloaded = hf_hub_download(
            repo_id="black-forest-labs/FLUX.1-schnell",
            filename="ae.safetensors",
            local_dir=str(vae_dir),
            resume_download=True,
        )
        logger.info(f"✅ Скачано: {downloaded}")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        
        # Альтернативный источник
        logger.info("\nПробую альтернативный источник...")
        try:
            downloaded = hf_hub_download(
                repo_id="black-forest-labs/FLUX.1-dev",
                filename="ae.safetensors",
                local_dir=str(vae_dir),
                resume_download=True,
            )
            logger.info(f"✅ Скачано: {downloaded}")
            return True
        except Exception as e2:
            logger.error(f"❌ Альтернативный источник тоже не сработал: {e2}")
            return False


def main():
    logger.info("""
╔══════════════════════════════════════════════════════════════╗
║     Скачивание моделей для ComfyUI + GGUF                    ║
║     CLIP + T5 + VAE                                          ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    if not HF_AVAILABLE:
        return 1
    
    if not COMFYUI_DIR.exists():
        logger.error(f"ComfyUI не найден: {COMFYUI_DIR}")
        logger.error("Сначала запустите: python setup_comfyui.py")
        return 1
    
    success = True
    
    # CLIP и T5
    logger.info("\n[1/2] Скачивание CLIP и T5 encoders...")
    if not download_clip_models():
        success = False
    
    # VAE
    logger.info("\n[2/2] Скачивание VAE decoder...")
    if not download_vae():
        success = False
    
    # Итог
    logger.info(f"\n{'='*60}")
    if success:
        logger.info("✅ ВСЕ МОДЕЛИ СКАЧАНЫ!")
        logger.info(f"{'='*60}")
        logger.info("""
Теперь можно запустить ComfyUI:
  start_comfyui.bat

И открыть в браузере:
  http://127.0.0.1:8188
""")
    else:
        logger.info("⚠️ Некоторые модели не скачались")
        logger.info(f"{'='*60}")
        logger.info("Проверьте подключение к интернету и повторите")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

