"""
Models Module - Генераторы для всех моделей
Поддержка множества моделей с оптимизацией для 8GB VRAM
Надёжная загрузка с автоповторами
"""

import torch
import gc
import time
import sys
from PIL import Image
from pathlib import Path
from typing import Optional, List, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Максимальное количество попыток загрузки
MAX_LOAD_RETRIES = 3
RETRY_DELAY = 5.0  # секунд

# ============================================
# БАЗОВЫЙ КЛАСС ГЕНЕРАТОРА
# ============================================

class BaseGenerator(ABC):
    """Базовый класс для всех генераторов с надёжной загрузкой"""
    
    def __init__(self, model_name: str = "", device: str = "auto"):
        self.model_name = model_name
        self.pipeline = None
        self.device = self._detect_device(device)
        self.is_loaded = False
        self.load_progress = 0.0
        self.load_stage = ""
        
    def _detect_device(self, device: str) -> str:
        """Определение устройства"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _get_torch_dtype(self):
        """Определение типа данных для модели"""
        if self.device == "cuda":
            return torch.float16
        return torch.float32
    
    def _clear_memory(self):
        """Очистка памяти"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _print_status(self, message: str, progress: float = None):
        """Вывод статуса в консоль"""
        if progress is not None:
            self.load_progress = progress
            bar_len = 30
            filled = int(bar_len * progress)
            bar = '█' * filled + '░' * (bar_len - filled)
            pct = int(progress * 100)
            print(f"\r  [{bar}] {pct:3d}% | {message:<50}", end='', flush=True)
        else:
            print(f"  {message}")
    
    def _apply_optimizations(self):
        """Применение оптимизаций для экономии VRAM"""
        if self.pipeline is None:
            return
        
        self._print_status("Applying CPU offload...", 0.7)
        try:
            self.pipeline.enable_model_cpu_offload()
            logger.info("CPU Offload enabled")
        except Exception as e:
            logger.warning(f"CPU Offload not available: {e}")
        
        self._print_status("Enabling attention slicing...", 0.8)
        try:
            self.pipeline.enable_attention_slicing(1)
        except:
            pass
        
        self._print_status("Enabling VAE slicing...", 0.85)
        try:
            self.pipeline.enable_vae_slicing()
        except:
            pass
        
        self._print_status("Enabling VAE tiling...", 0.9)
        try:
            self.pipeline.enable_vae_tiling()
        except:
            pass
    
    def load_model_with_retry(self) -> bool:
        """Загрузка модели с автоматическими повторами"""
        import traceback
        
        print(f"\n{'='*60}")
        print(f"LOADING: {self.model_name}")
        print(f"{'='*60}")
        
        for attempt in range(1, MAX_LOAD_RETRIES + 1):
            try:
                print(f"\n  Attempt {attempt}/{MAX_LOAD_RETRIES}")
                
                # Показываем состояние памяти
                self._show_memory_status()
                
                # Очистка памяти перед попыткой
                self._clear_memory()
                time.sleep(0.5)
                
                # Загрузка
                self.load_model()
                
                print(f"\n  [SUCCESS] Model loaded!")
                self._show_memory_status()
                print(f"{'='*60}\n")
                return True
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"\n  [ERROR] GPU OUT OF MEMORY!")
                print(f"  Details: {str(e)[:100]}")
                self._clear_memory()
                
                if attempt < MAX_LOAD_RETRIES:
                    print(f"  Clearing memory and retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                    
            except MemoryError as e:
                print(f"\n  [ERROR] SYSTEM RAM OUT OF MEMORY!")
                print(f"  This model requires more RAM than available.")
                print(f"  Try closing other applications.")
                self._clear_memory()
                
                if attempt < MAX_LOAD_RETRIES:
                    print(f"  Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                    
            except Exception as e:
                error_msg = str(e)
                print(f"\n  [ERROR] {error_msg[:150]}")
                print(f"\n  Full traceback:")
                traceback.print_exc()
                self._clear_memory()
                
                if attempt < MAX_LOAD_RETRIES:
                    print(f"  Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
        
        print(f"\n  [FAILED] Could not load model after {MAX_LOAD_RETRIES} attempts")
        print(f"{'='*60}\n")
        return False
    
    def _show_memory_status(self):
        """Показать состояние памяти"""
        try:
            import psutil
            ram = psutil.virtual_memory()
            ram_used = ram.used / 1024**3
            ram_total = ram.total / 1024**3
            ram_free = ram.available / 1024**3
            
            print(f"  RAM: {ram_used:.1f}GB / {ram_total:.1f}GB (free: {ram_free:.1f}GB)")
            
            if torch.cuda.is_available():
                gpu_used = torch.cuda.memory_allocated() / 1024**3
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"  GPU: {gpu_used:.1f}GB / {gpu_total:.1f}GB")
        except:
            pass
    
    @abstractmethod
    def load_model(self):
        """Загрузка модели"""
        pass
    
    @abstractmethod
    def generate(self, **kwargs) -> Image.Image:
        """Генерация изображения"""
        pass
    
    def edit(self, **kwargs) -> Image.Image:
        """Редактирование изображения (по умолчанию не поддерживается)"""
        raise NotImplementedError("This model does not support editing")
    
    def inpaint(self, **kwargs) -> Image.Image:
        """Inpainting (по умолчанию не поддерживается)"""
        raise NotImplementedError("This model does not support inpainting")
    
    def unload(self):
        """Выгрузка модели из памяти"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self.is_loaded = False
            self._clear_memory()
            logger.info(f"Model {self.model_name} unloaded")

# ============================================
# Z-IMAGE-TURBO GENERATOR
# ============================================

class ZImageGenerator(BaseGenerator):
    """Генератор для Z-Image-Turbo - оптимизировано для 8GB VRAM"""
    
    def __init__(self, device: str = "auto"):
        super().__init__("Tongyi-MAI/Z-Image-Turbo", device)
    
    def load_model(self):
        """Загрузка Z-Image-Turbo с CPU offload для 8GB VRAM"""
        self._clear_memory()
        
        self._print_status("Importing diffusers...", 0.1)
        from diffusers import DiffusionPipeline
        
        self._print_status("Loading Z-Image-Turbo...", 0.2)
        print("\n  [INFO] Model ~19GB, your VRAM 8GB - using CPU offload")
        print()
        
        # Загружаем с low_cpu_mem_usage для экономии RAM
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        
        self._print_status("Enabling CPU offload for 8GB VRAM...", 0.6)
        
        # Используем CPU offload - единственный способ для 8GB VRAM
        try:
            self.pipeline.enable_sequential_cpu_offload()
            print("\n  [OK] Sequential CPU offload enabled")
            print("  [INFO] Generation: 30-60 sec per image")
        except Exception as e:
            print(f"\n  [WARN] Sequential offload failed: {e}")
            try:
                self.pipeline.enable_model_cpu_offload()
                print("  [OK] Model CPU offload enabled")
            except:
                pass
        
        # Оптимизации памяти
        self._print_status("Enabling memory optimizations...", 0.8)
        try:
            self.pipeline.enable_attention_slicing("max")
        except:
            pass
        
        try:
            self.pipeline.enable_vae_slicing()
        except:
            pass
        
        try:
            self.pipeline.enable_vae_tiling()
        except:
            pass
        
        self._print_status("Z-Image-Turbo ready!", 1.0)
        print()
        
        self.is_loaded = True
        logger.info("Z-Image-Turbo loaded with CPU offload")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 9,
        guidance_scale: float = 0.0,
        seed: int = -1,
        callback_on_step_end=None,
        **kwargs
    ) -> Image.Image:
        """Генерация изображения с поддержкой callback"""
        if not self.is_loaded:
            self.load_model()
        
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Параметры генерации
        gen_kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }
        
        # Добавляем callback если поддерживается
        if callback_on_step_end is not None:
            gen_kwargs["callback_on_step_end"] = callback_on_step_end
        
        try:
            result = self.pipeline(**gen_kwargs)
        except TypeError:
            # Если callback не поддерживается, убираем его
            gen_kwargs.pop("callback_on_step_end", None)
            result = self.pipeline(**gen_kwargs)
        
        return result.images[0]

# ============================================
# SDXL TURBO GENERATOR
# ============================================

class SDXLTurboGenerator(BaseGenerator):
    """Генератор для SDXL Turbo - быстрый + LoRA"""
    
    def __init__(self, device: str = "auto"):
        super().__init__("stabilityai/sdxl-turbo", device)
        self.lora_loaded = None
    
    def load_model(self):
        """Загрузка SDXL Turbo с прогрессом"""
        self._clear_memory()
        
        self._print_status("Importing diffusers...", 0.1)
        from diffusers import AutoPipelineForText2Image
        
        self._print_status("Downloading/loading model files...", 0.2)
        
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            self.model_name,
            torch_dtype=self._get_torch_dtype(),
            variant="fp16" if self.device == "cuda" else None,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
        
        self._print_status("Pipeline loaded, applying optimizations...", 0.6)
        
        self._apply_optimizations()
        
        self._print_status("SDXL Turbo ready!", 1.0)
        print()  # New line
        
        self.is_loaded = True
        logger.info("SDXL Turbo loaded")
    
    def load_lora(self, lora_path: str, weight: float = 0.8):
        """Загрузка LoRA"""
        if not self.is_loaded:
            self.load_model()
        
        try:
            self.pipeline.load_lora_weights(lora_path)
            self.pipeline.fuse_lora(lora_scale=weight)
            self.lora_loaded = lora_path
            logger.info(f"✅ LoRA загружен: {lora_path}")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки LoRA: {e}")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        seed: int = -1,
        callback_on_step_end=None,
        **kwargs
    ) -> Image.Image:
        """Генерация изображения с callback"""
        if not self.is_loaded:
            self.load_model()
        
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }
        
        if callback_on_step_end:
            gen_kwargs["callback_on_step_end"] = callback_on_step_end
        
        try:
            result = self.pipeline(**gen_kwargs)
        except TypeError:
            gen_kwargs.pop("callback_on_step_end", None)
            result = self.pipeline(**gen_kwargs)
        
        return result.images[0]

# ============================================
# PONY DIFFUSION GENERATOR
# ============================================

class PonyDiffusionGenerator(BaseGenerator):
    """Генератор для Pony Diffusion V6 - аниме/NSFW"""
    
    def __init__(self, device: str = "auto"):
        super().__init__("Astralite/pony-diffusion-v6-xl", device)
    
    def load_model(self):
        """Загрузка Pony Diffusion с прогрессом"""
        self._clear_memory()
        
        self._print_status("Importing diffusers...", 0.1)
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
        
        self._print_status("Downloading/loading Pony Diffusion...", 0.2)
        
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model_name,
            torch_dtype=self._get_torch_dtype(),
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
        
        self._print_status("Setting up scheduler...", 0.5)
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        self._print_status("Applying optimizations...", 0.6)
        self._apply_optimizations()
        
        self._print_status("Pony Diffusion V6 ready!", 1.0)
        print()
        
        self.is_loaded = True
        logger.info("Pony Diffusion V6 loaded")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "worst quality, low quality, blurry, bad anatomy",
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.0,
        seed: int = -1,
        **kwargs
    ) -> Image.Image:
        """Генерация изображения в стиле аниме"""
        if not self.is_loaded:
            self.load_model()
        
        # Добавляем качественные теги если их нет
        if "score_" not in prompt.lower():
            prompt = f"score_9, score_8_up, {prompt}"
        
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        return result.images[0]

# ============================================
# REALVISXL GENERATOR
# ============================================

class RealVisXLGenerator(BaseGenerator):
    """Генератор для RealVisXL V4 - фотореализм"""
    
    def __init__(self, device: str = "auto"):
        super().__init__("SG161222/RealVisXL_V4.0", device)
    
    def load_model(self):
        """Загрузка RealVisXL с прогрессом"""
        self._clear_memory()
        
        self._print_status("Importing diffusers...", 0.1)
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
        
        self._print_status("Downloading/loading RealVisXL V4...", 0.2)
        
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model_name,
            torch_dtype=self._get_torch_dtype(),
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
        
        self._print_status("Setting up scheduler...", 0.5)
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        self._print_status("Applying optimizations...", 0.6)
        self._apply_optimizations()
        
        self._print_status("RealVisXL V4 ready!", 1.0)
        print()
        
        self.is_loaded = True
        logger.info("RealVisXL V4 loaded")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "cartoon, anime, drawing, painting, blurry, low quality",
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 25,
        guidance_scale: float = 5.0,
        seed: int = -1,
        **kwargs
    ) -> Image.Image:
        """Генерация фотореалистичного изображения"""
        if not self.is_loaded:
            self.load_model()
        
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        return result.images[0]

# ============================================
# INSTRUCT-PIX2PIX GENERATOR
# ============================================

class InstructPix2PixGenerator(BaseGenerator):
    """Генератор для InstructPix2Pix - редактирование по тексту"""
    
    def __init__(self, device: str = "auto"):
        super().__init__("timbrooks/instruct-pix2pix", device)
    
    def load_model(self):
        """Загрузка InstructPix2Pix с прогрессом"""
        self._clear_memory()
        
        self._print_status("Importing diffusers...", 0.1)
        from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
        
        self._print_status("Downloading/loading InstructPix2Pix...", 0.2)
        
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.model_name,
            torch_dtype=self._get_torch_dtype(),
            safety_checker=None,
            low_cpu_mem_usage=True,
        )
        
        self._print_status("Setting up scheduler...", 0.5)
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        self._print_status("Applying optimizations...", 0.6)
        self._apply_optimizations()
        
        self._print_status("InstructPix2Pix ready!", 1.0)
        print()
        
        self.is_loaded = True
        logger.info("InstructPix2Pix loaded")
    
    def generate(self, **kwargs) -> Image.Image:
        """Генерация не поддерживается"""
        raise NotImplementedError("InstructPix2Pix только для редактирования")
    
    def edit(
        self,
        image: Image.Image,
        prompt: str,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        seed: int = -1,
        **kwargs
    ) -> Image.Image:
        """Редактирование изображения по инструкции"""
        if not self.is_loaded:
            self.load_model()
        
        # Изменяем размер под модель
        image = image.convert("RGB").resize((512, 512))
        
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        result = self.pipeline(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            generator=generator,
        )
        
        return result.images[0]

# ============================================
# SDXL INPAINTING GENERATOR
# ============================================

class SDXLInpaintingGenerator(BaseGenerator):
    """Генератор для SDXL Inpainting - замена частей"""
    
    def __init__(self, device: str = "auto"):
        super().__init__("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", device)
    
    def load_model(self):
        """Загрузка SDXL Inpainting с прогрессом"""
        self._clear_memory()
        
        self._print_status("Importing diffusers...", 0.1)
        from diffusers import StableDiffusionXLInpaintPipeline
        
        self._print_status("Downloading/loading SDXL Inpainting...", 0.2)
        
        self.pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.model_name,
            torch_dtype=self._get_torch_dtype(),
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
        
        self._print_status("Applying optimizations...", 0.6)
        self._apply_optimizations()
        
        self._print_status("SDXL Inpainting ready!", 1.0)
        print()
        
        self.is_loaded = True
        logger.info("SDXL Inpainting loaded")
    
    def generate(self, **kwargs) -> Image.Image:
        """Генерация не поддерживается"""
        raise NotImplementedError("SDXL Inpainting только для inpainting")
    
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        strength: float = 0.99,
        seed: int = -1,
        **kwargs
    ) -> Image.Image:
        """Inpainting - замена частей изображения"""
        if not self.is_loaded:
            self.load_model()
        
        # Подготовка изображений
        image = image.convert("RGB").resize((1024, 1024))
        mask = mask.convert("L").resize((1024, 1024))
        
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator,
        )
        
        return result.images[0]

# ============================================
# QWEN-IMAGE-EDIT GENERATOR
# ============================================

class QwenImageEditGenerator(BaseGenerator):
    """Генератор для Qwen-Image-Edit-2511 - мощное редактирование"""
    
    def __init__(self, device: str = "auto"):
        super().__init__("Qwen/Qwen-Image-Edit-2511", device)
    
    def load_model(self):
        """Загрузка Qwen-Image-Edit с МАКСИМАЛЬНЫМИ оптимизациями скорости"""
        import psutil
        import os
        
        # Ускорение загрузки через переменные окружения
        os.environ["SAFETENSORS_FAST_GPU"] = "1"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Быстрая загрузка
        
        ram = psutil.virtual_memory()
        ram_free = ram.available / 1024**3
        
        print("\n  [!] Model: Qwen-Image-Edit-2511 (~20GB)")
        print(f"  [!] Your free RAM: {ram_free:.1f} GB")
        print("  [!] SPEED OPTIMIZATIONS ENABLED")
        print("  [!] First load: 5-10 min, next loads: 1-2 min (cached)\n")
        
        self._clear_memory()
        
        self._print_status("Importing...", 0.05)
        from diffusers import QwenImageEditPlusPipeline
        
        self._print_status("Loading from cache/downloading...", 0.1)
        print()
        
        # БЫСТРАЯ ЗАГРУЗКА с safetensors
        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,  # Быстрее чем pickle
        )
        
        self._print_status("Enabling CPU offload for 8GB VRAM...", 0.7)
        
        # CPU offload - позволяет работать на 8GB VRAM
        try:
            self.pipeline.enable_sequential_cpu_offload()
            print("\n  [OK] Sequential CPU offload enabled")
        except Exception as e:
            print(f"\n  [WARN] Sequential offload failed, trying model offload...")
            try:
                self.pipeline.enable_model_cpu_offload()
                print("  [OK] Model CPU offload enabled")
            except:
                pass
        
        self._print_status("Memory optimizations...", 0.85)
        
        # Оптимизации памяти
        try:
            self.pipeline.enable_attention_slicing("max")
        except:
            pass
        
        try:
            self.pipeline.enable_vae_slicing()
        except:
            pass
        
        try:
            self.pipeline.enable_vae_tiling()
        except:
            pass
        
        # Попытка torch.compile для ускорения (PyTorch 2.0+)
        self._print_status("Trying torch.compile for speed...", 0.95)
        try:
            if hasattr(torch, 'compile') and self.device == "cuda":
                # Компиляция UNet для ускорения
                self.pipeline.transformer = torch.compile(
                    self.pipeline.transformer, 
                    mode="reduce-overhead",
                    fullgraph=False
                )
                print("\n  [OK] torch.compile enabled - 20-40% faster!")
        except Exception as e:
            print(f"\n  [INFO] torch.compile not available: {str(e)[:50]}")
        
        self._print_status("Qwen ready! Generation: 1-3 min per image", 1.0)
        print()
        
        self.is_loaded = True
        logger.info("Qwen-Image-Edit-2511 loaded")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 40,
        guidance_scale: float = 1.0,
        true_cfg_scale: float = 4.0,
        seed: int = -1,
        **kwargs
    ) -> Image.Image:
        """Генерация изображения"""
        if not self.is_loaded:
            self.load_model()
        
        generator = None
        if seed >= 0:
            generator = torch.Generator().manual_seed(seed)
        
        result = self.pipeline(
            image=[],  # Пустой список = генерация
            prompt=prompt,
            negative_prompt=negative_prompt or " ",
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            generator=generator,
        )
        
        return result.images[0]
    
    def edit(
        self,
        images: List[Image.Image],
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 40,
        guidance_scale: float = 1.0,
        true_cfg_scale: float = 4.0,
        seed: int = -1,
        **kwargs
    ) -> Image.Image:
        """Редактирование изображений"""
        if not self.is_loaded:
            self.load_model()
        
        generator = None
        if seed >= 0:
            generator = torch.Generator().manual_seed(seed)
        
        result = self.pipeline(
            image=images,
            prompt=prompt,
            negative_prompt=negative_prompt or " ",
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            generator=generator,
        )
        
        return result.images[0]

# ============================================
# ФАБРИКА МОДЕЛЕЙ
# ============================================

# ============================================
# OMNIGEN GENERATOR (NEW! Nov 2024)
# ============================================

class OmniGenGenerator(BaseGenerator):
    """OmniGen - новейшая модель (ноябрь 2024) для генерации и редактирования"""
    
    def __init__(self, device: str = "auto"):
        super().__init__("Shitao/OmniGen-v1", device)
    
    def load_model(self):
        """Загрузка OmniGen"""
        import psutil
        
        ram = psutil.virtual_memory()
        ram_free = ram.available / 1024**3
        
        print("\n  [!] OmniGen-v1 (November 2024)")
        print(f"  [!] Free RAM: {ram_free:.1f} GB")
        print("  [!] High quality generation + editing\n")
        
        self._clear_memory()
        
        self._print_status("Installing OmniGen...", 0.1)
        
        try:
            from OmniGen import OmniGenPipeline
        except ImportError:
            print("\n  Installing OmniGen package...")
            import subprocess
            subprocess.run(["pip", "install", "OmniGen", "-q"], check=True)
            from OmniGen import OmniGenPipeline
        
        self._print_status("Loading OmniGen model...", 0.3)
        print()
        
        self.pipeline = OmniGenPipeline.from_pretrained(self.model_name)
        
        self._print_status("OmniGen ready!", 1.0)
        print()
        
        self.is_loaded = True
        logger.info("OmniGen-v1 loaded")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        seed: int = -1,
        **kwargs
    ) -> Image.Image:
        """Генерация изображения"""
        if not self.is_loaded:
            self.load_model()
        
        result = self.pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed if seed >= 0 else None,
        )
        
        return result[0]
    
    def edit(
        self,
        image: Image.Image,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        seed: int = -1,
        **kwargs
    ) -> Image.Image:
        """Редактирование изображения"""
        if not self.is_loaded:
            self.load_model()
        
        # OmniGen использует <img> тег для входных изображений
        result = self.pipeline(
            prompt=f"<img><|image_1|></img> {prompt}",
            input_images=[image],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed if seed >= 0 else None,
        )
        
        return result[0]


class ModelFactory:
    """Фабрика для создания генераторов"""
    
    GENERATORS = {
        "z-image-turbo": ZImageGenerator,
        "sdxl-turbo": SDXLTurboGenerator,
        "pony-diffusion": PonyDiffusionGenerator,
        "realvis-xl": RealVisXLGenerator,
        "instruct-pix2pix": InstructPix2PixGenerator,
        "sdxl-inpainting": SDXLInpaintingGenerator,
        "qwen-image-edit": QwenImageEditGenerator,
        "omnigen": OmniGenGenerator,  # NEW! Nov 2024
    }
    
    @classmethod
    def create_generator(cls, model_key: str, device: str = "auto") -> BaseGenerator:
        """Создание генератора по ключу"""
        if model_key not in cls.GENERATORS:
            available = ", ".join(cls.GENERATORS.keys())
            raise ValueError(f"Неизвестная модель: {model_key}. Доступны: {available}")
        
        return cls.GENERATORS[model_key](device=device)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Получение списка доступных моделей"""
        return list(cls.GENERATORS.keys())

# ============================================
# УТИЛИТЫ
# ============================================

def get_system_info() -> dict:
    """Получение информации о системе"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_name": None,
        "vram_total": None,
        "vram_free": None,
    }
    
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["vram_total"] = props.total_memory / 1024**3
        info["vram_free"] = (props.total_memory - torch.cuda.memory_allocated()) / 1024**3
    
    return info

if __name__ == "__main__":
    # Тест
    print("Доступные модели:")
    for model in ModelFactory.get_available_models():
        print(f"  - {model}")
    
    print("\nИнформация о системе:")
    info = get_system_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

