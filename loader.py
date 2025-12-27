"""
Robust Model Loader with Progress Animation
Надёжная загрузка моделей с визуализацией прогресса
"""

import sys
import time
import threading
import psutil
import torch
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum

# ============================================
# СТАТУСЫ ЗАГРУЗКИ
# ============================================

class LoadStage(Enum):
    INIT = "Инициализация"
    DOWNLOAD = "Скачивание"
    LOAD_WEIGHTS = "Загрузка весов"
    OPTIMIZE = "Оптимизация"
    READY = "Готово"
    ERROR = "Ошибка"

@dataclass
class LoadProgress:
    stage: LoadStage
    progress: float  # 0.0 - 1.0
    message: str
    download_speed: float = 0.0  # MB/s
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    gpu_used_gb: float = 0.0
    gpu_percent: float = 0.0

# ============================================
# АНИМАЦИЯ В ТЕРМИНАЛЕ
# ============================================

class TerminalAnimation:
    """Анимация загрузки в терминале"""
    
    SPINNER = ['|', '/', '-', '\\']
    PROGRESS_CHARS = ['░', '▒', '▓', '█']
    
    def __init__(self):
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.current_progress = LoadProgress(
            stage=LoadStage.INIT,
            progress=0.0,
            message="Preparing..."
        )
        self.frame = 0
    
    def start(self):
        """Запуск анимации"""
        self.running = True
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Остановка анимации"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        print()  # New line
    
    def update(self, progress: LoadProgress):
        """Обновление прогресса"""
        self.current_progress = progress
    
    def _get_system_stats(self) -> tuple:
        """Получение системных метрик"""
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        ram_used = ram.used / 1024**3
        
        gpu_used = 0.0
        gpu_percent = 0.0
        if torch.cuda.is_available():
            try:
                gpu_used = torch.cuda.memory_allocated() / 1024**3
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_percent = (gpu_used / gpu_total) * 100
            except:
                pass
        
        return cpu, ram_used, gpu_used, gpu_percent
    
    def _make_progress_bar(self, progress: float, width: int = 30) -> str:
        """Создание прогресс-бара"""
        filled = int(width * progress)
        empty = width - filled
        
        bar = '█' * filled + '░' * empty
        percent = int(progress * 100)
        
        return f"[{bar}] {percent:3d}%"
    
    def _animate(self):
        """Цикл анимации"""
        while self.running:
            self.frame = (self.frame + 1) % len(self.SPINNER)
            spinner = self.SPINNER[self.frame]
            
            p = self.current_progress
            cpu, ram, gpu, gpu_pct = self._get_system_stats()
            
            # Формируем строку
            stage_icon = {
                LoadStage.INIT: "...",
                LoadStage.DOWNLOAD: "NET",
                LoadStage.LOAD_WEIGHTS: "CPU",
                LoadStage.OPTIMIZE: "GPU",
                LoadStage.READY: "OK!",
                LoadStage.ERROR: "ERR",
            }.get(p.stage, "???")
            
            progress_bar = self._make_progress_bar(p.progress)
            
            # Индикаторы ресурсов
            net_indicator = "NET" if p.stage == LoadStage.DOWNLOAD else "   "
            cpu_indicator = f"CPU:{cpu:4.0f}%"
            ram_indicator = f"RAM:{ram:4.1f}G"
            gpu_indicator = f"GPU:{gpu:4.1f}G" if torch.cuda.is_available() else "GPU:N/A "
            
            # Скорость загрузки
            speed = ""
            if p.download_speed > 0:
                speed = f" {p.download_speed:.1f}MB/s"
            
            line = f"\r{spinner} [{stage_icon}] {progress_bar} | {cpu_indicator} {ram_indicator} {gpu_indicator}{speed} | {p.message[:40]:<40}"
            
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except:
                pass
            
            time.sleep(0.15)

# ============================================
# НАДЁЖНЫЙ ЗАГРУЗЧИК
# ============================================

class RobustModelLoader:
    """Надёжный загрузчик моделей с повторами"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 5.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.animation = TerminalAnimation()
        self.on_progress: Optional[Callable[[LoadProgress], None]] = None
    
    def _update_progress(self, stage: LoadStage, progress: float, message: str, **kwargs):
        """Обновление прогресса"""
        p = LoadProgress(
            stage=stage,
            progress=progress,
            message=message,
            **kwargs
        )
        self.animation.update(p)
        if self.on_progress:
            self.on_progress(p)
    
    def _clear_memory(self):
        """Очистка памяти перед загрузкой"""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def load_model(self, model_key: str, generator_class, device: str = "auto") -> Optional[object]:
        """
        Загрузка модели с повторами и визуализацией
        
        Args:
            model_key: ключ модели
            generator_class: класс генератора
            device: устройство
            
        Returns:
            Загруженный генератор или None
        """
        print(f"\n{'='*60}")
        print(f"LOADING MODEL: {model_key}")
        print(f"{'='*60}")
        
        self.animation.start()
        
        for attempt in range(1, self.max_retries + 1):
            try:
                self._update_progress(
                    LoadStage.INIT, 
                    0.0, 
                    f"Attempt {attempt}/{self.max_retries}..."
                )
                
                # Очистка памяти
                self._clear_memory()
                time.sleep(0.5)
                
                self._update_progress(
                    LoadStage.INIT, 
                    0.1, 
                    "Creating generator..."
                )
                
                # Создание генератора
                generator = generator_class(device=device)
                
                self._update_progress(
                    LoadStage.DOWNLOAD, 
                    0.2, 
                    "Downloading/checking files..."
                )
                
                # Загрузка модели
                generator.load_model()
                
                self._update_progress(
                    LoadStage.READY, 
                    1.0, 
                    "Model loaded successfully!"
                )
                
                self.animation.stop()
                
                print(f"\n[OK] Model {model_key} loaded successfully!")
                print(f"{'='*60}\n")
                
                return generator
                
            except torch.cuda.OutOfMemoryError as e:
                self.animation.stop()
                print(f"\n[ERROR] GPU out of memory!")
                print(f"Clearing memory and retrying in {self.retry_delay}s...")
                
                self._clear_memory()
                time.sleep(self.retry_delay)
                self.animation.start()
                
            except Exception as e:
                self.animation.stop()
                error_msg = str(e)[:100]
                print(f"\n[ERROR] Attempt {attempt} failed: {error_msg}")
                
                if attempt < self.max_retries:
                    print(f"Retrying in {self.retry_delay}s...")
                    self._clear_memory()
                    time.sleep(self.retry_delay)
                    self.animation.start()
                else:
                    print(f"[FAILED] All {self.max_retries} attempts failed!")
                    return None
        
        self.animation.stop()
        return None

# ============================================
# ОПТИМИЗИРОВАННЫЕ ГЕНЕРАТОРЫ
# ============================================

def create_optimized_sdxl_turbo(device: str = "auto"):
    """Создание SDXL Turbo с максимальными оптимизациями"""
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    from diffusers import AutoPipelineForText2Image
    
    # Определяем устройство
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("  [1/4] Loading pipeline...")
    
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    
    print("  [2/4] Enabling CPU offload...")
    
    # Агрессивные оптимизации
    pipe.enable_model_cpu_offload()
    
    print("  [3/4] Enabling memory optimizations...")
    
    try:
        pipe.enable_attention_slicing(1)
    except:
        pass
    
    try:
        pipe.enable_vae_slicing()
    except:
        pass
    
    try:
        pipe.enable_vae_tiling()
    except:
        pass
    
    print("  [4/4] Model ready!")
    
    return pipe


def create_optimized_z_image(device: str = "auto"):
    """Создание Z-Image-Turbo с оптимизациями"""
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    from diffusers import FluxPipeline
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("  [1/3] Loading pipeline...")
    
    pipe = FluxPipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    
    print("  [2/3] Enabling optimizations...")
    
    pipe.enable_model_cpu_offload()
    
    try:
        pipe.enable_attention_slicing(1)
    except:
        pass
    
    print("  [3/3] Model ready!")
    
    return pipe


# ============================================
# ТЕСТ
# ============================================

if __name__ == "__main__":
    print("Testing animation...")
    
    anim = TerminalAnimation()
    anim.start()
    
    stages = [
        (LoadStage.INIT, 0.0, "Initializing..."),
        (LoadStage.DOWNLOAD, 0.2, "Downloading model files..."),
        (LoadStage.DOWNLOAD, 0.4, "Downloading weights..."),
        (LoadStage.LOAD_WEIGHTS, 0.6, "Loading into memory..."),
        (LoadStage.OPTIMIZE, 0.8, "Optimizing for GPU..."),
        (LoadStage.READY, 1.0, "Ready!"),
    ]
    
    for stage, progress, msg in stages:
        anim.update(LoadProgress(stage=stage, progress=progress, message=msg))
        time.sleep(2)
    
    anim.stop()
    print("\nDone!")

