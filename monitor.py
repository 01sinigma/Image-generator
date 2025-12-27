"""
Модуль для мониторинга процесса загрузки и использования GPU/CPU
Показывает прогресс, индикацию зависаний и использование ресурсов
"""

import time
import threading
import sys
from datetime import datetime
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None


class ProcessMonitor:
    """Мониторинг процесса с индикацией прогресса и проверкой зависаний"""
    
    def __init__(self, 
                 check_interval: float = 2.0,
                 timeout: float = 300.0,  # 5 минут по умолчанию
                 heartbeat_interval: float = 10.0):
        """
        Args:
            check_interval: Интервал проверки состояния (секунды)
            timeout: Таймаут для определения зависания (секунды)
            heartbeat_interval: Интервал heartbeat сообщений (секунды)
        """
        self.check_interval = check_interval
        self.timeout = timeout
        self.heartbeat_interval = heartbeat_interval
        self.last_activity = time.time()
        self.start_time = time.time()
        self.is_running = False
        self.monitor_thread = None
        self.status_callback: Optional[Callable[[str], None]] = None
        self.gpu_usage_history = []
        self.cpu_usage_history = []
        
    def set_status_callback(self, callback: Callable[[str], None]):
        """Установка callback для вывода статуса"""
        self.status_callback = callback
    
    def update_activity(self):
        """Обновление времени последней активности"""
        self.last_activity = time.time()
    
    def _log_status(self, message: str, level: str = "INFO"):
        """Логирование статуса с временной меткой"""
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_msg = f"[{timestamp}] [{elapsed:6.1f}s] {message}"
        
        if self.status_callback:
            self.status_callback(status_msg)
        else:
            if level == "INFO":
                logger.info(status_msg)
            elif level == "WARNING":
                logger.warning(status_msg)
            elif level == "ERROR":
                logger.error(status_msg)
    
    def _get_gpu_info(self) -> dict:
        """Получение информации о GPU"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"available": False}
        
        try:
            device = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)  # GB
            memory_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
            
            # Попытка получить использование GPU через nvidia-smi (если доступно)
            gpu_utilization = None
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    gpu_utilization = float(result.stdout.strip())
            except:
                pass
            
            return {
                "available": True,
                "device": device,
                "name": torch.cuda.get_device_name(device),
                "memory_allocated": memory_allocated,
                "memory_reserved": memory_reserved,
                "memory_total": memory_total,
                "memory_free": memory_total - memory_reserved,
                "utilization": gpu_utilization
            }
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def _get_cpu_info(self) -> dict:
        """Получение информации о CPU"""
        if not PSUTIL_AVAILABLE:
            return {"available": False}
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            return {
                "available": True,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3)
            }
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def _monitor_loop(self):
        """Основной цикл мониторинга"""
        last_heartbeat = time.time()
        last_gpu_check = time.time()
        last_cpu_check = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                time_since_activity = current_time - self.last_activity
                elapsed_total = current_time - self.start_time
                
                # Проверка зависания
                if time_since_activity > self.timeout:
                    self._log_status(
                        f"⚠️ ВНИМАНИЕ: Процесс не отвечает {time_since_activity:.1f} секунд (таймаут: {self.timeout:.1f}s)",
                        "WARNING"
                    )
                    self._log_status("Возможные причины: сетевые проблемы, зависание загрузки, нехватка памяти", "WARNING")
                    self._log_status("Рекомендация: проверьте интернет-соединение и доступное место на диске", "WARNING")
                
                # Heartbeat каждые heartbeat_interval секунд
                if current_time - last_heartbeat >= self.heartbeat_interval:
                    self._log_status("💓 Heartbeat: процесс активен...")
                    last_heartbeat = current_time
                
                # Проверка GPU каждые 5 секунд
                if current_time - last_gpu_check >= 5.0:
                    gpu_info = self._get_gpu_info()
                    if gpu_info.get("available"):
                        mem_used = gpu_info.get("memory_allocated", 0)
                        mem_total = gpu_info.get("memory_total", 0)
                        util = gpu_info.get("utilization")
                        if util is not None:
                            self._log_status(f"🎮 GPU: {gpu_info['name']} | Использование: {util:.1f}% | VRAM: {mem_used:.2f}/{mem_total:.2f} GB")
                        else:
                            self._log_status(f"🎮 GPU: {gpu_info['name']} | VRAM: {mem_used:.2f}/{mem_total:.2f} GB")
                    last_gpu_check = current_time
                
                # Проверка CPU каждые 5 секунд
                if current_time - last_cpu_check >= 5.0:
                    cpu_info = self._get_cpu_info()
                    if cpu_info.get("available"):
                        cpu_percent = cpu_info.get("cpu_percent", 0)
                        mem_percent = cpu_info.get("memory_percent", 0)
                        self._log_status(f"💻 CPU: {cpu_percent:.1f}% | RAM: {mem_percent:.1f}%")
                    last_cpu_check = current_time
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Ошибка в мониторе: {e}")
                time.sleep(self.check_interval)
    
    def start(self):
        """Запуск мониторинга"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = time.time()
        self.last_activity = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self._log_status("🚀 Мониторинг процесса запущен")
    
    def stop(self):
        """Остановка мониторинга"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        elapsed = time.time() - self.start_time
        self._log_status(f"✅ Мониторинг завершен (время работы: {elapsed:.1f}s)")
    
    def __enter__(self):
        """Контекстный менеджер: вход"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер: выход"""
        self.stop()


class ProgressTracker:
    """Отслеживание прогресса загрузки с визуализацией"""
    
    def __init__(self, total_steps: int = 5, description: str = "Загрузка"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.step_times = []
        self.last_update = time.time()
        
    def update(self, step: int = None, message: str = ""):
        """Обновление прогресса"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        step_elapsed = current_time - self.last_update
        
        self.step_times.append(step_elapsed)
        self.last_update = current_time
        
        # Визуализация прогресса
        progress_percent = (self.current_step / self.total_steps) * 100
        bar_length = 30
        filled = int(bar_length * self.current_step / self.total_steps)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        status = f"[{timestamp}] [{elapsed:6.1f}s] {self.description}: [{bar}] {progress_percent:5.1f}% ({self.current_step}/{self.total_steps})"
        
        if message:
            status += f" | {message}"
        
        # Предупреждение о медленном прогрессе
        if step_elapsed > 60 and self.current_step < self.total_steps:
            status += f" ⚠️ Шаг занял {step_elapsed:.1f}s"
        
        print(status, flush=True)
        logger.info(status)
    
    def finish(self, message: str = "Завершено"):
        """Завершение отслеживания"""
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime("%H:%M:%S")
        avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0
        
        status = f"[{timestamp}] [{elapsed:6.1f}s] ✅ {message} (среднее время шага: {avg_step_time:.1f}s)"
        print(status, flush=True)
        logger.info(status)

