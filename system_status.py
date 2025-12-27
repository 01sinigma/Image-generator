"""
Модуль для проверки статуса системы и функций
"""

import torch
import logging
from typing import Dict, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class SystemStatusChecker:
    """Проверка статуса системы и доступных функций"""
    
    def __init__(self):
        self.status = {}
        self.functions_status = {}
    
    def check_system(self) -> Dict:
        """Проверка системных компонентов"""
        status = {
            "cuda": self._check_cuda(),
            "gpu": self._check_gpu(),
            "models": self._check_models(),
            "config": self._check_config(),
            "dependencies": self._check_dependencies()
        }
        self.status = status
        return status
    
    def check_functions(self, generators: dict) -> Dict:
        """Проверка доступности функций для загруженных моделей"""
        functions = {}
        
        for model_type, generator in generators.items():
            model_functions = {
                "generate": hasattr(generator, 'generate') and callable(getattr(generator, 'generate', None)),
                "edit": hasattr(generator, 'edit') and callable(getattr(generator, 'edit', None)),
                "loaded": generator is not None and hasattr(generator, 'pipe') and generator.pipe is not None
            }
            functions[model_type] = model_functions
        
        self.functions_status = functions
        return functions
    
    def _check_cuda(self) -> Dict:
        """Проверка CUDA"""
        try:
            available = torch.cuda.is_available()
            if available:
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                
                # Проверка памяти
                memory_total = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
                memory_allocated = torch.cuda.memory_allocated(current_device) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(current_device) / (1024**3)
                
                return {
                    "available": True,
                    "status": "✅ Работает",
                    "device_count": device_count,
                    "current_device": current_device,
                    "device_name": device_name,
                    "memory_total_gb": round(memory_total, 2),
                    "memory_allocated_gb": round(memory_allocated, 2),
                    "memory_reserved_gb": round(memory_reserved, 2),
                    "memory_free_gb": round(memory_total - memory_reserved, 2),
                    "message": f"CUDA доступна: {device_name} ({memory_total:.1f}GB VRAM)"
                }
            else:
                return {
                    "available": False,
                    "status": "❌ Недоступна",
                    "message": "CUDA недоступна, будет использован CPU"
                }
        except Exception as e:
            logger.error(f"Ошибка проверки CUDA: {e}")
            return {
                "available": False,
                "status": "❌ Ошибка",
                "message": f"Ошибка проверки CUDA: {e}"
            }
    
    def _check_gpu(self) -> Dict:
        """Проверка GPU через nvidia-smi"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=3
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(', ')
                    return {
                        "available": True,
                        "status": "✅ Работает",
                        "name": parts[0],
                        "memory_total_mb": parts[1],
                        "memory_used_mb": parts[2],
                        "utilization_percent": parts[3],
                        "message": f"GPU: {parts[0]}, Использование: {parts[3]}%"
                    }
            
            return {
                "available": False,
                "status": "⚠️ Недоступен",
                "message": "nvidia-smi недоступен"
            }
        except FileNotFoundError:
            return {
                "available": False,
                "status": "⚠️ Недоступен",
                "message": "nvidia-smi не найден (возможно, не установлены драйверы NVIDIA)"
            }
        except Exception as e:
            logger.debug(f"Не удалось проверить GPU через nvidia-smi: {e}")
            return {
                "available": None,
                "status": "❓ Неизвестно",
                "message": "Не удалось проверить GPU"
            }
    
    def _check_models(self) -> Dict:
        """Проверка доступности моделей"""
        try:
            from models import ModelFactory
            available_models = ModelFactory.get_available_models()
            return {
                "available": True,
                "status": "✅ Доступны",
                "models": available_models,
                "count": len(available_models),
                "message": f"Доступно моделей: {len(available_models)} ({', '.join(available_models)})"
            }
        except Exception as e:
            logger.error(f"Ошибка проверки моделей: {e}")
            return {
                "available": False,
                "status": "❌ Ошибка",
                "message": f"Ошибка проверки моделей: {e}"
            }
    
    def _check_config(self) -> Dict:
        """Проверка конфигурации"""
        try:
            import yaml
            config_path = Path("config.yaml")
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # Проверка ключевых настроек
                device_config = config.get('device', {})
                enable_cpu_offload = device_config.get('enable_cpu_offload', False)
                sequential_offload = device_config.get('sequential_offload', False)
                
                issues = []
                if not enable_cpu_offload:
                    issues.append("CPU offload отключен (рекомендуется для 8GB VRAM)")
                
                return {
                    "available": True,
                    "status": "✅ Загружена",
                    "cpu_offload": enable_cpu_offload,
                    "sequential_offload": sequential_offload,
                    "issues": issues,
                    "message": f"Конфигурация загружена" + (f" ({len(issues)} предупреждений)" if issues else "")
                }
            else:
                return {
                    "available": False,
                    "status": "❌ Не найдена",
                    "message": "config.yaml не найден"
                }
        except Exception as e:
            logger.error(f"Ошибка проверки конфигурации: {e}")
            return {
                "available": False,
                "status": "❌ Ошибка",
                "message": f"Ошибка загрузки конфигурации: {e}"
            }
    
    def _check_dependencies(self) -> Dict:
        """Проверка зависимостей"""
        dependencies = {
            "torch": self._check_module("torch"),
            "diffusers": self._check_module("diffusers"),
            "transformers": self._check_module("transformers"),
            "gradio": self._check_module("gradio"),
            "pillow": self._check_module("PIL"),
            "yaml": self._check_module("yaml"),
        }
        
        available_count = sum(1 for dep in dependencies.values() if dep["available"])
        total_count = len(dependencies)
        
        return {
            "dependencies": dependencies,
            "available_count": available_count,
            "total_count": total_count,
            "status": "✅ Все установлены" if available_count == total_count else f"⚠️ {available_count}/{total_count}",
            "message": f"Зависимости: {available_count}/{total_count} установлены"
        }
    
    def _check_module(self, module_name: str) -> Dict:
        """Проверка наличия модуля"""
        try:
            if module_name == "PIL":
                import PIL
                version = PIL.__version__
            elif module_name == "yaml":
                import yaml
                version = getattr(yaml, '__version__', 'unknown')
            else:
                module = __import__(module_name)
                version = getattr(module, '__version__', 'unknown')
            
            return {
                "available": True,
                "status": "✅ Установлен",
                "version": version,
                "message": f"{module_name} {version}"
            }
        except ImportError:
            return {
                "available": False,
                "status": "❌ Не установлен",
                "version": None,
                "message": f"{module_name} не установлен"
            }
        except Exception as e:
            return {
                "available": False,
                "status": "❌ Ошибка",
                "version": None,
                "message": f"Ошибка проверки {module_name}: {e}"
            }
    
    def get_status_summary(self) -> str:
        """Получить текстовую сводку статуса"""
        lines = []
        lines.append("📊 Статус системы:")
        lines.append("")
        
        if "cuda" in self.status:
            cuda = self.status["cuda"]
            lines.append(f"  CUDA: {cuda.get('status', '❓')} - {cuda.get('message', '')}")
        
        if "gpu" in self.status:
            gpu = self.status["gpu"]
            lines.append(f"  GPU: {gpu.get('status', '❓')} - {gpu.get('message', '')}")
        
        if "models" in self.status:
            models = self.status["models"]
            lines.append(f"  Модели: {models.get('status', '❓')} - {models.get('message', '')}")
        
        if "config" in self.status:
            config = self.status["config"]
            lines.append(f"  Конфигурация: {config.get('status', '❓')} - {config.get('message', '')}")
        
        if "dependencies" in self.status:
            deps = self.status["dependencies"]
            lines.append(f"  Зависимости: {deps.get('status', '❓')} - {deps.get('message', '')}")
        
        lines.append("")
        lines.append("🔧 Функции:")
        
        for model_type, functions in self.functions_status.items():
            lines.append(f"  {model_type}:")
            for func_name, available in functions.items():
                status = "✅" if available else "❌"
                lines.append(f"    {status} {func_name}: {'Доступна' if available else 'Недоступна'}")
        
        return "\n".join(lines)
    
    def get_status_html(self) -> str:
        """Получить HTML представление статуса"""
        html_parts = []
        html_parts.append("<div style='font-family: monospace; font-size: 12px;'>")
        html_parts.append("<h3>📊 Статус системы</h3>")
        
        # Системные компоненты
        for component_name, component_status in self.status.items():
            status_icon = component_status.get('status', '❓')
            message = component_status.get('message', '')
            html_parts.append(f"<p><strong>{component_name.upper()}:</strong> {status_icon} {message}</p>")
        
        # Функции
        html_parts.append("<h3>🔧 Функции моделей</h3>")
        for model_type, functions in self.functions_status.items():
            html_parts.append(f"<p><strong>{model_type}:</strong></p>")
            html_parts.append("<ul>")
            for func_name, available in functions.items():
                status_icon = "✅" if available else "❌"
                status_text = "Доступна" if available else "Недоступна"
                html_parts.append(f"<li>{status_icon} <code>{func_name}</code>: {status_text}</li>")
            html_parts.append("</ul>")
        
        html_parts.append("</div>")
        return "".join(html_parts)

