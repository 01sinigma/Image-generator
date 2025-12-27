"""
ComfyUI API Integration
Интеграция с ComfyUI для использования GGUF моделей

Позволяет:
- Отправлять запросы на генерацию в ComfyUI
- Получать результаты
- Использовать GGUF модели через ComfyUI backend
"""

import json
import urllib.request
import urllib.parse
import io
import time
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

logger = logging.getLogger(__name__)


class ComfyUIClient:
    """
    Клиент для работы с ComfyUI API
    
    Требует запущенный ComfyUI с флагом --enable-cors-header
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8188,
        client_id: Optional[str] = None
    ):
        """
        Инициализация клиента
        
        Args:
            host: Хост ComfyUI сервера
            port: Порт ComfyUI сервера
            client_id: Уникальный ID клиента
        """
        self.host = host
        self.port = port
        self.client_id = client_id or str(uuid.uuid4())
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws?clientId={self.client_id}"
        
        self._ws = None
    
    def is_available(self) -> bool:
        """Проверка доступности ComfyUI"""
        try:
            response = urllib.request.urlopen(f"{self.base_url}/system_stats", timeout=5)
            return response.status == 200
        except Exception as e:
            logger.debug(f"ComfyUI недоступен: {e}")
            return False
    
    def get_system_stats(self) -> Optional[Dict[str, Any]]:
        """Получение статистики системы"""
        try:
            response = urllib.request.urlopen(f"{self.base_url}/system_stats")
            return json.loads(response.read())
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return None
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Получение списка доступных моделей"""
        try:
            response = urllib.request.urlopen(f"{self.base_url}/object_info")
            data = json.loads(response.read())
            
            models = {
                'checkpoints': [],
                'loras': [],
                'unet': [],
                'vae': [],
            }
            
            # Извлечение списков моделей из object_info
            if 'CheckpointLoaderSimple' in data:
                models['checkpoints'] = data['CheckpointLoaderSimple']['input']['required'].get('ckpt_name', [[]])[0]
            
            if 'UnetLoaderGGUF' in data:
                models['unet'] = data['UnetLoaderGGUF']['input']['required'].get('unet_name', [[]])[0]
            
            return models
            
        except Exception as e:
            logger.error(f"Ошибка получения моделей: {e}")
            return {}
    
    def queue_prompt(self, workflow: Dict[str, Any]) -> Optional[str]:
        """
        Отправка workflow в очередь
        
        Args:
            workflow: Словарь с workflow
        
        Returns:
            prompt_id или None при ошибке
        """
        try:
            data = json.dumps({
                "prompt": workflow,
                "client_id": self.client_id
            }).encode('utf-8')
            
            request = urllib.request.Request(
                f"{self.base_url}/prompt",
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            response = urllib.request.urlopen(request)
            result = json.loads(response.read())
            
            return result.get('prompt_id')
            
        except Exception as e:
            logger.error(f"Ошибка отправки в очередь: {e}")
            return None
    
    def get_history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Получение истории выполнения"""
        try:
            response = urllib.request.urlopen(f"{self.base_url}/history/{prompt_id}")
            return json.loads(response.read())
        except Exception as e:
            logger.error(f"Ошибка получения истории: {e}")
            return None
    
    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> Optional[Image.Image]:
        """
        Получение изображения с сервера
        
        Args:
            filename: Имя файла
            subfolder: Подпапка
            folder_type: Тип папки (output, input, temp)
        
        Returns:
            PIL Image или None
        """
        try:
            params = urllib.parse.urlencode({
                'filename': filename,
                'subfolder': subfolder,
                'type': folder_type
            })
            
            response = urllib.request.urlopen(f"{self.base_url}/view?{params}")
            image_data = response.read()
            
            return Image.open(io.BytesIO(image_data))
            
        except Exception as e:
            logger.error(f"Ошибка получения изображения: {e}")
            return None
    
    def upload_image(self, image: Image.Image, name: str = "input.png") -> Optional[str]:
        """
        Загрузка изображения на сервер
        
        Args:
            image: PIL Image
            name: Имя файла
        
        Returns:
            Имя загруженного файла или None
        """
        try:
            # Конвертация в bytes
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_data = buffer.getvalue()
            
            # Multipart form data
            boundary = '----WebKitFormBoundary' + str(uuid.uuid4()).replace('-', '')
            
            body = (
                f'--{boundary}\r\n'
                f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'
                f'Content-Type: image/png\r\n\r\n'
            ).encode() + image_data + f'\r\n--{boundary}--\r\n'.encode()
            
            request = urllib.request.Request(
                f"{self.base_url}/upload/image",
                data=body,
                headers={
                    'Content-Type': f'multipart/form-data; boundary={boundary}'
                }
            )
            
            response = urllib.request.urlopen(request)
            result = json.loads(response.read())
            
            return result.get('name')
            
        except Exception as e:
            logger.error(f"Ошибка загрузки изображения: {e}")
            return None
    
    def wait_for_completion(
        self,
        prompt_id: str,
        timeout: float = 300.0,
        poll_interval: float = 1.0
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Ожидание завершения генерации
        
        Args:
            prompt_id: ID запроса
            timeout: Таймаут в секундах
            poll_interval: Интервал опроса
        
        Returns:
            (success, result_data)
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)
            
            if history and prompt_id in history:
                return True, history[prompt_id]
            
            time.sleep(poll_interval)
        
        logger.error(f"Таймаут ожидания завершения: {timeout}s")
        return False, None
    
    def generate_with_gguf(
        self,
        prompt: str,
        negative_prompt: str = "",
        model_name: str = "qwen-image-edit-2511-Q4_K_M.gguf",
        width: int = 512,
        height: int = 512,
        steps: int = 4,
        cfg: float = 1.0,
        seed: int = -1,
        timeout: float = 300.0
    ) -> Optional[Image.Image]:
        """
        Генерация изображения через GGUF модель
        
        Args:
            prompt: Позитивный промпт
            negative_prompt: Негативный промпт
            model_name: Имя GGUF модели
            width: Ширина
            height: Высота
            steps: Количество шагов
            cfg: CFG scale
            seed: Seed (-1 для случайного)
            timeout: Таймаут
        
        Returns:
            PIL Image или None
        """
        if not self.is_available():
            logger.error("ComfyUI недоступен!")
            return None
        
        # Workflow для GGUF генерации с Qwen
        # Qwen-Image-Edit требует специальный text encoder который загружается вместе с моделью
        # Используем специальные ноды: TextEncodeQwenImageEdit и EmptyQwenImageLayeredLatentImage
        
        # ВАЖНО: Qwen GGUF через ComfyUI требует скачать text encoder отдельно
        # https://huggingface.co/Qwen/Qwen-Image-Edit-2511/tree/main/text_encoder
        # или использовать специальный workflow с CLIPLoader type="qwen_image"
        
        workflow = {
            "1": {
                "class_type": "UnetLoaderGGUF",
                "inputs": {
                    "unet_name": model_name
                }
            },
            # Используем CLIPLoader с типом qwen_image
            # Требуется файл text_encoder из Qwen модели
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "clip_l.safetensors",  # Пробуем clip_l
                    "type": "qwen_image"
                }
            },
            # Специальный латент для Qwen с слоями
            "3": {
                "class_type": "EmptyQwenImageLayeredLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "layers": 1,
                    "batch_size": 1
                }
            },
            # Используем специальный Qwen text encoder
            "4": {
                "class_type": "TextEncodeQwenImageEdit",
                "inputs": {
                    "prompt": prompt,
                    "clip": ["2", 0]
                }
            },
            "5": {
                "class_type": "TextEncodeQwenImageEdit",
                "inputs": {
                    "prompt": negative_prompt or "blurry, low quality, bad anatomy",
                    "clip": ["2", 0]
                }
            },
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": ["3", 0],
                    "seed": seed if seed >= 0 else int(time.time()),
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0
                }
            },
            # Используем pixel_space VAE для Qwen
            "7": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": "pixel_space"  # Qwen использует pixel space VAE
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["6", 0],
                    "vae": ["7", 0]
                }
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["8", 0],
                    "filename_prefix": "ComfyUI_Qwen_GGUF"
                }
            }
        }
        
        logger.info(f"Отправка запроса в ComfyUI...")
        logger.info(f"  Модель: {model_name}")
        logger.info(f"  Размер: {width}x{height}")
        logger.info(f"  Шаги: {steps}")
        
        # Отправка в очередь
        prompt_id = self.queue_prompt(workflow)
        if not prompt_id:
            logger.error("Не удалось отправить запрос")
            return None
        
        logger.info(f"Запрос в очереди: {prompt_id}")
        
        # Ожидание результата
        success, result = self.wait_for_completion(prompt_id, timeout)
        
        if not success or not result:
            logger.error("Генерация не завершена")
            return None
        
        # Получение изображения
        try:
            outputs = result.get('outputs', {})
            for node_id, output in outputs.items():
                if 'images' in output:
                    for img_info in output['images']:
                        image = self.get_image(
                            img_info['filename'],
                            img_info.get('subfolder', ''),
                            img_info.get('type', 'output')
                        )
                        if image:
                            logger.info("✅ Изображение получено!")
                            return image
        except Exception as e:
            logger.error(f"Ошибка получения результата: {e}")
        
        return None


class ComfyUIGGUFGenerator:
    """
    Генератор изображений через ComfyUI с GGUF моделями
    
    Совместим с интерфейсом BaseImageGenerator
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8188,
        model_name: str = "qwen-image-edit-2511-Q4_K_M.gguf"
    ):
        self.client = ComfyUIClient(host, port)
        self.model_name = model_name
        self.device = "comfyui"
    
    def is_available(self) -> bool:
        """Проверка доступности"""
        return self.client.is_available()
    
    def generate(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> Optional[Image.Image]:
        """
        Генерация изображения
        
        Args:
            prompt: Текстовый промпт
            height: Высота
            width: Ширина
            num_inference_steps: Количество шагов
            guidance_scale: CFG scale
            seed: Seed
            save_path: Путь для сохранения
        
        Returns:
            PIL Image или None
        """
        image = self.client.generate_with_gguf(
            prompt=prompt,
            model_name=self.model_name,
            width=width,
            height=height,
            steps=num_inference_steps,
            cfg=guidance_scale,
            seed=seed if seed is not None else -1,
        )
        
        if image and save_path:
            image.save(save_path)
            logger.info(f"Изображение сохранено: {save_path}")
        
        return image
    
    def edit(
        self,
        images: List[Image.Image],
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        true_cfg_scale: float = 4.0,
        seed: Optional[int] = None,
        save_path: Optional[str] = None,
        **kwargs
    ) -> Optional[Image.Image]:
        """
        Редактирование изображения
        
        Требует специальный workflow для редактирования
        """
        if not images:
            return self.generate(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                save_path=save_path
            )
        
        # Загрузка входного изображения
        input_image = images[0]
        uploaded_name = self.client.upload_image(input_image, "input_edit.png")
        
        if not uploaded_name:
            logger.error("Не удалось загрузить изображение")
            return None
        
        # TODO: Создать workflow для редактирования с GGUF
        # Пока используем генерацию с промптом
        logger.warning("Редактирование через GGUF пока использует генерацию с нуля")
        
        return self.generate(
            prompt=prompt,
            width=input_image.width,
            height=input_image.height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            save_path=save_path
        )


def check_comfyui_status(host: str = "127.0.0.1", port: int = 8188) -> Dict[str, Any]:
    """
    Проверка статуса ComfyUI
    
    Returns:
        Словарь с информацией о статусе
    """
    client = ComfyUIClient(host, port)
    
    status = {
        'available': client.is_available(),
        'host': host,
        'port': port,
        'url': f"http://{host}:{port}",
    }
    
    if status['available']:
        stats = client.get_system_stats()
        if stats:
            status['system'] = stats
        
        models = client.get_available_models()
        if models:
            status['models'] = models
    
    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("ComfyUI API Test")
    print("="*60)
    
    status = check_comfyui_status()
    
    print(f"\nСтатус ComfyUI:")
    print(f"  Доступен: {status['available']}")
    print(f"  URL: {status['url']}")
    
    if status['available']:
        print(f"\nДоступные модели:")
        for model_type, models in status.get('models', {}).items():
            if models:
                print(f"  {model_type}:")
                for m in models[:5]:
                    print(f"    - {m}")
    else:
        print(f"\n⚠️ ComfyUI недоступен!")
        print(f"Запустите ComfyUI: start_comfyui.bat")

