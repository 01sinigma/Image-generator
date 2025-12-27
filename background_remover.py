"""
Background Remover
Удаление фона с изображений с использованием RMBG-2.0

RMBG-2.0 - легковесная модель для удаления фона, 
оптимизированная для работы на GPU с ограниченной памятью
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, Tuple
from PIL import Image
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Попытка импорта onnxruntime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime не установлен. Установите через: pip install onnxruntime-gpu")

# Попытка импорта transformers для RMBG-2.0
try:
    from transformers import AutoModelForImageSegmentation, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers не установлен. Установите через: pip install transformers")


class BackgroundRemover:
    """
    Удаление фона с изображений с использованием RMBG-2.0
    
    Поддерживает:
    - Работу через ONNX (быстрее, меньше памяти)
    - Работу через Transformers (проще настроить)
    """
    
    MODEL_REPO = "briaai/RMBG-2.0"
    
    def __init__(
        self,
        device: str = "cuda",
        precision: str = "fp16",
        use_onnx: bool = False,
        model_dir: str = "./models/rmbg"
    ):
        """
        Инициализация
        
        Args:
            device: Устройство для вычислений ('cuda' или 'cpu')
            precision: Точность ('fp16' или 'fp32')
            use_onnx: Использовать ONNX (быстрее, меньше памяти)
            model_dir: Директория для кэширования модели
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.precision = precision
        self.use_onnx = use_onnx and ONNX_AVAILABLE
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.processor = None
        self.onnx_session = None
        
        logger.info(f"BackgroundRemover инициализирован: device={self.device}, precision={precision}")
    
    def load_model(self) -> bool:
        """
        Загрузка модели RMBG-2.0
        
        Returns:
            True если загрузка успешна
        """
        if self.use_onnx:
            return self._load_onnx_model()
        else:
            return self._load_transformers_model()
    
    def _load_transformers_model(self) -> bool:
        """Загрузка модели через transformers"""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers не установлен")
            return False
        
        try:
            logger.info(f"Загрузка RMBG-2.0 через transformers...")
            
            # Определение dtype
            if self.precision == "fp16" and self.device == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32
            
            # Загрузка модели
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.MODEL_REPO,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Загрузка процессора
            self.processor = AutoProcessor.from_pretrained(
                self.MODEL_REPO,
                trust_remote_code=True,
            )
            
            logger.info("RMBG-2.0 успешно загружена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке RMBG-2.0: {e}")
            return False
    
    def _load_onnx_model(self) -> bool:
        """Загрузка модели через ONNX"""
        if not ONNX_AVAILABLE:
            logger.error("onnxruntime не установлен")
            return False
        
        try:
            onnx_path = self.model_dir / "rmbg-2.0.onnx"
            
            if not onnx_path.exists():
                logger.info("ONNX модель не найдена, конвертация из PyTorch...")
                # Здесь можно добавить конвертацию, но для простоты используем transformers fallback
                logger.warning("Переключение на transformers загрузчик...")
                self.use_onnx = False
                return self._load_transformers_model()
            
            # Настройка ONNX провайдеров
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
            
            self.onnx_session = ort.InferenceSession(
                str(onnx_path),
                providers=providers
            )
            
            logger.info("ONNX модель RMBG-2.0 загружена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке ONNX модели: {e}")
            logger.warning("Переключение на transformers загрузчик...")
            self.use_onnx = False
            return self._load_transformers_model()
    
    def remove_background(
        self,
        image: Union[Image.Image, str, np.ndarray],
        return_mask: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
        """
        Удаление фона с изображения
        
        Args:
            image: Входное изображение (PIL, путь или numpy array)
            return_mask: Возвращать ли маску вместе с результатом
        
        Returns:
            Изображение с прозрачным фоном (и маска, если return_mask=True)
        """
        # Загрузка модели если не загружена
        if self.model is None and self.onnx_session is None:
            if not self.load_model():
                raise RuntimeError("Не удалось загрузить модель RMBG-2.0")
        
        # Преобразование входа в PIL Image
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Убедимся, что изображение в RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_size = image.size
        
        try:
            if self.use_onnx and self.onnx_session is not None:
                mask = self._process_onnx(image)
            else:
                mask = self._process_transformers(image)
            
            # Изменение размера маски до оригинального
            mask = mask.resize(original_size, Image.BILINEAR)
            
            # Создание изображения с прозрачным фоном
            result = image.copy()
            result.putalpha(mask)
            
            if return_mask:
                return result, mask
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при удалении фона: {e}")
            raise
    
    def _process_transformers(self, image: Image.Image) -> Image.Image:
        """Обработка через transformers"""
        # Предобработка
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Если fp16
        if self.precision == "fp16" and self.device == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
        
        # Инференс
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Постобработка
        mask = outputs.logits.squeeze().cpu()
        mask = torch.sigmoid(mask)
        mask = (mask * 255).numpy().astype(np.uint8)
        
        return Image.fromarray(mask, mode='L')
    
    def _process_onnx(self, image: Image.Image) -> Image.Image:
        """Обработка через ONNX"""
        # Предобработка для ONNX
        input_size = (1024, 1024)
        resized = image.resize(input_size)
        
        # Нормализация
        img_array = np.array(resized).astype(np.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5  # [-1, 1]
        img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
        img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
        
        # Инференс
        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name
        
        result = self.onnx_session.run(
            [output_name],
            {input_name: img_array}
        )[0]
        
        # Постобработка
        mask = result.squeeze()
        mask = 1 / (1 + np.exp(-mask))  # sigmoid
        mask = (mask * 255).astype(np.uint8)
        
        return Image.fromarray(mask, mode='L')
    
    def replace_background(
        self,
        image: Union[Image.Image, str],
        background: Union[Image.Image, str, Tuple[int, int, int]]
    ) -> Image.Image:
        """
        Замена фона на изображении
        
        Args:
            image: Входное изображение
            background: Новый фон (изображение, путь или цвет RGB)
        
        Returns:
            Изображение с новым фоном
        """
        # Удаление фона
        foreground, mask = self.remove_background(image, return_mask=True)
        
        # Преобразование входа в PIL Image
        if isinstance(image, str):
            image = Image.open(image)
        
        # Создание фона
        if isinstance(background, tuple):
            # Однотонный цвет
            bg = Image.new('RGB', image.size, background)
        elif isinstance(background, str):
            bg = Image.open(background).convert('RGB').resize(image.size)
        else:
            bg = background.convert('RGB').resize(image.size)
        
        # Композиция
        result = Image.composite(
            foreground.convert('RGB'),
            bg,
            mask
        )
        
        return result
    
    def unload_model(self):
        """Выгрузка модели для освобождения памяти"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        if self.onnx_session is not None:
            del self.onnx_session
            self.onnx_session = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("RMBG-2.0 модель выгружена")


class BackgroundPresets:
    """Пресеты фонов для замены"""
    
    PRESETS = {
        'white_studio': (255, 255, 255),
        'black_studio': (0, 0, 0),
        'gray_studio': (128, 128, 128),
        'blue_sky': (135, 206, 235),
        'sunset_orange': (255, 100, 50),
        'neon_pink': (255, 20, 147),
        'cyberpunk_purple': (75, 0, 130),
        'forest_green': (34, 139, 34),
    }
    
    @classmethod
    def get_preset(cls, name: str) -> Optional[Tuple[int, int, int]]:
        """Получение пресета по имени"""
        return cls.PRESETS.get(name)
    
    @classmethod
    def list_presets(cls) -> list:
        """Список доступных пресетов"""
        return list(cls.PRESETS.keys())


def download_rmbg_model(output_dir: str = "./models/rmbg") -> bool:
    """
    Скачивание модели RMBG-2.0
    
    Args:
        output_dir: Директория для сохранения
    
    Returns:
        True если успешно
    """
    try:
        from huggingface_hub import snapshot_download
        
        logger.info("Скачивание RMBG-2.0...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        snapshot_download(
            repo_id="briaai/RMBG-2.0",
            local_dir=output_dir,
            resume_download=True,
        )
        
        logger.info(f"RMBG-2.0 скачана в {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при скачивании RMBG-2.0: {e}")
        return False


if __name__ == "__main__":
    # Тестирование
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Background Remover Test")
    print("=" * 60)
    
    # Проверка зависимостей
    print(f"ONNX доступен: {ONNX_AVAILABLE}")
    print(f"Transformers доступен: {TRANSFORMERS_AVAILABLE}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"VRAM: {vram:.2f}GB")
    
    print(f"\nДоступные пресеты фона:")
    for preset in BackgroundPresets.list_presets():
        color = BackgroundPresets.get_preset(preset)
        print(f"  - {preset}: RGB{color}")

