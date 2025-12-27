# Настройка Image Generator для 8GB VRAM

Полное руководство по настройке системы для работы на видеокартах с 8GB VRAM (RTX 4060, RTX 3060 и т.д.)

## Быстрый старт

```bash
# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Скачивание моделей (автоопределение квантизации)
python download_all_models.py --model all

# 3. Запуск с оптимизацией памяти
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python app.py
```

---

## Шаг 1: Подготовка моделей

### 1.1. Qwen-Image-Edit-2511 GGUF

Для 8GB VRAM рекомендуется **Q4_K_M** квантизация:

```bash
python download_all_models.py --model qwen --quantization Q4_K_M
```

**Размер:** ~12GB
**Время загрузки:** 10-30 минут

### 1.2. RMBG-2.0 (удаление фона)

```bash
python download_all_models.py --model rmbg
```

**Размер:** ~500MB
**Скорость:** Очень быстрая обработка (1-2 секунды)

### 1.3. Lightning LoRA (ускорение)

LoRA необходимо скачать вручную:

1. Посетите [Civitai](https://civitai.com/) или [Hugging Face](https://huggingface.co/)
2. Найдите "Qwen Lightning" или "4-step LoRA"
3. Скачайте файл `.safetensors`
4. Поместите в `./models/lora/`

**Эффект:** Генерация за 4 шага вместо 40 (ускорение ~10x)

---

## Шаг 2: Конфигурация

### Использование оптимизированного конфига

```bash
# Скопируйте конфиг для 8GB
copy config_8gb_vram.yaml config.yaml
```

### Ключевые настройки в `config.yaml`:

```yaml
# Модели
models:
  qwen-image-edit:
    torch_dtype: "float16"        # Экономия памяти
    low_cpu_mem_usage: true       # Экономное использование RAM
    gguf:
      enabled: true
      quantization: "Q4_K_M"      # Оптимально для 8GB

# Устройство
device:
  enable_cpu_offload: true        # КРИТИЧНО для 8GB
  sequential_offload: true        # Экономит память

# Выполнение
execution:
  quantization: "4bit"            # GGUF квантизация
  low_vram_mode: true
  cpu_offload: true
```

---

## Шаг 3: Установка дополнительных зависимостей

### Для LoRA:
```bash
pip install peft>=0.10.0
```

### Для ONNX ускорения RMBG-2.0:
```bash
# Для CUDA 11.x
pip install onnxruntime-gpu>=1.17.0

# Для CUDA 12.x
pip install onnxruntime-gpu>=1.18.0
```

### Для GGUF через llama-cpp-python:
```bash
# Windows с CUDA
set CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python
```

---

## Шаг 4: Режимы работы

### Standard (по умолчанию)
- Обычное редактирование
- 40 шагов
- Качественный результат

### Fast (с Lightning LoRA)
- 4 шага
- Быстрый результат (30-40 секунд)
- Требует Lightning LoRA

### Uncensored
- Снятие ограничений на контент
- Измененный системный промпт

Настройка в `config.yaml`:
```yaml
pipeline:
  mode: "standard"  # или "uncensored", "fast"
```

---

## Шаг 5: Использование Pipeline

### Удаление фона → Редактирование → Замена фона

```python
from image_pipeline import ImagePipeline, PipelineMode

# Создание пайплайна
pipeline = ImagePipeline(config)
pipeline.initialize()

# Обработка изображения
result, metadata = pipeline.process(
    image="input.png",
    prompt="Place this object on a cyberpunk street",
    remove_background=True,
    new_background="cyberpunk_purple",  # пресет
    use_lightning_lora=True,
)
```

---

## Шаг 6: Возможные проблемы и решения

### "CUDA out of memory"
```yaml
# В config.yaml:
device:
  enable_cpu_offload: true
  sequential_offload: true

generation:
  default_height: 384
  default_width: 384
```

### Медленная генерация
1. Используйте Lightning LoRA
2. Уменьшите количество шагов
3. Уменьшите разрешение

### Модель не загружается
```bash
# Проверка полноты модели
python -c "from model_checker import ModelCompletenessChecker; c = ModelCompletenessChecker('Qwen/Qwen-Image-Edit-2511'); print(c.check_model_completeness())"

# Докачка недостающих файлов
python download_all_models.py --model qwen
```

### GGUF не работает
GGUF требует `llama-cpp-python` с CUDA:
```bash
set CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

---

## Оптимальные настройки для RTX 4060 (8GB)

| Параметр | Значение |
|----------|----------|
| Квантизация | Q4_K_M |
| Разрешение | 512x512 |
| Шаги (с LoRA) | 4 |
| Шаги (без LoRA) | 20-30 |
| CPU Offload | Да |
| Sequential Offload | Да |
| torch_dtype | float16 |

---

## Структура файлов

```
Image generator/
├── models/
│   ├── qwen-image-edit-gguf/
│   │   └── qwen-image-edit-2511-Q4_K_M.gguf
│   ├── rmbg/
│   │   └── (модель RMBG-2.0)
│   └── lora/
│       └── (ваши LoRA файлы)
├── app.py                    # Веб-интерфейс
├── config.yaml               # Конфигурация
├── config_8gb_vram.yaml      # Конфиг для 8GB
├── background_remover.py     # Удаление фона
├── lora_manager.py           # Управление LoRA
├── image_pipeline.py         # Pipeline обработки
├── gguf_loader.py            # Загрузка GGUF
└── download_all_models.py    # Скачивание моделей
```

---

## Часто задаваемые вопросы

**Q: Сколько времени занимает генерация?**
- С Lightning LoRA: 30-40 секунд
- Без LoRA: 2-5 минут

**Q: Можно ли использовать модель на CPU?**
Да, но очень медленно (10-30 минут на изображение)

**Q: Какой размер модели нужен?**
- GGUF Q4_K_M: ~12GB на диске
- В VRAM: ~6-7GB (с offload остальное в RAM)

**Q: Как улучшить качество?**
1. Увеличьте количество шагов
2. Используйте Q5_K_M или Q6_K квантизацию
3. Увеличьте разрешение (если хватает памяти)

