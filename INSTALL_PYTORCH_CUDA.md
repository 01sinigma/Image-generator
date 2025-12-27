# Установка PyTorch с поддержкой CUDA

## Проблема
У вас установлена CPU-версия PyTorch (`2.9.1+cpu`), хотя у вас есть NVIDIA GeForce RTX 4060 с 8GB VRAM.

## Решение: Установить PyTorch с CUDA

### Вариант 1: Через официальный сайт PyTorch (рекомендуется)

1. Перейдите на https://pytorch.org/get-started/locally/
2. Выберите:
   - **PyTorch Build**: Stable (2.7.1)
   - **Your OS**: Windows
   - **Package**: Pip
   - **Language**: Python
   - **Compute Platform**: CUDA 11.8 или CUDA 12.1 (ваш драйвер поддерживает CUDA 12.9)

3. Скопируйте команду и выполните в терминале

### Вариант 2: Прямая установка через pip

Для CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Для CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Внимание:** Файл весит ~2.8GB, установка может занять время (10-30 минут в зависимости от скорости интернета).

### Вариант 3: Через conda (если используете conda)

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## После установки

Проверьте, что CUDA работает:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Должно вывести:
```
CUDA available: True
Device: NVIDIA GeForce RTX 4060 ...
```

## Ваша конфигурация

- **GPU**: NVIDIA GeForce RTX 4060
- **VRAM**: 8GB (идеально для Z-Image-Turbo с оптимизациями!)
- **Драйвер**: 576.80 (CUDA 12.9)
- **Рекомендуемая версия PyTorch**: с CUDA 11.8 или 12.1

## После установки PyTorch с CUDA

1. Обновите config.yaml:
   ```yaml
   device:
     type: "cuda"  # или "auto"
     enable_cpu_offload: true  # Теперь будет работать!
   ```

2. Запустите снова:
   ```bash
   python app.py
   ```

Модель будет использовать ваш GPU и работать намного быстрее!

