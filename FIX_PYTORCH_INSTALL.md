# Решение проблем с установкой PyTorch

## Проблемы

1. **Недостаточно места на диске C:** (~2.8GB свободно, нужно ~2.8GB для PyTorch)
2. **PyTorch установился неполно** (поврежден после прерывания установки)

## Решения

### Вариант 1: Освободить место на диске C: (рекомендуется)

Нужно освободить минимум **3-4 GB** на диске C:

1. **Очистка временных файлов:**
   ```powershell
   # Очистка кэша pip
   pip cache purge
   
   # Очистка временных файлов Windows
   Remove-Item -Path "$env:TEMP\*" -Recurse -Force -ErrorAction SilentlyContinue
   ```

2. **Очистка кэша Python:**
   ```powershell
   # Кэш pip
   pip cache purge
   
   # Кэш __pycache__
   Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
   ```

3. **Удаление ненужных программ/файлов**

4. **Очистка корзины**

### Вариант 2: Установить PyTorch на другой диск

Если есть другой диск с достаточным местом:

1. **Создать виртуальное окружение на другом диске:**
   ```powershell
   # Например, на диске D:
   D:\Python\python.exe -m venv D:\venv_image_gen
   D:\venv_image_gen\Scripts\Activate.ps1
   ```

2. **Или изменить кэш pip:**
   ```powershell
   # Установить переменную окружения для кэша pip
   $env:PIP_CACHE_DIR = "D:\pip_cache"
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Вариант 3: Установка через conda (если установлен)

Conda может использовать другой диск для пакетов:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## После освобождения места

1. **Полностью удалить поврежденный PyTorch:**
   ```powershell
   pip uninstall torch torchvision -y
   # Удалить вручную если остались файлы
   Remove-Item -Path "$env:LOCALAPPDATA\Programs\Python\Python313\Lib\site-packages\torch" -Recurse -Force -ErrorAction SilentlyContinue
   ```

2. **Очистить кэш pip:**
   ```powershell
   pip cache purge
   ```

3. **Установить PyTorch с CUDA:**
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Проверить установку:**
   ```powershell
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
   ```

## Быстрая очистка места

Выполните эти команды для быстрой очистки:

```powershell
# Очистка кэша pip
pip cache purge

# Очистка временных файлов
Remove-Item -Path "$env:TEMP\*" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$env:LOCALAPPDATA\Temp\*" -Recurse -Force -ErrorAction SilentlyContinue

# Очистка кэша Python
Get-ChildItem -Path "$env:LOCALAPPDATA\Programs\Python" -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force

# Проверка свободного места
Get-PSDrive C | Select-Object @{Name="Free(GB)";Expression={[math]::Round($_.Free/1GB,2)}}
```

## После успешной установки

Обновите config.yaml (уже сделано):
```yaml
device:
  type: "cuda"  # Теперь будет работать!
  enable_cpu_offload: true
```

Запустите:
```bash
python app.py
```

