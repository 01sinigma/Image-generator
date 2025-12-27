@echo off
REM ============================================================
REM Image Generator - Оптимизированный запуск для 8GB VRAM
REM Поддержка: RTX 4060, RTX 3060, RTX 3070 и аналогичных
REM ============================================================

echo.
echo ============================================================
echo    Image Generator - 8GB VRAM Optimized
echo ============================================================
echo.

REM Оптимизация памяти CUDA
echo [1/3] Установка оптимизаций памяти CUDA...
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM Отключение телеметрии для ускорения
set HF_HUB_OFFLINE=0
set TOKENIZERS_PARALLELISM=false

REM Проверка Python
echo [2/3] Проверка Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Python не найден!
    echo Установите Python 3.10+ и добавьте в PATH
    pause
    exit /b 1
)

REM Проверка CUDA
echo [3/3] Проверка CUDA...
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>nul
if errorlevel 1 (
    echo ПРЕДУПРЕЖДЕНИЕ: Не удалось проверить CUDA
)

echo.
echo ============================================================
echo    Запуск приложения...
echo    Откройте в браузере: http://localhost:7860
echo ============================================================
echo.

REM Запуск приложения
python app.py

if errorlevel 1 (
    echo.
    echo ОШИБКА при запуске приложения!
    echo Проверьте:
    echo   1. Установлены ли все зависимости: pip install -r requirements.txt
    echo   2. Скачаны ли модели: python download_all_models.py
    echo   3. Есть ли свободное место на диске
)

pause
