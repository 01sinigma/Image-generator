@echo off
REM Простой скрипт для скачивания моделей
REM Требуется минимум 62 GB свободного места

echo ========================================
echo Скачивание моделей для Image Generator
echo ========================================
echo.
echo ВНИМАНИЕ: Требуется минимум 62 GB свободного места!
echo.
pause

echo.
echo Начинаю скачивание моделей...
echo.

REM Скачивание Z-Image-Turbo
echo [1/2] Скачивание Z-Image-Turbo (~12 GB)...
python download_models.py --model z-image-turbo
if %errorlevel% neq 0 (
    echo ОШИБКА при скачивании Z-Image-Turbo
    pause
    exit /b 1
)

echo.
echo [2/2] Скачивание Qwen-Image-Edit-2511 (~50 GB)...
python download_models.py --model qwen-image-edit
if %errorlevel% neq 0 (
    echo ОШИБКА при скачивании Qwen-Image-Edit-2511
    pause
    exit /b 1
)

echo.
echo ========================================
echo Все модели успешно скачаны!
echo ========================================
echo.
echo Не забудьте обновить config.yaml для использования локальных моделей.
echo.
pause

