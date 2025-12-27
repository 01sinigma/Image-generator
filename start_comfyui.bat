@echo off
REM Запуск ComfyUI с поддержкой GGUF
echo.
echo ============================================================
echo    ComfyUI + GGUF для Qwen-Image-Edit-2511
echo ============================================================
echo.

cd /d "C:\Work Projeckt\TEst\Image generator\ComfyUI"

REM Оптимизация памяти
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo Запуск ComfyUI...
echo Откройте в браузере: http://127.0.0.1:8188
echo.

python main.py --listen --lowvram

pause
