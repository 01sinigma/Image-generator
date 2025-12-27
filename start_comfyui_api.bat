@echo off
REM Запуск ComfyUI в режиме API
cd /d "C:\Work Projeckt\TEst\Image generator\ComfyUI"
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main.py --listen --lowvram --enable-cors-header
pause
