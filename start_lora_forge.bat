@echo off
rem LoRA Forge on Windows: React UI + FastAPI over forge_core, at http://127.0.0.1:8800
rem Real captions with a local Ollama vision model; real training only on a GPU box (ai-toolkit), simulated here.
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist .venv\Scripts\python.exe (
  echo Creating .venv with uv...
  uv venv --python 3.12 .venv || goto :error
  uv pip install --python .venv\Scripts\python.exe -e ".[web,dev]" || goto :error
)
if not exist forge_web\frontend\dist\index.html (
  echo Building the frontend...
  pushd forge_web\frontend && call npm ci --no-audit --no-fund && call npm run build && popd
)
echo LoRA Forge at http://127.0.0.1:8800
start "" http://127.0.0.1:8800
.venv\Scripts\python.exe forge_web\api.py
goto :eof
:error
echo Setup failed. Check the error above.
pause
endlocal
