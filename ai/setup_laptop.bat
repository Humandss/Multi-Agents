@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   AI 서버 노트북 셋업 (LLM-only, 약 15-20분)
echo ============================================
echo.
echo LoRA 학습은 폐기됨 (eval에서 prompting baseline 우위 확인).
echo 본 셋업은 추론 서버용 — uv sync + HF 로그인 + 시드 메모리.
echo.
echo LoRA 재학습이 필요하면 setup_laptop_with_lora.bat 사용.
echo.

echo [1/3] uv sync — Python 의존성 설치 (10-15분)
python -m uv sync
if errorlevel 1 (
    echo uv sync 실패. uv가 설치되어 있나요?  pip install uv
    pause
    exit /b 1
)

echo.
echo [2/3] HF 토큰 확인 — 처음이면 토큰 입력 필요
python -m uv run huggingface-cli whoami >nul 2>nul
if errorlevel 1 (
    echo HF 로그인 필요. 토큰을 입력하세요.
    python -m uv run huggingface-cli login
)

echo.
echo [3/3] 시드 메모리 적재 (BGE-M3 다운로드 포함, 첫 실행 시 ~2분)
python -m uv run python scripts/seed_memory.py

echo.
echo ============================================
echo   셋업 완료
echo   서버 실행: start_server.bat 더블클릭
echo ============================================
pause
