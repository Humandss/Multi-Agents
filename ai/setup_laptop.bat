@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   AI 서버 노트북 셋업 (총 약 40분)
echo ============================================
echo.

echo [1/5] uv sync — Python 의존성 설치 (10-15분)
python -m uv sync
if errorlevel 1 (
    echo uv sync 실패. uv가 설치되어 있나요?  pip install uv
    pause
    exit /b 1
)

echo.
echo [2/5] HF 토큰 확인 — 처음이면 토큰 입력 필요
python -m uv run hf auth whoami >nul 2>nul
if errorlevel 1 (
    echo HF 로그인 필요. 토큰을 입력하세요.
    python -m uv run hf auth login
)

echo.
echo [3/5] 데이터 전처리
python -m uv run python scripts/preprocess.py
python -m uv run python scripts/build_multiturn_data.py

echo.
echo [4/5] 5종 LoRA 학습 (~10분, 베이스 모델 다운로드 포함 시 ~25분)
python -m uv run python scripts/train_lora.py --char all --epochs 3

echo.
echo [5/5] 시드 메모리 적재
python -m uv run python scripts/seed_memory.py

echo.
echo ============================================
echo   셋업 완료
echo   서버 실행: start_server.bat 더블클릭
echo ============================================
pause
