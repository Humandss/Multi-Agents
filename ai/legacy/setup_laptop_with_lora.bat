@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   AI 서버 노트북 셋업 + LoRA 재학습 (약 60-90분)
echo ============================================
echo.
echo 주의: LoRA는 측정 결과 prompting baseline에 짐 (ablation으로만 사용).
echo 재학습은 시간 비용이 큼 — 정말 필요한 경우만 실행.
echo.
pause

echo [1/5] uv sync — Python 의존성 설치 (10-15분)
python -m uv sync
if errorlevel 1 (
    echo uv sync 실패. uv가 설치되어 있나요?  pip install uv
    pause
    exit /b 1
)

echo.
echo [2/5] HF 토큰 확인
python -m uv run huggingface-cli whoami >nul 2>nul
if errorlevel 1 (
    echo HF 로그인 필요. 토큰을 입력하세요.
    python -m uv run huggingface-cli login
)

echo.
echo [3/5] 데이터 전처리
python -m uv run python scripts/preprocess.py
python -m uv run python scripts/build_multiturn_data.py

echo.
echo [4/5] 5종 LoRA 학습 (NPC당 별도 프로세스, ~60분 + 베이스 다운로드)
echo 8GB GPU에서는 NPC당 별도 프로세스로 실행해야 OOM 회피.
python -m uv run python scripts/train_lora.py --char elias --epochs 5
python -m uv run python scripts/train_lora.py --char hermann --epochs 5
python -m uv run python scripts/train_lora.py --char mathilda --epochs 5
python -m uv run python scripts/train_lora.py --char finn --epochs 5
python -m uv run python scripts/train_lora.py --char bernhardt --epochs 5

echo.
echo [5/5] 시드 메모리 적재
python -m uv run python scripts/seed_memory.py

echo.
echo ============================================
echo   셋업 완료 (LoRA 어댑터 학습됨)
echo   서버에서 LoRA 활성화: NpcServer(use_lora=True)
echo ============================================
pause
