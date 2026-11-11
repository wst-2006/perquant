#!/usr/bin/env bash
set -euo pipefail

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/gpu_research_full_${RUN_ID}"
OUTPUT_DIR="reports/ml_lab/research_harness_v2_full"
mkdir -p "${LOG_DIR}"

echo "[1/6] environment info"
python --version | tee "${LOG_DIR}/python_version.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "${LOG_DIR}/nvidia_smi_before.txt"
else
  echo "nvidia-smi not found" | tee "${LOG_DIR}/nvidia_smi_before.txt"
fi

echo "[2/6] install dependencies"
pip install -r requirements-gpu.txt 2>&1 | tee "${LOG_DIR}/pip_install.log"
pip freeze > "${LOG_DIR}/requirements_gpu_lock_before.txt"

echo "[3/6] command check"
python scripts/run_ml_lab.py research-harness -h > "${LOG_DIR}/command_help.txt"

echo "[4/6] full baseline start"
START_TS="$(date -Is)"
set +e
python scripts/run_ml_lab.py research-harness \
  --training-frame-path reports/ml_lab/training_frame_v2_full.csv \
  --output-dir "${OUTPUT_DIR}" \
  --train-days 120 \
  --validation-days 20 \
  --step-days 20 \
  --top-n 10,20,50 \
  --models lightgbm,xgboost,catboost \
  --feature-set factor_lab_v2 \
  --report-contract-version ml_research_contract_v2 \
  2>&1 | tee "${LOG_DIR}/research_harness_full.log"
EXIT_CODE="${PIPESTATUS[0]}"
set -e
END_TS="$(date -Is)"

echo "[5/6] save run metadata"
pip freeze > "${LOG_DIR}/requirements_gpu_lock_after.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${LOG_DIR}/nvidia_smi_after.txt"
fi
python - <<PY
import json
from pathlib import Path
payload = {
    "run_id": "${RUN_ID}",
    "started_at": "${START_TS}",
    "finished_at": "${END_TS}",
    "exit_code": int("${EXIT_CODE}"),
    "output_dir": "${OUTPUT_DIR}",
    "log_dir": "${LOG_DIR}",
}
Path("${LOG_DIR}/run_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY

echo "[6/6] result"
if [ "${EXIT_CODE}" -ne 0 ]; then
  echo "FAILED: see ${LOG_DIR}/research_harness_full.log"
  exit "${EXIT_CODE}"
fi
echo "DONE: output=${OUTPUT_DIR}, logs=${LOG_DIR}"
