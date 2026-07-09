#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ENV_FILE=".qwen_api.env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "Missing DASHSCOPE_API_KEY."
  echo "Create ${ENV_FILE} with: export DASHSCOPE_API_KEY='your_key'"
  exit 1
fi

export MASTER_API_BASE_URL="${MASTER_API_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions}"
export MASTER_API_KEY="${MASTER_API_KEY:-$DASHSCOPE_API_KEY}"
export MASTER_MODEL_NAME="${MASTER_MODEL_NAME:-qwen-max-latest}"

export TOOL_API_BASE_URL="${TOOL_API_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions}"
export TOOL_API_KEY="${TOOL_API_KEY:-$DASHSCOPE_API_KEY}"
export TOOL_MODEL_NAME="${TOOL_MODEL_NAME:-qwen-vl-max-latest}"

TASK="${1:-pss}"
START_Q="${2:-1}"
END_Q="${3:-10}"
MAX_TURNS="${MAX_TURNS:-12}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

case "$TASK" in
  pss|PSS)
    PSS_RESET_CHECKPOINT=1 PSS_IGNORE_CHECKPOINT=1 PSS_LOG_SUFFIX=_qwenapi_myprompts \
    "$PYTHON_BIN" run_tasks/run_PSS_agent.py \
      --master-backend api \
      --tool-backend api \
      --prompt-dir myprompts \
      --start "$START_Q" \
      --end "$END_Q" \
      --max-turns "$MAX_TURNS" \
      --master-max-tokens "${MASTER_MAX_TOKENS:-4096}" \
      --tool-max-tokens "${TOOL_MAX_TOKENS:-1024}"
    ;;
  cc|CC)
    CC_RESET_CHECKPOINT=1 CC_IGNORE_CHECKPOINT=1 TASK_LOG_SUFFIX=_qwenapi_myprompts \
    "$PYTHON_BIN" run_tasks/run_CC_agent.py \
      --master-backend api \
      --tool-backend api \
      --prompt-dir myprompts \
      --start "$START_Q" \
      --end "$END_Q" \
      --max-turns "$MAX_TURNS" \
      --master-max-tokens "${MASTER_MAX_TOKENS:-4096}" \
      --tool-max-tokens "${TOOL_MAX_TOKENS:-2048}"
    ;;
  ccqa|CCQA)
    "$PYTHON_BIN" run_tasks/run_CCQA_agent.py \
      --master-backend api \
      --tool-backend api \
      --prompt-dir myprompts \
      --start "$START_Q" \
      --end "$END_Q" \
      --max-turns "$MAX_TURNS" \
      --master-max-tokens "${MASTER_MAX_TOKENS:-4096}" \
      --tool-max-tokens "${TOOL_MAX_TOKENS:-2048}"
    ;;
  fsa|FSA)
    FSA_RESET_CHECKPOINT=1 FSA_IGNORE_CHECKPOINT=1 TASK_LOG_SUFFIX=_qwenapi_myprompts \
    "$PYTHON_BIN" run_tasks/run_FSA_agent.py \
      --master-backend api \
      --tool-backend api \
      --prompt-dir myprompts \
      --start "$START_Q" \
      --end "$END_Q" \
      --max-turns "$MAX_TURNS" \
      --master-max-tokens "${MASTER_MAX_TOKENS:-4096}" \
      --tool-max-tokens "${TOOL_MAX_TOKENS:-2048}"
    ;;
  pea|PEA)
    PEA_RESET_CHECKPOINT=1 PEA_IGNORE_CHECKPOINT=1 PEA_LOG_SUFFIX=_qwenapi_myprompts \
    "$PYTHON_BIN" run_tasks/run_PEA_agent.py \
      --master-backend api \
      --tool-backend api \
      --prompt-dir myprompts \
      --start "$START_Q" \
      --end "$END_Q" \
      --max-turns "$MAX_TURNS" \
      --master-max-tokens "${MASTER_MAX_TOKENS:-4096}" \
      --tool-max-tokens "${TOOL_MAX_TOKENS:-2048}"
    ;;
  moc|MOC)
    MOC_RESET_CHECKPOINT=1 MOC_IGNORE_CHECKPOINT=1 MOC_LOG_SUFFIX=_qwenapi_myprompts \
    "$PYTHON_BIN" run_tasks/run_MOC_agent.py \
      --master-backend api \
      --tool-backend api \
      --prompt-dir myprompts \
      --start "$START_Q" \
      --end "$END_Q" \
      --max-turns "$MAX_TURNS" \
      --master-max-tokens "${MASTER_MAX_TOKENS:-4096}" \
      --tool-max-tokens "${TOOL_MAX_TOKENS:-2048}"
    ;;
  msr|MSR)
    "$PYTHON_BIN" run_tasks/run_MSR_agent.py \
      --master-backend api \
      --tool-backend api \
      --prompt-dir myprompts \
      --start "$START_Q" \
      --end "$END_Q" \
      --max-turns "$MAX_TURNS" \
      --master-max-tokens "${MASTER_MAX_TOKENS:-4096}" \
      --tool-max-tokens "${TOOL_MAX_TOKENS:-2048}"
    ;;
  bu|BU)
    "$PYTHON_BIN" run_tasks/run_BU_agent.py \
      --master-backend api \
      --tool-backend api \
      --prompt-dir myprompts \
      --start "$START_Q" \
      --end "$END_Q" \
      --max-turns "$MAX_TURNS" \
      --master-max-tokens "${MASTER_MAX_TOKENS:-4096}" \
      --tool-max-tokens "${TOOL_MAX_TOKENS:-2048}"
    ;;
  pi|PI)
    "$PYTHON_BIN" run_tasks/run_PI_agent.py \
      --master-backend api \
      --tool-backend api \
      --prompt-dir myprompts \
      --start "$START_Q" \
      --end "$END_Q" \
      --max-turns "$MAX_TURNS" \
      --master-max-tokens "${MASTER_MAX_TOKENS:-4096}" \
      --tool-max-tokens "${TOOL_MAX_TOKENS:-2048}"
    ;;
  nc|NC)
    "$PYTHON_BIN" "run_tasks/run_NC _agent.py" \
      --master-backend api \
      --tool-backend api \
      --prompt-dir myprompts \
      --start "$START_Q" \
      --end "$END_Q" \
      --max-turns "$MAX_TURNS" \
      --master-max-tokens "${MASTER_MAX_TOKENS:-4096}" \
      --tool-max-tokens "${TOOL_MAX_TOKENS:-2048}"
    ;;
  *)
    echo "Unknown task: $TASK"
    echo "Usage: $0 {pss|cc|ccqa|fsa|pea|moc|msr|bu|pi|nc} [start] [end]"
    exit 2
    ;;
esac
