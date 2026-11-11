#!/usr/bin/env bash
# Run MERA TEXT v2 tasks one by one and pack the logs into a submission zip.
# Result: <output_path>_submission.zip, or <output_path>/errors.log on failure.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKER="$ROOT/scripts/log_to_submission_v2.py"

usage() {
  cat <<EOF
Usage: $(basename "$0") -a MODEL_ARGS [-g GEN_KWARGS] [options] [-- extra lm_eval args]

Required:
  -a, --model_args ARGS     lm_eval --model_args, e.g. "pretrained=Qwen/Qwen3-8B,dtype=bfloat16"

Options:
  -g, --gen_kwargs ARGS     lm_eval --gen_kwargs, e.g. "do_sample=False,max_gen_toks=8192".
                            Without max_gen_toks a task uses its yaml value, else the
                            harness default of 256, which empties reasoning-model answers
  -m, --model ENGINE        lm_eval engine (default: hf)
  -t, --tasks T1,T2,...     subset of tasks (default: all tasks known to the packer)
  -o, --output_path DIR     logs directory (default: ./mera_results)
  -b, --batch_size N        batch size (default: 1)
  -d, --device DEVICE       device (default: cuda)
  -l, --limit N             first N docs per task; smoke test only, not a valid submission
      --no_chat_template    drop --apply_chat_template --fewshot_as_multiturn (base models)
  -h, --help                show this help

Anything after "--" goes to every lm_eval call as is and overrides the defaults above,
e.g. -- --verbosity INFO --system_instruction "..." --use_cache ./cache

A failed task does not stop the run. Rerun only the failed ones with --tasks into the
same --output_path: the packer takes the latest log of each task.

Examples:
  # Local HF model
  bash scripts/run_evaluation.sh \\
    -a "pretrained=Qwen/Qwen3-8B,dtype=bfloat16" \\
    -g "do_sample=False,max_gen_toks=8192" \\
    -o ./mera_results/qwen3-8b

  # OpenAI-compatible API (vLLM serve, OpenRouter, ...); the key goes in OPENAI_API_KEY
  OPENAI_API_KEY=... bash scripts/run_evaluation.sh \\
    -m local-chat-completions \\
    -a "model=qwen/qwen3-8b,base_url=https://openrouter.ai/api/v1/chat/completions,num_concurrent=8,max_retries=5,timeout=600" \\
    -g "do_sample=False,max_gen_toks=8192" \\
    -o ./mera_results/qwen3-8b-api
EOF
}

die() { echo "error: $*" >&2; exit 2; }

# ---- arguments ----------------------------------------------------------------
MODEL="hf"
MODEL_ARGS=""
GEN_KWARGS=""
TASKS=""
OUTPUT_PATH="./mera_results"
BATCH_SIZE=1
DEVICE="cuda"
LIMIT=""
CHAT_TEMPLATE=true
EXTRA_ARGS=()

need_value() { [[ $# -ge 2 && -n "$2" ]] || die "$1 needs a value"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--model_args)   need_value "$@"; MODEL_ARGS="$2"; shift 2 ;;
    -g|--gen_kwargs)   need_value "$@"; GEN_KWARGS="$2"; shift 2 ;;
    -m|--model)        need_value "$@"; MODEL="$2"; shift 2 ;;
    -t|--tasks)        need_value "$@"; TASKS="$2"; shift 2 ;;
    -o|--output_path)  need_value "$@"; OUTPUT_PATH="$2"; shift 2 ;;
    -b|--batch_size)   need_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    -d|--device)       need_value "$@"; DEVICE="$2"; shift 2 ;;
    -l|--limit)        need_value "$@"; LIMIT="$2"; shift 2 ;;
    --no_chat_template) CHAT_TEMPLATE=false; shift ;;
    -h|--help)         usage; exit 0 ;;
    --)                shift; EXTRA_ARGS=("$@"); break ;;
    *)                 usage >&2; die "unknown option: $1" ;;
  esac
done

[[ -n "$MODEL_ARGS" ]] || die "--model_args is required (see --help)"
[[ "$GEN_KWARGS" == *max_gen_toks=* ]] || echo "warning: --gen_kwargs has no max_gen_toks; tasks without it in yaml" \
  "generate at most 256 tokens, too few for reasoning models (e.g. -g \"max_gen_toks=8192\")" >&2
command -v lm_eval >/dev/null || die "lm_eval not found; activate the environment with the harness installed"

# The packer's TASKS dict is the single list of v2 tasks: running a task it does not
# know would produce logs that never reach the submission.
KNOWN_TASKS=$(cd "$ROOT/scripts" && python -c 'from log_to_submission_v2 import TASKS; print(" ".join(TASKS))') \
  || die "cannot read the task list from $PACKER"
TASKS="${TASKS:-${KNOWN_TASKS// /,}}"
IFS=',' read -r -a TASK_LIST <<< "$TASKS"
for task in "${TASK_LIST[@]}"; do
  [[ " $KNOWN_TASKS " == *" $task "* ]] || die "unknown task '$task'; known: $KNOWN_TASKS"
done

# ---- run tasks ----------------------------------------------------------------
# Fixed for comparability with the leaderboard: answers are scored remotely.
FLAGS=(
  --model "$MODEL" --model_args "$MODEL_ARGS"
  --device "$DEVICE" --batch_size "$BATCH_SIZE"
  --include_path "$ROOT/benchmark_tasks" --output_path "$OUTPUT_PATH"
  --predict_only --log_samples --seed 1234 --verbosity ERROR
)
$CHAT_TEMPLATE && FLAGS+=(--apply_chat_template --fewshot_as_multiturn)
[[ -n "$GEN_KWARGS" ]] && FLAGS+=(--gen_kwargs "$GEN_KWARGS")
[[ -n "$LIMIT" ]] && FLAGS+=(--limit "$LIMIT")
FLAGS+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" TOKENIZERS_PARALLELISM=false

LOG_DIR="$OUTPUT_PATH/task_logs"
ERRORS="$OUTPUT_PATH/errors.log"
mkdir -p "$LOG_DIR"
rm -f "$ERRORS"
trap 'echo; echo "interrupted"; exit 130' INT

echo "Tasks (${#TASK_LIST[@]}): ${TASK_LIST[*]}"
echo "Output: $OUTPUT_PATH"
FAILED=()
for task in "${TASK_LIST[@]}"; do
  echo; echo "==> $task"
  if lm_eval --tasks "$task" "${FLAGS[@]}" 2>&1 | tee "$LOG_DIR/$task.log"; then
    echo "<== $task: ok"
  else
    code=$?
    FAILED+=("$task")
    { echo "### $task failed (exit $code), full log: $LOG_DIR/$task.log"
      tail -n 30 "$LOG_DIR/$task.log"; echo; } >> "$ERRORS"
    echo "<== $task: FAILED (exit $code)"
  fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  failed_csv=$(IFS=,; echo "${FAILED[*]}")
  echo "Rerun the failed tasks with: --tasks $failed_csv -o $OUTPUT_PATH" >> "$ERRORS"
  echo; echo "${#FAILED[@]} task(s) failed: $failed_csv. Nothing packed; see $ERRORS"
  exit 1
fi

# ---- pack ---------------------------------------------------------------------
SUBMISSION_DIR="${OUTPUT_PATH%/}_submission"
rm -rf "$SUBMISSION_DIR" "$SUBMISSION_DIR.zip"
echo; echo "==> packing"
if ! python "$PACKER" --outputs_dir "$OUTPUT_PATH" --dst_dir "$SUBMISSION_DIR" \
       --model_args "$MODEL_ARGS" 2>&1 | tee "$LOG_DIR/packing.log"; then
  { echo "### packing failed, full log: $LOG_DIR/packing.log"; tail -n 30 "$LOG_DIR/packing.log"; } > "$ERRORS"
  echo "Packing failed; see $ERRORS"
  exit 1
fi
rm -rf "$SUBMISSION_DIR"  # its contents are in the zip

# The packer skips tasks without logs and only prints a MISSING line about them.
if grep -q '^MISSING' "$LOG_DIR/packing.log"; then
  { grep '^MISSING' "$LOG_DIR/packing.log"
    echo "Run the missing tasks with --tasks ... -o $OUTPUT_PATH to complete it."; } > "$ERRORS"
  echo; echo "Submission $SUBMISSION_DIR.zip is incomplete; see $ERRORS"
  exit 1
fi

echo; echo "Submission: $SUBMISSION_DIR.zip"
[[ -z "$LIMIT" ]] || echo "warning: run with --limit $LIMIT, the submission is not complete"
