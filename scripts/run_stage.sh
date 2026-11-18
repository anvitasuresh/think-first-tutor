#!/bin/bash
# Run one pipeline stage. Kaggle's free 2x T4 is the default GPU target (PROFILE=t4); CPU stages run on
# any laptop. The Kaggle notebooks in kaggle/ are thin wrappers around this script.
#
#   bash scripts/run_stage.sh items       CPU   GSM8K items + stuck messages                    (seconds)
#   bash scripts/run_stage.sh variants    CPU   Multi-VALUE dialect + name variants             (~1-3 h, laptop)
#   bash scripts/run_stage.sh validate    CPU   analyze native-speaker ratings of the variants
#   bash scripts/run_stage.sh replies     GPU   tutor reply pool + blinded label set            (~1-1.5 h)
#   bash scripts/run_stage.sh judge       GPU   judge scores + rationales on the label set      (~10 min)
#   bash scripts/run_stage.sh audit       GPU   E2: pairs, judge + 2 RMs + 4-bit reward judge   (~4-5 h)
#   bash scripts/run_stage.sh explain     GPU   E3/E4: feature effects, rationales, probes      (~1-1.5 h)
#   bash scripts/run_stage.sh grpo-data   CPU   GRPO training prompts                           (seconds)
#   bash scripts/run_stage.sh grpo-smoke  GPU   10 GRPO steps: memory, speed, reward plumbing   (~10-15 min)
#   bash scripts/run_stage.sh grpo        GPU   Tutor A (RUN=tutor_b for the stretch run)       (~3 h)
#   bash scripts/run_stage.sh eval        GPU   E5-E7: base vs tutors on held-out questions     (~1.5 h)
#   bash scripts/run_stage.sh restore     -     Kaggle: copy data/results from attached inputs into the repo
#   bash scripts/run_stage.sh check       -     which GPUs this machine has (refuses P100)
#
# Env vars:
#   PROFILE   hardware profile, default t4 (configs/t4/<stage>.yaml where one exists). PROFILE=none uses
#             the base configs (bf16, for a single 24 GB+ Ampere GPU).
#   TUTORBIAS_BACKEND=hf   run vLLM model blocks with transformers instead (if vLLM fails on a GPU)
#   SHARDS / SHARD (variants), DATA_CONFIG (items/variants), MODELS (replies; default "qwen1.5b qwen7b"),
#   SCORERS (audit; default "judge skywork_v2 tulu3_rm judge_reward"), RUN (grpo; default tutor_a),
#   POLICY (grpo; override the policy model, e.g. Qwen/Qwen2.5-0.5B-Instruct), EVAL_MODELS (eval;
#   default "base tutor_a tutor_b", missing tutors are skipped), LIMIT (smoke-test any GPU stage on N rows).
# Uses .venv-data / .venv-gpu if they exist (scripts/setup_local.sh); otherwise the active Python.
# Every GPU stage is resumable: rerun the same command (after `restore` on Kaggle) and it continues.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:---help}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs
DATA_CFG="${DATA_CONFIG:-configs/data.yaml}"
PROFILE="${PROFILE:-t4}"
# Hardware profile: configs/<PROFILE>/<stage>.yaml where one exists, else configs/<stage>.yaml.
cfg() { if [ "$PROFILE" != "none" ] && [ -f "configs/$PROFILE/$1.yaml" ]; then echo "configs/$PROFILE/$1.yaml"; else echo "configs/$1.yaml"; fi; }
LIM=${LIMIT:+--limit $LIMIT}

py() {  # py data|gpu -> python executable for that env
  if [ -x ".venv-$1/bin/python" ]; then echo ".venv-$1/bin/python"; else echo "python"; fi
}
gpu_check() { $(py gpu) -m tutorbias.kaggle check; }

case "$STAGE" in
  items)
    P=$(py data)
    $P -m tutorbias.data.build items --config "$DATA_CFG"
    $P -m tutorbias.data.build messages --config "$DATA_CFG"
    ;;
  variants)
    P=$(py data)
    test -f data/processed/messages_test.jsonl || { echo "Run 'items' first."; exit 1; }
    # Each shard loads its own spaCy + stanza pipelines (~1.5-2 GB RAM), so default to at most 4.
    NCPU="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
    N="${SHARDS:-$(( NCPU < 4 ? NCPU : 4 ))}"
    if [ -n "${SHARD:-}" ]; then
      $P -m tutorbias.data.build variants --config "$DATA_CFG" --shard "$SHARD" --num-shards "$N"
    else
      # Finished shards are kept; interrupted ones resume from their .partial file. Delete
      # data/processed/variants.shard* to start from scratch.
      # On a Mac, caffeinate keeps the laptop awake while the shards run.
      CAF=""; command -v caffeinate >/dev/null && CAF="caffeinate -i"
      echo "Running $N shards in parallel; progress in logs/variants-shard*.log"
      pids=(); sids=()
      for i in $(seq 0 $((N - 1))); do
        if [ -f "data/processed/variants.shard$(printf %02d "$i")-of-$(printf %02d "$N").jsonl" ]; then
          echo "Shard $i already finished, skipping"; continue
        fi
        $CAF $P -m tutorbias.data.build variants --config "$DATA_CFG" --shard "$i" --num-shards "$N" \
          >> "logs/variants-shard$i.log" 2>&1 &
        pids+=($!); sids+=("$i")
      done
      failed=0
      for k in "${!pids[@]}"; do
        s="${sids[$k]}"
        wait "${pids[$k]}" || { echo "Shard $s FAILED, see logs/variants-shard$s.log:"; tail -5 "logs/variants-shard$s.log"; failed=1; }
      done
      [ "$failed" -eq 0 ] || exit 1
    fi
    # Merge once every shard is written (when shards run as separate jobs, the last one to finish does this).
    if [ "$(ls data/processed/variants.shard*-of-$(printf %02d "$N").jsonl 2>/dev/null | wc -l)" -eq "$N" ]; then
      $P -m tutorbias.data.build merge --config "$DATA_CFG"
      $P -m tutorbias.data.build check-sample --config "$DATA_CFG"
    fi
    ;;
  validate)
    # After native speakers fill in data/validation/ratings_<group>_<rater>.csv (docs/dialect_validation.md)
    $(py data) -m tutorbias.data.validate_variants --config "$DATA_CFG"
    ;;
  replies)
    gpu_check; P=$(py gpu)
    test -f data/processed/messages_audit.jsonl || { echo "No data/processed/ found. On Kaggle: attach your data dataset, then run 'restore'."; exit 1; }
    for m in ${MODELS:-qwen1.5b qwen7b}; do
      $P -m tutorbias.replies.generate --config "$(cfg replies)" --model "$m"
    done
    $P -m tutorbias.replies.generate --config "$(cfg replies)" --select
    $P -m tutorbias.labeling.label_set --config configs/calibration.yaml
    ;;
  judge)
    gpu_check; P=$(py gpu)
    $P -m tutorbias.judge.score --config "$(cfg judge)" --judge "${JUDGE:-primary}" \
      --input "${INPUT:-data/labels/label_set.jsonl}" --output "${OUTPUT:-results/judge/label_set_scores.jsonl}" \
      --id-field "${ID_FIELD:-label_id}" ${RUBRIC:+--rubric "$RUBRIC"} ${EXTRA---explain} $LIM
    ;;
  audit)
    gpu_check; P=$(py gpu)
    test -f data/processed/variants.jsonl || { echo "Missing data/processed/variants.jsonl: add it from your laptop to the tutorbias-data dataset (New Version)."; exit 1; }
    $P -m tutorbias.audit.pairs --config "$(cfg audit)"
    mkdir -p results/audit/scores
    # The 4-bit reward judge only needs the groups GRPO trains on (configs/grpo.yaml) plus the paraphrase
    # control: 7,500 pairs instead of 31,500 (the name pairs are 24,000 and never reach training).
    $P -c "import json; open('data/processed/audit_pairs_dialect.jsonl','w').writelines(l for l in open('data/processed/audit_pairs.jsonl') if json.loads(l)['axis'] != 'name')"
    failed=()
    for s in ${SCORERS:-judge skywork_v2 tulu3_rm judge_reward}; do
      case "$s" in
        judge)
          $P -m tutorbias.judge.score --config "$(cfg judge)" --judge primary \
            --input data/processed/audit_pairs.jsonl --output "results/audit/scores/$s.jsonl" --id-field pair_id $LIM ;;
        judge_reward)
          $P -m tutorbias.judge.score --config "$(cfg judge)" --judge reward \
            --input data/processed/audit_pairs_dialect.jsonl --output "results/audit/scores/$s.jsonl" --id-field pair_id $LIM ;;
        *)
          $P -m tutorbias.audit.reward_models --config "$(cfg audit)" --rm "$s" $LIM ;;
      esac || { echo "!!! scorer $s FAILED; continuing with the others (rerun later with SCORERS=$s)"; failed+=("$s"); }
    done
    # Analysis runs on whatever scorers finished, so one broken model never costs the whole run.
    $P -m tutorbias.audit.analyze --config "$(cfg audit)"
    [ ${#failed[@]} -eq 0 ] || echo "!!! Finished WITHOUT: ${failed[*]}"
    ;;
  explain)
    gpu_check; P=$(py gpu)
    test -f data/processed/audit_pairs.jsonl || { echo "Run 'audit' first (or restore its output)."; exit 1; }
    $P -m tutorbias.explain.features --config configs/explain.yaml
    $P -m tutorbias.explain.rationales subset --config configs/explain.yaml
    $P -m tutorbias.judge.score --config "$(cfg judge)" --judge primary --explain \
      --input data/processed/explain_pairs.jsonl --output results/explain/rationales.jsonl --id-field pair_id $LIM
    $P -m tutorbias.explain.rationales analyze --config configs/explain.yaml
    $P -m tutorbias.explain.probes extract --config configs/explain.yaml
    $P -m tutorbias.explain.probes fit --config configs/explain.yaml
    ;;
  grpo-data)
    $(py gpu) -m tutorbias.tutor.data --config configs/grpo.yaml
    ;;
  grpo-smoke|grpo)
    gpu_check; P=$(py gpu)
    test -f data/processed/grpo_train.jsonl || $P -m tutorbias.tutor.data --config configs/grpo.yaml
    R="${RUN:-tutor_a}"; [ "$STAGE" = "grpo-smoke" ] && R=smoke
    $P -m tutorbias.tutor.grpo --config configs/grpo.yaml --run "$R" ${POLICY:+--model $POLICY}
    ;;
  eval)
    gpu_check; P=$(py gpu)
    $P -m tutorbias.tutor.evaluate --config configs/eval.yaml --build
    for m in ${EVAL_MODELS:-base tutor_a tutor_b}; do
      $P -m tutorbias.tutor.evaluate --config configs/eval.yaml --model "$m" $LIM
    done
    $P -m tutorbias.judge.score --config "$(cfg judge)" --judge primary \
      --input results/eval/replies.jsonl --output results/eval/scores/judge.jsonl --id-field eval_id
    $P -m tutorbias.audit.reward_models --config "$(cfg audit)" --rm skywork_v2 \
      --input results/eval/replies.jsonl --output results/eval/scores/skywork_v2.jsonl --id-field eval_id
    $P -m tutorbias.tutor.analyze --config configs/eval.yaml
    ;;
  restore)
    $(py gpu) -m tutorbias.kaggle restore --repo .
    ;;
  check)
    gpu_check
    ;;
  *)
    sed -n "2,/^set -euo/p" "$0" | grep "^#"
    ;;
esac
