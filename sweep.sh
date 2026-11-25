#!/bin/bash
# Overnight sweep. Stage 1 = base rubric on all 250 items + T=0 determinism repeats (50 items);
# stage 2 = the three paraphrase variants. Smallest judge first. Resumable: re-running skips finished records.
#   caffeinate -ims bash sweep.sh > sweep.log 2>&1
cd "$(dirname "$0")" || exit 1
JUDGES="qwen2.5-0.5b llama3.2-1b qwen2.5-1.5b qwen2.5-3b llama3.2-3b qwen2.5-7b llama3.1-8b"

run() {  # up to 5 attempts: an interrupted run resumes from the records already written
  local j=$1; shift
  for try in 1 2 3 4 5; do
    python3 judge.py run --judge "$j" "$@" && return 0
    echo "attempt $try failed for $j at $(date +%H:%M); retrying"; sleep 10
  done
  echo "GAVE UP on $j"
}

for j in $JUDGES; do echo "=== stage1 $j $(date +%H:%M)"; run "$j" --variants 0 --det-n 50; done
for j in $JUDGES; do echo "=== stage2 $j $(date +%H:%M)"; run "$j" --variants 1,2,3 --det-n 0; done
echo "SWEEP DONE $(date)"
