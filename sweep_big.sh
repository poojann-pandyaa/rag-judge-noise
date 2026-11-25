#!/bin/bash
# Bigger-judge confirmation run: Qwen2.5-14B (4-bit) with the full protocol, plus a 4-bit Qwen2.5-7B control.
# Resumable: re-running skips finished records.   caffeinate -ims bash sweep_big.sh > sweep_big.log 2>&1
cd "$(dirname "$0")" || exit 1
run() {
  for try in 1 2 3 4 5; do
    python3 "$@" && return 0
    echo "attempt $try failed: $* at $(date +%H:%M); retrying"; sleep 20
  done
  echo "GAVE UP: $*"
}
echo "=== control qwen2.5-7b-4bit: base rubric + determinism $(date +%H:%M)"
run judge.py run --judge qwen2.5-7b-4bit --variants 0 --det-n 50
echo "=== control qwen2.5-7b-4bit: RAGTruth $(date +%H:%M)"
run ragtruth.py run --judge qwen2.5-7b-4bit
echo "=== qwen2.5-14b-4bit: base rubric + determinism $(date +%H:%M)"
run judge.py run --judge qwen2.5-14b-4bit --variants 0 --det-n 50
echo "=== qwen2.5-14b-4bit: RAGTruth $(date +%H:%M)"
run ragtruth.py run --judge qwen2.5-14b-4bit
echo "=== qwen2.5-14b-4bit: 3 rewordings $(date +%H:%M)"
run judge.py run --judge qwen2.5-14b-4bit --variants 1,2,3 --det-n 0
echo "=== extended analysis $(date +%H:%M)"
RJN_EXT=1 python3 analyze.py run
RJN_EXT=1 python3 ragtruth.py report
RJN_EXT=1 python3 ragtruth_ci.py
python3 confirm_big.py
echo "BIG DONE $(date)"
