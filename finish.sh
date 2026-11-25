#!/bin/bash
# Waits for the overnight sweep to end, then runs the RAGTruth faithfulness-validity judgments for every judge,
# then the full analysis. Resumable: re-running skips finished work.
#   caffeinate -ims bash finish.sh > finish.log 2>&1
cd "$(dirname "$0")" || exit 1
while pgrep -f "bash sweep.sh" > /dev/null; do sleep 60; done
echo "sweep finished $(date +%H:%M); starting RAGTruth runs"
for j in qwen2.5-0.5b llama3.2-1b qwen2.5-1.5b qwen2.5-3b llama3.2-3b qwen2.5-7b llama3.1-8b; do
  for try in 1 2 3 4 5 6 7 8 9 10; do
    python3 ragtruth.py run --judge "$j" && break
    echo "attempt $try failed for $j at $(date +%H:%M); retrying"; sleep 20
  done
done
python3 ragtruth.py report
python3 analyze.py run
echo "FINISH DONE $(date)"
