#!/usr/bin/env bash
# Check for INVALID/ERROR responses across all result files.
# Flags any file with >1% invalid rate.
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=== Single-Turn Results ==="
for f in data/results/single_turn/*.jsonl; do
  total=$(wc -l < "$f")
  invalids=$(grep -c '"model_answer": "INVALID"\|"model_answer": "ERROR"' "$f" || true)
  pct=$(python3 -c "print(f'{${invalids}/${total}*100:.1f}%')")
  flag=""
  if python3 -c "exit(0 if ${invalids}/${total} > 0.01 else 1)" 2>/dev/null; then
    flag=" *** HIGH"
  fi
  echo "  $(basename "$f"): ${invalids}/${total} (${pct})${flag}"
done

echo ""
echo "=== Multi-Turn Results ==="
for f in data/results/multi_turn/*.jsonl; do
  total=$(wc -l < "$f")
  invalids=$(grep -c '"model_answer": "INVALID"\|"model_answer": "ERROR"' "$f" || true)
  pct=$(python3 -c "print(f'{${invalids}/${total}*100:.1f}%')")
  flag=""
  if python3 -c "exit(0 if ${invalids}/${total} > 0.01 else 1)" 2>/dev/null; then
    flag=" *** HIGH"
  fi
  echo "  $(basename "$f"): ${invalids}/${total} (${pct})${flag}"
done
