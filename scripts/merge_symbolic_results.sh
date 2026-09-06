#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="${1:-exps/results_symbolic}"
OUT_FILE="${2:-${RESULT_DIR}/results_merged.csv}"

python - "${RESULT_DIR}" "${OUT_FILE}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

result_dir = Path(sys.argv[1])
out_file = Path(sys.argv[2])
files = sorted(result_dir.glob("results_task_*.csv"))
if not files:
    raise SystemExit(f"No result shards found in {result_dir}")

df = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
df.sort_values(["config_index", "run"], inplace=True)
out_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_file, index=False)
print(f"Wrote {len(df)} rows to {out_file}")
PY
