#!/usr/bin/env bash
#
# Sanity check: the same PUL fed through both input formats must yield the
# same prediction. We pick one CGC out of the example cgc_standard.out,
# extract the raw token-string the parser would produce, and run inference
# both ways. The pul_string and predicted substrate are compared.
#
# This protects against future drift between the CGC parser and the trained
# tokenizer/featurizer.
#
# Usage:
#   bash scripts/verify_cgc_format.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

CGC_FILE="data/example_cgc_standard.out"
TARGET_CGC_ID="scaffold_1|CGC1"

echo "[verify] running cgc-standard inference on the entire example file ..."
python3 scripts/06_inference.py \
    --cgc-standard "$CGC_FILE" \
    --tc-mode both \
    --out /tmp/cgc_predictions.csv >/dev/null

# pluck the row for our target CGC
PUL_STRING=$(python3 - <<PY
import pandas as pd
df = pd.read_csv("/tmp/cgc_predictions.csv")
row = df[df["cgc_id"] == "$TARGET_CGC_ID"].iloc[0]
print(row["pul_string"])
PY
)
CGC_PRED=$(python3 - <<PY
import pandas as pd
df = pd.read_csv("/tmp/cgc_predictions.csv")
row = df[df["cgc_id"] == "$TARGET_CGC_ID"].iloc[0]
print(row["predicted"])
PY
)
echo "[verify] target CGC: $TARGET_CGC_ID"
echo "[verify] parsed PUL string:  $PUL_STRING"
echo "[verify] cgc-standard prediction: $CGC_PRED"

echo
echo "[verify] now feeding that exact string through the --seq path ..."
SEQ_PRED=$(python3 scripts/06_inference.py --seq "$PUL_STRING" | python3 -c 'import sys, json; print(json.loads(sys.stdin.read())["predicted"])')
echo "[verify] --seq prediction:        $SEQ_PRED"

echo
if [ "$CGC_PRED" = "$SEQ_PRED" ]; then
    echo "[verify] ✅ PASS — both paths produce the same prediction ($CGC_PRED)"
    exit 0
else
    echo "[verify] ❌ FAIL — predictions differ:"
    echo "[verify]    cgc-standard: $CGC_PRED"
    echo "[verify]    --seq:        $SEQ_PRED"
    exit 1
fi
