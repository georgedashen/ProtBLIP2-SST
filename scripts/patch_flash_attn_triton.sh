#!/usr/bin/env bash
# Patch flash-attn Triton kernels for Triton 3.0+ (trans_b removed).
# Re-run after reinstalling flash-attn: bash scripts/patch_flash_attn_triton.sh
set -euo pipefail

TARGET="$(python -c "import flash_attn, os; print(os.path.join(os.path.dirname(flash_attn.__file__), 'flash_attn_triton.py'))")"

if [[ ! -f "$TARGET" ]]; then
  echo "flash_attn_triton.py not found" >&2
  exit 1
fi

python - <<'PY' "$TARGET"
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
replacements = {
    "tl.dot(q, k, trans_b=True)": "tl.dot(q, tl.trans(k))",
    "tl.dot(do, v, trans_b=True)": "tl.dot(do, tl.trans(v))",
}
changed = 0
for old, new in replacements.items():
    if old in text:
        text = text.replace(old, new)
        changed += 1
if changed == 0:
    print(f"Already patched or pattern missing: {path}")
else:
    path.write_text(text)
    print(f"Patched {changed} pattern(s) in {path}")
PY
