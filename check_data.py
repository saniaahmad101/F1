from pathlib import Path
import os
from script import FILE_MAP, DATA_DIR  # uses the constants from your script

print("DATA_DIR =", DATA_DIR)
missing = []
for k, fname in FILE_MAP.items():
    p = Path(DATA_DIR) / fname
    print(f"{k:22s} ->", "FOUND" if p.exists() else "MISSING", "-", p)
    if not p.exists():
        missing.append(fname)

if missing:
    print("\nMissing files:", ", ".join(missing))
else:
    print("\nAll expected CSVs are present ✅")
