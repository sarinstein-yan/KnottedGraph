# Temporary exact regression runner generated from the executed hard benchmark.
# Removed after validation. The deterministic payload is split only to keep the
# GitHub connector transport reliable.
from pathlib import Path
import base64
import zlib

ROOT = Path(__file__).resolve().parent
payload = "".join(
    (ROOT / f"_tmp_hard_payload_{index}.txt").read_text().strip()
    for index in range(1, 5)
)
assert len(payload) == 14200, len(payload)
exec(zlib.decompress(base64.b64decode(payload)))
