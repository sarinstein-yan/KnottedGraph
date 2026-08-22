from pathlib import Path
import base64
import zlib

ROOT = Path(__file__).resolve().parent
payload = "".join(
    (ROOT / f"_tmp_hard_payload_{index}.txt").read_text().strip()
    for index in range(1, 5)
)
source = zlib.decompress(base64.b64decode(payload)).decode("utf-8")
lines = source.splitlines()
print("=== hard-regression source around target loop ===")
for start, stop in ((1110, 1240),):
    for lineno in range(start, min(stop, len(lines)) + 1):
        print(f"{lineno:04d}: {lines[lineno - 1]}")
print("=== keyword hits ===")
for lineno, line in enumerate(lines, 1):
    if any(token in line for token in ("skeleton_image_to_graph", "adaptive_max_hops", "TARGET", "target_ids", "target_fail")):
        print(f"{lineno:04d}: {line}")
