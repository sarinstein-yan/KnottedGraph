"""Run one historical hard regression case with the exact benchmark definitions."""
from pathlib import Path
import base64
import os
import zlib

ROOT = Path(__file__).resolve().parent
payload = "".join(
    (ROOT / f"_tmp_hard_payload_{index}.txt").read_text().strip()
    for index in range(1, 5)
)
source = zlib.decompress(base64.b64decode(payload)).decode("utf-8")
marker = "TARGET_IDS = ["
cut = source.find(marker)
if cut < 0:
    raise RuntimeError("Could not locate hard-regression target loop")

# Execute only benchmark definitions, not the historical serial target loop.
namespace = {"__name__": "hard_exact_case"}
exec(compile(source[:cut], "<hard_exact_definitions>", "exec"), namespace)

case_id = int(os.environ["CASE_ID"])
generate = namespace["generate_hard_ground_truths"]
ground_truths = generate(namespace["N_GROUND_TRUTH"], seed=namespace["RANDOM_SEED"])
by_id = {int(item["id"]): item for item in ground_truths}
if case_id not in by_id:
    raise RuntimeError(f"Historical case {case_id} missing from deterministic catalogue")
item = by_id[case_id]

sp = namespace["sp"]
A = namespace["A"]
normalized_yamada = namespace["normalized_yamada"]
yamada_equal = namespace["yamada_equal"]
reconstruct = namespace["reconstruct_from_regular_neighborhood"]
max_allowed_degree = namespace["MAX_ALLOWED_DEGREE"]
yamada_n_jobs = namespace["YAMADA_N_JOBS"]

print(
    f"CASE {case_id} variant={item['variant']} grid={item['grid_size']} "
    f"radius={item['tube_radius_vox']}",
    flush=True,
)

y_true, _ = normalized_yamada(
    item["graph"],
    rotation_angles=item["true_projection_angles"],
)
graph_recovered, skeleton_seconds, extraction_seconds = reconstruct(
    item["graph"],
    grid_size=item["grid_size"],
    tube_radius_vox=item["tube_radius_vox"],
)
max_degree = max(dict(graph_recovered.degree()).values(), default=0)
print(
    f"reconstruction max_degree={max_degree} skeleton_s={skeleton_seconds:.6g} "
    f"extraction_s={extraction_seconds:.6g}",
    flush=True,
)
if max_degree > max_allowed_degree:
    raise RuntimeError(f"Recovered graph left subcubic class: max degree={max_degree}")

y_recovered, selected_projection = normalized_yamada(
    graph_recovered,
    rotation_angles=None,
)
if not yamada_equal(y_true, y_recovered):
    raise AssertionError("Selected recovered projection does not match ground truth")
print(
    f"selected projection crossings={selected_projection.num_crossings}: YAMADA MATCH",
    flush=True,
)

sampled = __import__(
    "knotted_graph.projection", fromlist=["sample_projections"]
).sample_projections(graph_recovered, num_rotation_samples=12)
print(f"valid sampled projections={len(sampled)}", flush=True)
for index, projection in enumerate(sampled):
    poly = sp.expand(
        projection.processor.compute_yamada(
            A,
            normalize=True,
            n_jobs=yamada_n_jobs,
        )
    )
    match = sp.expand(poly - y_true) == 0
    print(
        f"projection {index}: crossings={projection.num_crossings} match={match}",
        flush=True,
    )
    if not match:
        raise AssertionError(
            f"Recovered projection {index} differs from ground truth"
        )

print(f"HARD CASE {case_id}: PASS", flush=True)
