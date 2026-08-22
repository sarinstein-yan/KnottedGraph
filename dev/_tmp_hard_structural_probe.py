from pathlib import Path
import base64
import zlib

import networkx as nx

ROOT = Path(__file__).resolve().parent
payload = "".join(
    (ROOT / f"_tmp_hard_payload_{index}.txt").read_text().strip()
    for index in range(1, 5)
)
source = zlib.decompress(base64.b64decode(payload)).decode("utf-8")
marker = "ground_truths = generate_hard_ground_truths("
cut = source.find(marker)
if cut < 0:
    raise RuntimeError("Could not locate hard-catalogue execution marker")
namespace = {"__name__": "hard_structural_probe"}
exec(compile(source[:cut], "<hard_structural_definitions>", "exec"), namespace)

generate = namespace["generate_hard_ground_truths"]
random_seed = namespace["RANDOM_SEED"]
ground_truths = generate(250, seed=random_seed)
print(f"Generated {len(ground_truths)} deterministic hard cases", flush=True)

target_ids = {147, 148, 149, 174, 175, 178, 201, 202, 203}
for item in ground_truths:
    case_id = int(item["id"])
    if case_id not in target_ids:
        continue
    print("\n" + "=" * 72, flush=True)
    print(
        f"CASE {case_id} {item['variant']} grid={item['grid_size']} "
        f"r={item['tube_radius_vox']}",
        flush=True,
    )
    try:
        volume, dx = namespace["regular_neighborhood_volume"](
            item["graph"],
            grid_size=int(item["grid_size"]),
            tube_radius_vox=int(item["tube_radius_vox"]),
        )
        skeleton = namespace["skeletonize_volume"](volume)
        extracted = namespace["skeleton_image_to_graph"](
            skeleton,
            max_junction_degree=namespace["MAX_ALLOWED_DEGREE"],
        )
        raw_deg = dict(extracted.degree())
        raw_loops = [(u, v, k, len(data.get("pts", ()))) for u, v, k, data in extracted.edges(keys=True, data=True) if u == v]
        print(
            "RAW",
            {
                "nodes": extracted.number_of_nodes(),
                "edges": extracted.number_of_edges(),
                "components": nx.number_connected_components(extracted) if extracted.number_of_nodes() else 0,
                "max_degree": max(raw_deg.values(), default=0),
                "degree_hist": dict(sorted(__import__("collections").Counter(raw_deg.values()).items())),
                "self_loops": raw_loops,
            },
            flush=True,
        )
        world = namespace["_graph_from_voxel_to_world"](extracted, dx=dx)
        recovered = namespace["_cleanup_reconstructed_graph"](world, dx=dx)
        rec_deg = dict(recovered.degree())
        rec_loops = [(u, v, k, len(data.get("pts", ()))) for u, v, k, data in recovered.edges(keys=True, data=True) if u == v]
        print(
            "CLEAN",
            {
                "nodes": recovered.number_of_nodes(),
                "edges": recovered.number_of_edges(),
                "components": nx.number_connected_components(recovered) if recovered.number_of_nodes() else 0,
                "max_degree": max(rec_deg.values(), default=0),
                "degree_hist": dict(sorted(__import__("collections").Counter(rec_deg.values()).items())),
                "self_loops": rec_loops,
            },
            flush=True,
        )
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", flush=True)
