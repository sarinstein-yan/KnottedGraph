# FigureGeneration

Publication plotting notebooks and generated figure artifacts live here so the main package and benchmark tree stay tidy.

## Layout

- `notebooks/` contains the publication figure notebooks 06 and 07.
- `figures/` contains generated PDF/PNG/HTML figure outputs.

Figure input data and caches remain in `User_guide/benchmarks/results/`, where the benchmark/tutorial notebooks produce them. The figure notebooks read from that benchmark `results/` tree and write figures only under `FigureGeneration/figures/`.
