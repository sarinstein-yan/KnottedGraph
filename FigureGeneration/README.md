# FigureGeneration

Publication plotting and generated benchmark artifacts live here so the main package tree stays tidy.

## Layout

- `notebooks/` contains the publication figure notebooks 06 and 07.
- `figures/` contains generated PDF/PNG/HTML figure outputs.
- `results/` contains plotting inputs and benchmark caches used by figure notebooks.
- `data/handlebody_ground_truth/` contains the generated CSV and compressed JSON artifacts produced by notebook 04 and consumed by notebook 07.

`User_guide/benchmarks/figures` and `User_guide/benchmarks/results` are compatibility symlinks to the corresponding folders here, so older notebook paths continue to resolve.
