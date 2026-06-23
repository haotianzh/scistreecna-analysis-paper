# ScisTree-CNA — Experiment Scripts

Scripts used to reproduce the results in our paper on **ScisTree-CNA** — joint
single-cell phylogenetic inference from SNVs and copy-number alterations (CNAs),
built on top of [ScisTree2](https://github.com/yufengwudcs/ScisTree2).

> Note: this is research/development code, not a packaged tool. It may contain
> hard-coded paths and exploratory branches, but the core ideas match the paper.
> For actual use, install the publicly released ScisTree-CNA package; you can
> follow the ideas here and adapt them to the latest open-source version.

## Layout

- `simulation/` — Simulation benchmark. `run_simulation.py` is the main entry
  point (split into per-scenario `run_simulation_*()` functions, selected in the
  `__main__` block); `csv/` holds the result tables; `visualization.ipynb` makes
  the figures.
- `hgsoc/` — OV2295 real-data analysis (`run_hgsoc.py`).
- `missionbio/` — MissionBio mixed cell-line experiment (`run_3_mixed.py`).

These scripts import the method implementation (`scistree_cna.py`, etc.) from the
project root, so run them from the root.

## Installation

No installation instructions are provided. The scripts depend on `numpy`,
`pandas`, `scipy`, `cupy` (GPU), `biopython`, `scikit-learn`, `scistree2`,
`popgen`, and others, plus the baseline tools CellPhy, DICE, CONDOR, and CellCoal.
Please install the corresponding packages yourself. A CUDA-capable GPU is required
for the GPU code paths.

## Data

The OV2295 dataset is large and is **not included** in this repository. It is the
same dataset used by ScisTree2 and can be downloaded from the data source cited in
the paper (archived on Zenodo); place it under `HGSOC/`.
