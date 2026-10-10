# A Theory of Shape Reconstruction from Heat Conduction and Shading

By [Akihiko Oharazawa](https://akihikooharazawa.github.io/),
[Sriram Narayanan](https://nnsriram97.github.io/),
[Mani Ramanagopal](https://scholar.google.com/citations?user=46v2zoYAAAAJ&hl=en),
[Srinivasa G. Narasimhan](https://www.cs.cmu.edu/~srinivas/)

[Paper (arXiv:2610.04052)](https://arxiv.org/abs/2610.04052) &nbsp;|&nbsp;
[Project page](https://shape-from-heat-and-shading.github.io/)

Shape from shading using a single image of a Lambertian surface is inherently
ambiguous. When the light source direction is known, the surface normal
estimation has a cone-ambiguity, which worsens when the source is unknown.
Recently, shape from heat conduction has emerged as an approach that leverages
heat transport equations to estimate the Shape Laplacian operator, an intrinsic
measure of shape. However, deriving surface normals from the Laplacian operator
encounters a local binary convex/concave ambiguity. Our contribution introduces
a novel theory to resolve these local shape ambiguities (excluding a few
degeneracies) without relying on priors like smoothness, by combining the cues
from shading and heat conduction. Our method ensures the mathematical
constraints of both shading and the Laplacian are satisfied simultaneously, even
with an unknown light source. We validate our theory through simulations of
complex shapes and analyze its performance in the presence of noise, as well as
on a noisy single thermal video of real-world objects with complex shapes and
material properties, including varying albedo.

This repository holds the simulations and the real thermal-video reconstructions
reported in the paper. In the code the method is abbreviated **SFLS**, shape
from Laplacian and shading, where the Laplacian is the one recovered from heat
conduction.

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/akihikooharazawa/shape-from-heat-conduction-and-shading.git
cd shape-from-heat-conduction-and-shading
```

### 2. Create a conda environment

```bash
conda create -n sfhs python=3.9
conda activate sfhs
```

### 3. Install PyTorch with CUDA

```bash
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
```

### 4. Install PyTorch3D

Follow the official instructions at [PyTorch3D Installation](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for your CUDA version, or:

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

### 5. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 6. (Simulation only) Install Mitsuba 3

```bash
pip install mitsuba
```

---

## Notebooks

Run them in the order below; each one states what it needs at the top.

### Simulation

| Notebook | What it does |
|---|---|
| `sim_01_pipeline.ipynb` | The method end to end on one object: light source estimation, then normals from the Laplacian and shading |
| `sim_02_experiments.ipynb` | Sweeps over objects, noise levels, BRDF, albedo and interreflection (feature flags at the top) |
| `sim_03_light_source_estimation.ipynb` | Light source direction from the Laplacian alone, and its error maps |
| `sim_04_propagation_sfs.ipynb` | Resolving the convex/concave ambiguity by BFS propagation over the image, with the visiting order visualised |
| `sim_05_analysis_by_synthesis.ipynb` | Analysis by synthesis: RANSAC over reverse photometric stereo, then per-pixel sign correction against the measured shading |
| `sim_06_depth_profile.ipynb` | Depth profiles through the reconstructions, for the BRDF / albedo / interreflection runs |

### Real thermal video

| Notebook | What it does |
|---|---|
| `real_01_create_mask.ipynb` | Draw the object mask on a thermal video |
| `real_02_reconstruct.ipynb` | Reconstruct one object. `OBJECT` picks the recording, `BASELINE = True` runs the comparison setting |
| `real_03_depth_profile.ipynb` | Depth profiles of a reconstruction against ground truth and against the comparison setting |
| `real_04_laplacian_vis.ipynb` | Visualise the recovered Shape Laplacian |
| `real_05_temperature_gradient_vis.ipynb` | Visualise the measured temperature gradients |

`real_02_reconstruct.ipynb` reads every per-object value — recordings,
hyper-parameters, number of shape/temperature loops — from
`config/real_objects.py`:

```python
OBJECT = "bear"      # bear, bunny, droplet, heart, paintedbear, soda_can, sphere
BASELINE = False     # True: heat conduction only, shading stage switched off
```

The comparison setting (`BASELINE = True`) is shape from heat conduction alone:
the third stage, which enforces the shading constraint, never runs, and the
hyper-parameters tuned for that setting are applied. It is recorded for `bear`,
`heart`, `paintedbear`, `soda_can` and `sphere`.

#### Monitor training with TensorBoard

```bash
tensorboard --logdir tb-logs
# then open http://localhost:6006 in your browser
# (use SSH port forwarding if running on a remote server)
ssh -L 6006:localhost:6006 user@your-server
```

---

## Data

The thermal videos are not in this repository. To fetch them:

```bash
pip install gdown
RAW_URL=<drive folder> MASK_URL=<drive folder> ./scripts/download_data.sh
```

`bear`, `bunny`, `soda_can` and `sphere` were captured for the
[shape-from-heat](https://github.com/nnsriram97/shape-from-heat) dataset, whose
public folders are the defaults in the script. `heart`, `droplet` and
`paintedbear` were captured for this paper.

The notebooks refer to each recording by a short name (`bear-0.npz`,
`paintedbear.npz`, ...). `scripts/download_data.sh` ends by calling
`scripts/prepare_data.py`, which renames the downloaded captures and their masks
accordingly; `SOURCE_RECORDINGS` in `config/real_objects.py` records which
capture each short name came from. Run it again at any time — it only touches
files that are not already named correctly:

```bash
python3 scripts/prepare_data.py --dry-run
```

| Location | Format | Description |
|---|---|---|
| `DATA/obj_study_raw/` | `.npz` or `.npz.zst` | Raw thermal video frames |
| `DATA/obj_study_proc/` | `*_mask.png`, `*_mask.pkl` | Binary object mask |
| `DATA/obj_study_proc/` | `opti_results*.pkl` | Laplacian stage checkpoint, written by the first shape/temperature loop and read by later ones |
| `DATA/sim-heat/` | `.pkl` | Heat simulation results, read by `sim_02_experiments.ipynb` when `USE_HEAT_SIM` is on |
| `objs/` | `.obj` | Reference meshes for the simulations |
| `results/<obj>/` | `<exp_name>.pkl` | Mesh vertex sequence (`verts_seq`, `faces`) |

---

## Repository layout

```
shape-from-heat-conduction-and-shading/
├── config/
│   └── real_objects.py     # Per-object settings for the real experiments
├── lib/
│   ├── utils_sim.py        # Simulation pipeline
│   ├── utils_real.py       # Real-data pipeline
│   ├── optimization.py     # Shape & temperature optimization (real data)
│   ├── propagation.py      # BFS propagation over the image
│   ├── constants.py        # Camera and physical constants
│   └── mitsuba_scenes.py   # Mitsuba scenes for BRDF / albedo / interreflection
├── notebooks/              # See the tables above
├── scripts/
│   ├── download_data.sh    # Fetch thermal videos and masks
│   └── prepare_data.py     # Rename captures to the names the notebooks use
├── largesteps/             # Large-step optimization (bundled)
├── meshplot/               # Mesh visualization (bundled)
├── objs/                   # Reference meshes for the simulations
├── DATA/                   # Thermal videos, masks, simulation results
├── results/                # Saved mesh sequences
└── tb-logs/                # TensorBoard logs
```

---

## Camera Support

The following FLIR thermal cameras are supported (configured in `lib/constants.py`):

| Camera | Description |
|---|---|
| `Boson` | FLIR Boson |
| `Boson+` | FLIR Boson+ |

Set `CAMERA` at the top of `lib/constants.py` to match your hardware.

---

## Citation

```bibtex
@article{Oharazawa_2026_arXiv,
  author  = {Oharazawa, Akihiko and Narayanan, Sriram and Ramanagopal, Mani and Narasimhan, Srinivasa G.},
  title   = {A Theory of Shape Reconstruction from Heat Conduction and Shading},
  journal = {arXiv preprint arXiv:2610.04052},
  year    = {2026},
}
```

The earlier conference version:

```bibtex
@inproceedings{Oharazawa_2025_ICCP,
  author    = {Oharazawa, Akihiko and Narayanan, Sriram and Ramanagopal, Mani and Narasimhan, Srinivasa G.},
  title     = {Resolving Shape Ambiguities using Heat Conduction and Shading},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computational Photography (ICCP)},
  month     = {July},
  year      = {2025},
  pages     = {182-192}
}
```

## Acknowledgements

The real-data pipeline builds on
[shape-from-heat](https://github.com/nnsriram97/shape-from-heat) by Narayanan et
al., and the bundled `largesteps/` (the Laplacian parameterization of [Large
Steps in Inverse Rendering of
Geometry](https://github.com/rgl-epfl/large-steps-pytorch) by Nicolet et al.)
and `meshplot/` come from it unchanged.
