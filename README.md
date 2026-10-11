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

This repository holds a runnable example of each setting in the paper: the
method on a simulated shape, and the method on a single real thermal video. In
the code it is abbreviated **SFLS**, shape from Laplacian and shading, where the
Laplacian is the one recovered from heat conduction.

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

---

## Notebooks

Two notebooks, one per setting. Each runs the method end to end and is meant to
be read top to bottom; they are the representative examples rather than the full
set of experiments in the paper.

### `notebooks/sim.ipynb` — simulation

Ground-truth Laplacian and shading of a mesh, then the method:

1. **Light source estimation** — near-planar patches give candidate normals,
   whose two solutions are separated by shading; RANSAC over reverse
   photometric stereo and an iterative refinement give the source direction.
2. **Dense normal initialization** — per pixel, the convex/concave candidate
   that agrees with the shading is selected; invalid and ambiguous pixels are
   filled.
3. **Global refinement** — Adam on the Laplacian and shading residuals.
4. **Results** — normal and shading error against ground truth.

`sim_params` at the top selects the shape (`obj_name`, any `.obj` in `objs/`)
and the light source (`theta_s`, `phi_s`).

### `notebooks/real.ipynb` — real thermal video

The same method on a single thermal video: the Shape Laplacian is recovered
from heat conduction, then shape optimization and the shading stage alternate.
Two settings at the top:

```python
OBJECT = "paintedbear"   # bear, bunny, droplet, heart, paintedbear, soda_can, sphere
BASELINE = False         # True: heat conduction only, shading stage switched off
```

`BASELINE = True` is the comparison setting, shape from heat conduction alone:
the shading stage never runs and the hyper-parameters tuned for that setting are
applied. It is recorded for `bear`, `heart`, `paintedbear`, `soda_can` and
`sphere`.

Every per-object value — recordings, hyper-parameters, number of
shape/temperature loops — lives in `config/real_objects.py`, taken from the runs
reported in the paper. `paintedbear` is the default because its mask ships with
this repository; the other objects need the dataset below.

`lib/propagation.py` holds the BFS propagation variant of step 2, which
resolves the sign pixel by pixel along a scan order instead of filling. Neither
notebook uses it; it is kept because it is part of the method.

#### Monitor training with TensorBoard

```bash
tensorboard --logdir tb-logs
# then open http://localhost:6006 in your browser
# (use SSH port forwarding if running on a remote server)
ssh -L 6006:localhost:6006 user@your-server
```

---

## Data

The object masks are in this repository. The thermal videos are not — they are
tens to hundreds of megabytes each — so they are attached to a GitHub release
and fetched by:

```bash
./scripts/download_data.sh              # paintedbear, the object real.ipynb opens with
./scripts/download_data.sh bear sphere  # the objects you name
./scripts/download_data.sh --all        # every object
./scripts/download_data.sh --list       # what the release holds, and how large
```

It reads which recordings an object needs from `config/real_objects.py`, so the
script and the notebook cannot drift apart, and it skips anything already
downloaded (an interrupted download resumes).

`paintedbear` is the worked example and the one recording published here.
`config/real_objects.py` records the settings for every object reported in the
paper, so the notebook can run them too, but their recordings are not
distributed from this repository: `bear`, `bunny`, `soda_can` and `sphere` come
from the [shape-from-heat](https://github.com/nnsriram97/shape-from-heat)
dataset, and asking for any of the others prints where it comes from.

Recordings are referred to by a short name (`paintedbear.npz`, `bear-0.npz`,
...). `SOURCE_RECORDINGS` in `config/real_objects.py` records which capture each
one is, and `scripts/prepare_data.py` renames the original captures to those
names when you obtained them elsewhere.

| Location | Format | Description |
|---|---|---|
| `DATA/obj_study_raw/` | `.npz` or `.npz.zst` | Raw thermal video frames (downloaded) |
| `DATA/obj_study_proc/` | `*_mask.png`, `*_mask.pkl` | Binary object mask, one per recording (in this repository) |
| `DATA/obj_study_proc/` | `opti_results*.pkl` | Laplacian stage checkpoint, written by the first shape/temperature loop and read by later ones |
| `objs/` | `.obj` | Meshes for `sim.ipynb` (`igea-60kf`, `concave_wave`); any `.obj` dropped here can be selected by `obj_name` |
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
│   ├── propagation.py      # BFS propagation variant of the sign resolution
│   └── constants.py        # Camera and physical constants
├── notebooks/
│   ├── sim.ipynb           # Simulation, end to end
│   └── real.ipynb          # Real thermal video, end to end
├── scripts/
│   ├── download_data.sh    # Fetch thermal videos and masks
│   └── prepare_data.py     # Rename captures to the names the notebooks use
├── largesteps/             # Large-step optimization (bundled)
├── meshplot/               # Mesh visualization (bundled)
├── objs/                   # Meshes for the simulation
├── DATA/                   # Thermal videos and masks
├── results/                # Saved mesh sequences
└── tb-logs/                # TensorBoard logs
```

---

## Camera Support

`CAMERA` at the top of `lib/constants.py` selects the radiometric constants used
to turn raw counts into temperature, and it has to match the camera the
recording came from — with the wrong one the conversion takes the log of a
non-positive number and the whole pipeline becomes NaN without raising.

| Camera | Description |
|---|---|
| `Boson-Colocated` | FLIR Boson, co-located with the light source — the default, and the setting the paper's runs used for the distributed recordings |
| `Boson` | FLIR Boson |
| `Boson+` | FLIR Boson+ |

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
