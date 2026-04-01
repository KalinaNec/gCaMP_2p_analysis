# Calcium Imaging ROI and Spike Analysis

Python scripts for:
- motion correction of calcium movies,
- ROI detection in GCaMP cell bodies,
- trace extraction and spike/event detection,
- labeled GIF generation for a few large, visible, strongly spiking cells,
- pre/post statistics with experimental vs control comparison.

This repository is set up to be uploaded to GitHub as code and documentation. Raw imaging data and generated outputs are intentionally ignored by `.gitignore`.

## Scripts
- `alignment.py`
  - motion-corrects TIFF/CZI inputs into `*_rigid.tif`
- `find_ring_rois.py`
  - detects ROIs on motion-corrected movies and saves masks, overlay PNGs, and ROI stats
- `spikes.py`
  - extracts traces, detects events, saves trace/event tables, EEG-like summaries, and representative GIFs
- `pre_post_stats.py`
  - aggregates activity outputs and runs mouse-level pre/post statistics with experimental vs control comparisons

## Environment
The workflow was validated in a Conda environment similar to:

```bash
conda env create -f environment.yml
conda activate caiman-calcium
```

Validated package versions:
- Python 3.10
- `cellpose==4.0.6`
- `numpy==1.26.4`
- `pandas==2.3.1`
- `scipy==1.15.2`
- `matplotlib==3.10.5`
- `tifffile==2024.12.12`
- `imageio==2.37.0`

## Expected folder layout
Minimal working layout:

```text
project/
  alignment.py
  find_ring_rois.py
  spikes.py
  pre_post_stats.py
  raw_data/
  corrected/
  cellpose/
  activity/
```

You can rename `raw_data/` to whatever you want; just pass the matching path with script arguments.

## Main workflow
1. Motion correction

```bash
python alignment.py
```

2. ROI detection

```bash
python find_ring_rois.py --input-dir ./corrected --output-dir ./cellpose --gpu
```

3. Spike extraction and GIF generation

```bash
python spikes.py --movie-dir ./corrected --mask-dir ./cellpose --out-dir ./activity --top-gif-cells 3
```

4. Mouse-level pre/post statistics

```bash
python pre_post_stats.py --activity-dir ./activity --out-dir ./activity/pre_post_stats
```

## Important analysis rules
- Control cohort:
  - recordings under `2025_03_05_C`
- Main comparison:
  - mouse-level pre vs post
- Excluded from statistics:
  - files with `green`, `red`, `max`, `outside`, `gfp`, or `reda` in the movie name
  - `_tmp_for_caiman` duplicates
  - root-level ad hoc duplicates

## Example outputs
See curated examples in:
- `paper_figures/README.md`
- `paper_figures/figure_captions.md`

Representative packaged examples include:
- ROI overlays
- stacked trace plots
- EEG-like plots
- a few representative GIFs
- pre/post summary figures

## Notes
- `spikes.py` chooses only a few GIFs per movie and favors large, visible, strongly spiking cells.
- ROI labels in overlays are pushed outside cell bodies with leader lines.
- GIFs use arrow cues instead of dots on top of the neuron.

## Example commands
See `examples/EXAMPLES.md` for copy-paste commands using representative inputs.

