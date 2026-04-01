# Calcium imaging roi + spike analysis

Python scripts for calcium imaging analysis. The main use case here is:
- motion correction
- ROI finding on GCaMP movies
- trace extraction and event / spike counting
- making a few labeled gifs for cells that are actually easy to see
- pre/post stats, including control vs experimental comparison.

## what is in here
- `alignment.py`
  - does motion correction and writes `*_rigid.tif`
- `find_ring_rois.py`
  - finds ROIs, saves masks, overlay images, and ROI stats tables
- `spikes.py`
  - extracts traces, calls events, makes EEG-like summary plots, and writes a few gifs
- `pre_post_stats.py`
  - aggregates finished activity outputs and runs the pre/post stats

## env
This was tested in a conda env close to this:

```bash
conda env create -f environment.yml
conda activate caiman-calcium
```

Versions that were actually used here:
- python 3.10
- `cellpose==4.0.6`
- `numpy==1.26.4`
- `pandas==2.3.1`
- `scipy==1.15.2`
- `matplotlib==3.10.5`
- `tifffile==2024.12.12`
- `imageio==2.37.0`


Why is the env called `caiman` when the repo uses Cellpose now:

Because this thing has history. I started from CaImAn based analysis, then tried Suite2p, then switched again and ended up with Cellpose plus downstream code here. So the env name stayed from the beginning with CaImAn.

## dir idea
Something like this is enough:

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

`raw_data/` is just an example name. Use whatever folder layout you want and pass paths in the commands.

## usual workflow
1. motion correction

```bash
python alignment.py
```

2. roi finding

```bash
python find_ring_rois.py --input-dir ./corrected --output-dir ./cellpose --gpu
```

3. spike extraction + gifs

```bash
python spikes.py --movie-dir ./corrected --mask-dir ./cellpose --out-dir ./activity --top-gif-cells 3
```

4. mouse-level stats

```bash
python pre_post_stats.py --activity-dir ./activity --out-dir ./activity/pre_post_stats
```

## important for analysis
- experiment:
  - this is for an optogenetic experiment
  - mice had the hippocampus stimulated and retrosplenial cortex recorded in an optogenetic setup
  - GFP was only used as a marker of expression
  - CNO was given only to the experimental group, to activate the DREADD receptor
  - control mice did not receive CNO
  - activity was recorded before stimulation
  - in the directory / script language this ends up as `pre` vs `post`
  - so `post` is the manipulation condition
- control cohort:
  - recordings under `2025_03_05_C`
- main comparison:
  - mouse-level pre vs post

## what the code does a bit differently
- `spikes.py` only generates a few gifs per movie
- gif choice is biased toward cells that are larger, more visible, and spike hard enough to be worth looking at
- gif labels use arrows, not dots covering the neuron

## examples
There is a small examples file with copy-paste commands:
- `examples/EXAMPLES.md`

And there is a curated examples folder:
- `paper_figures/`

Useful things in there:
- overlays
- stacked traces
- EEG-like plots
- a few representative gifs
- stats summary figures

## a note on the stats
The current stats are mouse-level and green/red/reference-like recordings are excluded by name. The main signal that survived in the finished analysis was a between-group difference in pre-to-post change for active-cell event count and event rate.

So, short version:
- control mice went up post
- experimental mice went down or stayed flatter
- the difference between those trajectories is the interesting part

## if you just want to run something quickly
See:

```text
examples/EXAMPLES.md
```
