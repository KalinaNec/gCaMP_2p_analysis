# Example Runs

These are concrete commands you can copy and adapt. They assume you are in the project root and using the validated Conda environment.

```bash
conda activate caiman-calcium
```

## 1. Motion-correct one dataset folder

If your raw movies are under a dated folder tree:

```bash
python alignment.py
```

This writes corrected movies into `./corrected`.

## 2. Detect ROIs on a representative corrected movie

Example sample:

```bash
python find_ring_rois.py \
  --input-dir ./corrected \
  --output-dir ./cellpose \
  --gpu
```

Representative output to inspect:

```text
cellpose/2024_08_22/M05pre_rigid_overlay.png
cellpose/2024_08_22/M05pre_rigid_masks.tif
cellpose/2024_08_22/M05pre_rigid_roi_stats.csv
```

## 3. Run spike extraction and generate a few representative GIFs

```bash
python spikes.py \
  --movie-dir ./corrected \
  --mask-dir ./cellpose \
  --out-dir ./activity \
  --top-gif-cells 3
```

Representative outputs:

```text
activity/2024_08_22/M05pre/stacked_traces.png
activity/2024_08_22/M05pre/eeg_like.png
activity/2024_08_22/M05pre/gifs/
activity/2025_03_05_C/M06post/gifs/
```

## 4. Run mouse-level statistics

```bash
python pre_post_stats.py \
  --activity-dir ./activity \
  --out-dir ./activity/pre_post_stats
```

Representative outputs:

```text
activity/pre_post_stats/pre_post_by_cohort.png
activity/pre_post_stats/pre_post_deltas.png
activity/pre_post_stats/between_cohort_effects.png
activity/pre_post_stats/results_summary_between_cohorts.csv
```

## 5. Small end-to-end sample check

If you want to test on a single corrected movie before running everything:

```bash
mkdir -p /tmp/calcium_demo/in /tmp/calcium_demo/masks /tmp/calcium_demo/activity
ln -sf "$PWD/corrected/m01pre_rigid.tif" /tmp/calcium_demo/in/m01pre_rigid.tif

python find_ring_rois.py \
  --input-dir /tmp/calcium_demo/in \
  --output-dir /tmp/calcium_demo/masks \
  --gpu

python spikes.py \
  --movie-dir /tmp/calcium_demo/in \
  --mask-dir /tmp/calcium_demo/masks \
  --out-dir /tmp/calcium_demo/activity
```

Expected sample outputs:

```text
/tmp/calcium_demo/masks/m01pre_rigid_overlay.png
/tmp/calcium_demo/activity/m01pre/stacked_traces.png
/tmp/calcium_demo/activity/m01pre/eeg_like.png
/tmp/calcium_demo/activity/m01pre/gifs/
```

## 6. Curated packaged examples

See:

```text
paper_figures/
```

Useful starting examples:

```text
paper_figures/exp_M05pre_overlay.png
paper_figures/exp_M05post_overlay.png
paper_figures/ctrl_M06pre_overlay.png
paper_figures/ctrl_M06post_overlay.png
paper_figures/exp_M05pre_ROI011.gif
paper_figures/ctrl_M06post_ROI037.gif
paper_figures/between_cohort_effects.png
```

