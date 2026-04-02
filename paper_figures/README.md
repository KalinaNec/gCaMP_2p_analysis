# Paper Figures Summary

## Dataset Structure
- Experimental cohort: all non-`2025_03_05_C` recordings.
- Control cohort: `2025_03_05_C`.
- Statistical unit: paired movie or field-of-view pre vs post comparison.
- Upgraded tests now include:
  - paired Wilcoxon,
  - paired permutation tests,
  - bootstrap confidence intervals for change,
  - between-cohort permutation tests on pre-to-post deltas,
  - FDR correction across metrics.

## Channel analysis add-on
- Separate red/green channel analysis is now included from the corrected reference-channel images.
- `green` is the GCaMP/expression channel, `red` is the opsin channel, and `merged` is the pixelwise sum of the two after robust rescaling.
- These channel plots are descriptive structural/reference analyses, not spike/event analyses.
- The current channel comparison used `30` mouse-level channel pairs in total: `20` experimental and `10` control.
- The added `gcamp_ctrl` folder contributes to the channel analysis, but most of its calcium `pre/post` movies still had empty ROI masks and therefore did not materially expand the spike-based pre/post comparison.

## Main Statistical Interpretation
- Across all matched pairs combined, there is no FDR-significant global pre/post effect when cohorts are pooled.
- Within the experimental cohort, post recordings trend toward lower event burden, but the within-cohort paired effects do not survive FDR correction.
- Within the control cohort, post recordings trend toward higher event burden, but the within-cohort paired effects also do not survive FDR correction, likely because the control sample is small (`n=4` pairs).
- The strongest and most reliable result is the between-cohort difference in pre-to-post change:
  - `mean_event_count_active`: experimental delta `-0.256`, control delta `+0.888`, delta difference `-1.144`, 95% CI `[-1.391, -0.914]`, permutation FDR `0.005`, Hedges' g `-1.99`
  - `mean_event_rate_active`: experimental delta `-0.00065`, control delta `+0.00494`, delta difference `-0.00559`, 95% CI `[-0.0070, -0.0042]`, permutation FDR `0.0155`, Hedges' g `-1.63`
- Interpretation:
  - Control movies become more active post condition.
  - Experimental movies move in the opposite direction or stay flat.
  - The between-group contrast is strongest for active-ROI event count and event rate, not for active fraction or peak amplitude.

## How To Read The Main Plots
- `pre_post_by_cohort.png`:
  - Each panel shows paired pre/post points and connecting lines.
  - Significance stars and permutation-FDR values are printed directly on the plot.
  - Read this figure as the within-cohort direction-of-change view.
- `pre_post_deltas.png`:
  - Each panel compares `post - pre` between experimental and control.
  - This is the key inferential figure for the current dataset.
  - Significant stars here indicate that the cohorts differ in how they changed, even if neither cohort alone crosses significance.
- `between_cohort_effects.png`:
  - Compact effect-summary figure.
  - Points are mean delta differences with 95% bootstrap intervals.
  - Metrics whose interval stays away from zero and carry stars are the most compelling candidates for reporting.
- `pre_post_summary.png`:
  - Same paired logic as above, but with cohorts pooled.
  - Use it descriptively, not as the main biological conclusion figure.

## How To Read The Channel Plots
- `channel_intensity_by_cohort.png`:
  - Experimental and control mice are compared directly for green, red, and merged intensity/fraction metrics.
  - The experimental group tends to sit a bit higher for several intensity measures, especially green and merged signal, but none of these effects survive FDR correction in the current sample.
- `channel_overlap_by_cohort.png`:
  - These panels ask whether green and red spatially overlap more in one cohort than the other.
  - The overlap/correlation metrics trend slightly upward in experimental mice, but again none pass FDR correction, so this should be described as a trend rather than a supported group difference.
- `channel_within_mouse.png`:
  - Each line is one mouse, compared across green-only, red-only, and merged images.
  - The merged channel is significantly higher than either green or red alone across intensity and area metrics, which is expected because the merged image is the sum of both channels.
  - Green-only versus red-only is not convincingly different after correction, so there is no strong evidence here that one reference channel dominates the other overall.

## Channel interpretation
- The current channel analysis does not show an FDR-significant structural/reference-channel split between experimental and control mice.
- The cleanest negative result is that neither green intensity, red intensity, nor green/red overlap survives correction at the between-cohort level.
- In plain terms: the strong group difference in this dataset is still in calcium event dynamics, not in the static red/green channel reference images.
- That matters because it argues against the pre/post spiking result being a trivial artifact of one group simply having dramatically brighter green or red reference-channel images overall.

## Recommended Figure Order
1. `pre_post_deltas.png`
2. `between_cohort_effects.png`
3. `pre_post_by_cohort.png`
4. `channel_intensity_by_cohort.png`
5. `channel_overlap_by_cohort.png`
6. `channel_within_mouse.png`
7. `exp_M05pre_overlay.png` and `exp_M05post_overlay.png`
8. `ctrl_M06pre_overlay.png` and `ctrl_M06post_overlay.png`
9. `exp_M05pre_ROI011.gif`, `exp_M05post_ROI007.gif`, `ctrl_M06pre_ROI018.gif`, `ctrl_M06post_ROI037.gif`

## Caveats
- This is still a movie-level paired analysis, not a mixed-effects model with cells nested in mice.
- Some channels are structural or reference channels and were correctly skipped because masks were empty.
- The new `gcamp_ctrl` controls mostly contributed to the channel analysis rather than the spike analysis, because ROI detection on those calcium movies was often empty or near-empty with the current settings.
- Temp-derived `_tmp_for_caiman` activity outputs were removed from the packaged dataset and excluded from the final statistics.
- Some representative root-level files have very small ROI counts; they remain in the dataset, but they should be interpreted cautiously.

## Source Tables
- Full test table: `../activity/pre_post_stats/statistical_tests.csv`
- Between-cohort delta tests: `../activity/pre_post_stats/delta_between_cohorts.csv`
- Compact between-cohort summary: `../activity/pre_post_stats/results_summary_between_cohorts.csv`
- Compact within-cohort summary: `../activity/pre_post_stats/results_summary_within_cohorts.csv`
- Movie-level metrics: `../activity/pre_post_stats/movie_metrics.csv`
- Channel between-cohort table: `../activity/channel_stats/channel_between_cohorts.csv`
- Channel overlap table: `../activity/channel_stats/channel_overlap_between_cohorts.csv`
- Channel within-mouse tests: `../activity/channel_stats/channel_within_mouse_tests.csv`
