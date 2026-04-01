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

## Recommended Figure Order
1. `pre_post_deltas.png`
2. `between_cohort_effects.png`
3. `pre_post_by_cohort.png`
4. `exp_M05pre_overlay.png` and `exp_M05post_overlay.png`
5. `ctrl_M06pre_overlay.png` and `ctrl_M06post_overlay.png`
6. `exp_M05pre_ROI011.gif`, `exp_M05post_ROI007.gif`, `ctrl_M06pre_ROI018.gif`, `ctrl_M06post_ROI037.gif`

## Caveats
- This is still a movie-level paired analysis, not a mixed-effects model with cells nested in mice.
- Some channels are structural or reference channels and were correctly skipped because masks were empty.
- Temp-derived `_tmp_for_caiman` activity outputs were removed from the packaged dataset and excluded from the final statistics.
- Some representative root-level files have very small ROI counts; they remain in the dataset, but they should be interpreted cautiously.

## Source Tables
- Full test table: `../activity/pre_post_stats/statistical_tests.csv`
- Between-cohort delta tests: `../activity/pre_post_stats/delta_between_cohorts.csv`
- Compact between-cohort summary: `../activity/pre_post_stats/results_summary_between_cohorts.csv`
- Compact within-cohort summary: `../activity/pre_post_stats/results_summary_within_cohorts.csv`
- Movie-level metrics: `../activity/pre_post_stats/movie_metrics.csv`
