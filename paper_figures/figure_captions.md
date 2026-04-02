# Figure Captions

## Figure 1. Cohort-wise pre/post calcium activity comparison
File: `pre_post_by_cohort.png`

Paired pre/post comparisons for the experimental and control cohorts shown separately. Each line corresponds to one matched movie pair. Panels summarize active ROI fraction, mean event rate among active ROIs, mean event count among active ROIs, and median peak `dF/F` among active ROIs. Significance bars report permutation-test FDR-adjusted significance, and text boxes include permutation-FDR values and paired effect sizes (`d_z`).

## Figure 2. Distribution of pre-to-post changes
File: `pre_post_deltas.png`

Between-cohort comparison of the post minus pre change for key activity metrics. Experimental and control delta distributions are shown separately in each panel. Significance bars report permutation-test FDR-adjusted between-cohort effects, and text boxes include effect sizes (`g`).

## Figure 3. Between-cohort effect summary
File: `between_cohort_effects.png`

Effect-summary panel showing the mean difference in pre-to-post change between experimental and control cohorts, with bootstrap 95% confidence intervals. Metrics with intervals displaced from zero and significance stars are the strongest candidates for reporting as group-level effects.

## Figure 4. Global paired summary
File: `pre_post_summary.png`

Overall paired pre/post activity comparison across all matched pairs, irrespective of cohort. This figure is useful as a compact descriptive summary but should be interpreted together with the cohort-separated view.

## Figure 5. Experimental example field
Files: `exp_M05pre_overlay.png`, `exp_M05post_overlay.png`

Representative experimental pre and post ROI overlays. ROI labels are placed outside cell bodies with leader lines to avoid covering somata or dense clusters.

## Figure 6. Experimental example activity
Files: `exp_M05pre_stacked_traces.png`, `exp_M05post_stacked_traces.png`, `exp_M05pre_eeg_like.png`, `exp_M05post_eeg_like.png`

Representative experimental activity plots showing ranked ROI traces and a population EEG-like summary signal. These panels illustrate the extracted dynamics used for event quantification.

## Figure 7. Control example field
Files: `ctrl_M06pre_overlay.png`, `ctrl_M06post_overlay.png`

Representative control pre and post ROI overlays from the `2025_03_05_C` cohort.

## Figure 8. Control example activity
Files: `ctrl_M06pre_stacked_traces.png`, `ctrl_M06post_stacked_traces.png`, `ctrl_M06pre_eeg_like.png`, `ctrl_M06post_eeg_like.png`

Representative control activity plots showing the same extraction pipeline applied to the control cohort.

## Figure 9. Channel intensity by cohort
File: `channel_intensity_by_cohort.png`

Experimental and control mice are compared directly for green, red, and merged channel intensity/fraction metrics. Points show mouse-level measurements, violin shapes summarize the distributions, and significance bars report permutation-test FDR-adjusted between-cohort results. In the current dataset these channel-intensity differences trend upward in the experimental group but do not survive FDR correction.

## Figure 10. Green/red overlap by cohort
File: `channel_overlap_by_cohort.png`

Between-cohort comparison of green/red overlap, overlap fractions, and pixelwise green-red correlation. These panels test whether the spatial relationship between the GCaMP/expression channel and the opsin channel differs by cohort. None of the overlap metrics cross the FDR threshold in the current sample.

## Figure 11. Within-mouse channel comparison
File: `channel_within_mouse.png`

Within each mouse, green-only, red-only, and merged channel images are compared directly. Brackets show FDR-adjusted paired permutation-test significance for the three pairwise comparisons. The merged image is consistently higher than either single channel, which is expected from summing the channels, while green-only versus red-only differences are not compelling after correction.

## Supplementary GIFs
Files: `exp_M05pre_ROI011.gif`, `exp_M05post_ROI007.gif`, `ctrl_M06pre_ROI018.gif`, `ctrl_M06post_ROI037.gif`

Representative animated examples of selected large, visible, strongly spiking ROIs. Each GIF includes an off-cell arrow label on the image panel and a moving arrow marker on the trace panel to indicate the current frame.
