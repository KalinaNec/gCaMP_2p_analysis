import argparse
import glob
import math
import os
import re

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats


matplotlib.use("Agg")
import matplotlib.pyplot as plt


RNG = np.random.default_rng(42)
METRICS = [
    ("active_fraction", "Active ROI fraction"),
    ("mean_event_rate_active", "Mean event rate (active ROIs)"),
    ("mean_event_count_active", "Mean event count (active ROIs)"),
    ("median_peak_dff_active", "Median peak dF/F (active ROIs)"),
    ("n_rois", "Detected ROIs"),
    ("n_active_rois", "Active ROIs"),
    ("mean_event_count_all", "Mean event count (all ROIs)"),
    ("mean_event_rate_all", "Mean event rate (all ROIs)"),
    ("mean_peak_dff_active", "Mean peak dF/F (active ROIs)"),
    ("mean_trace_std", "Mean trace SD"),
]
PRIMARY_METRICS = [metric for metric, _ in METRICS[:4]]
METRIC_LABELS = {metric: label for metric, label in METRICS}
PANEL_TITLES = {
    "active_fraction": "Active fraction",
    "mean_event_rate_active": "Event rate\n(active ROIs)",
    "mean_event_count_active": "Event count\n(active ROIs)",
    "median_peak_dff_active": "Peak dF/F\n(active ROIs)",
    "n_rois": "Detected ROIs",
    "n_active_rois": "Active ROIs",
    "mean_event_count_all": "Event count\n(all ROIs)",
    "mean_event_rate_all": "Event rate\n(all ROIs)",
    "mean_peak_dff_active": "Mean peak dF/F\n(active ROIs)",
    "mean_trace_std": "Mean trace SD",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate calcium activity outputs, run upgraded paired tests, and render publication-style figures."
    )
    parser.add_argument("--activity-dir", default="./activity")
    parser.add_argument("--out-dir", default="./activity/pre_post_stats")
    parser.add_argument("--n-permutations", type=int, default=20000)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_plot_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#faf8f4",
            "axes.edgecolor": "#3b3b3b",
            "axes.labelcolor": "#222222",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#d8d2c6",
            "grid.linewidth": 0.8,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "savefig.bbox": "tight",
        }
    )


def detect_stage(name):
    lower = name.lower()
    if "pre" in lower:
        return "pre"
    if "post" in lower:
        return "post"
    return None


def normalize_subject_key(rel_dir, movie_name):
    stage = detect_stage(movie_name)
    if stage is None:
        return None
    subject = movie_name.lower()
    subject = re.sub("post", "", subject)
    subject = re.sub("pre", "", subject)
    subject = re.sub(r"[_-]+", "_", subject).strip("_")
    rel_norm = rel_dir.replace("\\", "/").strip("/")
    return f"{rel_norm}/{subject}" if rel_norm else subject


def detect_cohort(group_dir):
    group_dir = (group_dir or "").replace("\\", "/")
    if group_dir.endswith("_C") or group_dir == "2025_03_05_C" or "/2025_03_05_C" in f"/{group_dir}":
        return "control"
    return "experimental"


def is_primary_calcium_movie(group_dir, movie_name):
    if group_dir in ("", ".", "nan") or pd.isna(group_dir):
        return False
    lower = movie_name.lower()
    # Exclude reference-channel recordings from the biological comparison.
    excluded_tokens = [
        "green",
        "red",
        "max",
        "outside",
        "gfp",
        "reda",
    ]
    return not any(token in lower for token in excluded_tokens)


def collect_movie_metrics(activity_dir):
    rows = []
    roi_paths = sorted(glob.glob(os.path.join(activity_dir, "**", "roi_summary.csv"), recursive=True))
    for path in roi_paths:
        if f"{os.sep}_tmp_for_caiman{os.sep}" in os.path.abspath(path):
            continue
        movie_dir = os.path.dirname(path)
        rel_dir = os.path.relpath(movie_dir, start=activity_dir)
        movie_name = os.path.basename(rel_dir)
        group_dir = os.path.dirname(rel_dir)
        stage = detect_stage(movie_name)
        if stage is None:
            continue
        if not is_primary_calcium_movie(group_dir, movie_name):
            continue

        df = pd.read_csv(path)
        if df.empty:
            continue

        active = df[df["event_count"] > 0].copy()
        rows.append(
            {
                "group_dir": group_dir.replace("\\", "/"),
                "cohort": detect_cohort(group_dir),
                "movie_name": movie_name,
                "mouse_id": normalize_subject_key(group_dir, movie_name),
                "pair_key": normalize_subject_key(group_dir, movie_name),
                "stage": stage,
                "n_rois": int(len(df)),
                "n_active_rois": int(len(active)),
                "active_fraction": float(len(active) / len(df)),
                "mean_event_count_all": float(df["event_count"].mean()),
                "mean_event_count_active": float(active["event_count"].mean()) if len(active) else 0.0,
                "mean_event_rate_all": float(df["event_rate_hz"].mean()),
                "mean_event_rate_active": float(active["event_rate_hz"].mean()) if len(active) else 0.0,
                "median_peak_dff_active": float(active["peak_dff_max"].median()) if len(active) else 0.0,
                "mean_peak_dff_active": float(active["peak_dff_max"].mean()) if len(active) else 0.0,
                "mean_trace_std": float(df["trace_std"].mean()),
            }
        )

    return pd.DataFrame(rows)


def paired_table(metrics_df):
    agg_keys = ["mouse_id", "cohort", "group_dir", "stage"]
    metric_names = [metric for metric, _ in METRICS]
    # Collapse repeated fields within a mouse/stage into one mouse-level row.
    mouse_level = metrics_df.groupby(agg_keys, as_index=False)[metric_names].mean()

    counts = mouse_level.groupby("mouse_id")["stage"].nunique()
    valid_keys = counts[counts == 2].index
    paired = mouse_level[mouse_level["mouse_id"].isin(valid_keys)].copy()
    if paired.empty:
        return paired, pd.DataFrame()

    wide = paired.pivot(index="mouse_id", columns="stage", values=metric_names)
    wide.columns = [f"{metric}_{stage}" for metric, stage in wide.columns]
    wide = wide.reset_index()
    meta = paired.groupby("mouse_id", as_index=False).agg(
        cohort=("cohort", "first"),
        group_dir=("group_dir", "first"),
    )
    return paired, meta.merge(wide, on="mouse_id", how="left")


def benjamini_hochberg(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(np.nan_to_num(pvalues, nan=np.inf))
    ranked = pvalues[order]
    adjusted = np.full(n, np.nan)
    running = 1.0
    for idx in range(n - 1, -1, -1):
        pval = ranked[idx]
        if np.isnan(pval):
            continue
        running = min(running, pval * n / (idx + 1))
        adjusted[order[idx]] = running
    return adjusted


def significance_label(p):
    if np.isnan(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def cohens_dz(pre, post):
    diff = np.asarray(post) - np.asarray(pre)
    if diff.size < 2:
        return np.nan
    sd = np.std(diff, ddof=1)
    if np.isclose(sd, 0):
        return np.nan
    return float(np.mean(diff) / sd)


def hedges_g(group_a, group_b):
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2)
    if np.isclose(pooled, 0):
        return np.nan
    d = (np.mean(a) - np.mean(b)) / math.sqrt(pooled)
    correction = 1.0 - (3.0 / (4.0 * (len(a) + len(b)) - 9.0))
    return float(d * correction)


def bootstrap_ci(values, n_bootstrap):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    boots = []
    for _ in range(n_bootstrap):
        sample = RNG.choice(values, size=len(values), replace=True)
        boots.append(np.mean(sample))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def paired_permutation_p(pre, post, n_permutations):
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    diff = post - pre
    if len(diff) < 2:
        return np.nan
    observed = abs(np.mean(diff))
    # For small paired n, use the exact sign-flip test instead of only random permutations.
    if len(diff) <= 14:
        count = 0
        total = 2 ** len(diff)
        for mask in range(total):
            signs = np.ones(len(diff), dtype=float)
            for idx in range(len(diff)):
                if (mask >> idx) & 1:
                    signs[idx] = -1.0
            statistic = abs(np.mean(diff * signs))
            count += statistic >= observed - 1e-12
        return float(count / total)

    random_signs = RNG.choice([-1.0, 1.0], size=(n_permutations, len(diff)))
    statistics = np.abs((diff * random_signs).mean(axis=1))
    p = (np.sum(statistics >= observed - 1e-12) + 1) / (n_permutations + 1)
    return float(p)


def independent_permutation_p(group_a, group_b, n_permutations):
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    observed = abs(np.mean(a) - np.mean(b))
    pooled = np.concatenate([a, b])
    n_a = len(a)
    statistics = []
    for _ in range(n_permutations):
        perm = RNG.permutation(pooled)
        statistics.append(abs(np.mean(perm[:n_a]) - np.mean(perm[n_a:])))
    statistics = np.asarray(statistics, dtype=float)
    return float((np.sum(statistics >= observed - 1e-12) + 1) / (n_permutations + 1))


def paired_test_row(metric, pre, post, args):
    diff = post - pre
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(pre, post, zero_method="wilcox", alternative="two-sided")
        wilcoxon_stat = float(wilcoxon_stat)
        wilcoxon_p = float(wilcoxon_p)
    except ValueError:
        wilcoxon_stat, wilcoxon_p = np.nan, np.nan

    try:
        t_stat, t_p = stats.ttest_rel(post, pre, nan_policy="omit")
        t_stat = float(t_stat)
        t_p = float(t_p)
    except Exception:
        t_stat, t_p = np.nan, np.nan

    ci_low, ci_high = bootstrap_ci(diff, args.n_bootstrap)
    return {
        "metric": metric,
        "label": METRIC_LABELS[metric],
        "n_pairs": int(len(pre)),
        "pre_mean": float(np.mean(pre)),
        "post_mean": float(np.mean(post)),
        "mean_change_post_minus_pre": float(np.mean(diff)),
        "median_change_post_minus_pre": float(np.median(diff)),
        "change_ci95_low": ci_low,
        "change_ci95_high": ci_high,
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p,
        "paired_t_stat": t_stat,
        "paired_t_p": t_p,
        "paired_perm_p": paired_permutation_p(pre, post, args.n_permutations),
        "cohens_dz": cohens_dz(pre, post),
    }


def run_paired_tests(wide_df, args):
    rows = []
    for metric, _ in METRICS:
        pre = wide_df[f"{metric}_pre"].to_numpy(dtype=float)
        post = wide_df[f"{metric}_post"].to_numpy(dtype=float)
        rows.append(paired_test_row(metric, pre, post, args))
    tests = pd.DataFrame(rows)
    tests["wilcoxon_fdr_bh"] = benjamini_hochberg(tests["wilcoxon_p"].to_numpy())
    tests["paired_t_fdr_bh"] = benjamini_hochberg(tests["paired_t_p"].to_numpy())
    tests["paired_perm_fdr_bh"] = benjamini_hochberg(tests["paired_perm_p"].to_numpy())
    return tests


def run_between_cohort_tests(wide_df, args):
    rows = []
    for metric, _ in METRICS:
        delta = wide_df[f"{metric}_post"] - wide_df[f"{metric}_pre"]
        exp = delta[wide_df["cohort"] == "experimental"].to_numpy(dtype=float)
        ctrl = delta[wide_df["cohort"] == "control"].to_numpy(dtype=float)
        if len(exp) < 2 or len(ctrl) < 2:
            continue
        try:
            u_stat, u_p = stats.mannwhitneyu(exp, ctrl, alternative="two-sided")
            u_stat = float(u_stat)
            u_p = float(u_p)
        except ValueError:
            u_stat, u_p = np.nan, np.nan
        ci_low, ci_high = bootstrap_ci(exp - np.mean(ctrl), args.n_bootstrap)
        rows.append(
            {
                "metric": metric,
                "label": METRIC_LABELS[metric],
                "n_experimental_pairs": int(len(exp)),
                "n_control_pairs": int(len(ctrl)),
                "experimental_delta_mean": float(np.mean(exp)),
                "control_delta_mean": float(np.mean(ctrl)),
                "delta_difference_mean": float(np.mean(exp) - np.mean(ctrl)),
                "delta_difference_ci95_low": ci_low,
                "delta_difference_ci95_high": ci_high,
                "mannwhitney_u": u_stat,
                "mannwhitney_p": u_p,
                "perm_p": independent_permutation_p(exp, ctrl, args.n_permutations),
                "hedges_g": hedges_g(exp, ctrl),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["mannwhitney_fdr_bh"] = benjamini_hochberg(df["mannwhitney_p"].to_numpy())
    df["perm_fdr_bh"] = benjamini_hochberg(df["perm_p"].to_numpy())
    return df


def add_significance_bar(ax, x0, x1, y, label, color="#2f2f2f"):
    h = 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0] + 1e-9)
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], color=color, lw=1.2, clip_on=False)
    ax.text((x0 + x1) / 2, y + h * 1.15, label, ha="center", va="bottom", color=color, fontsize=10, weight="bold")


def draw_paired_panel(ax, df, metric, stats_row, panel_title):
    pre = df[f"{metric}_pre"].to_numpy(dtype=float)
    post = df[f"{metric}_post"].to_numpy(dtype=float)
    jitter_pre = RNG.normal(0, 0.03, size=len(pre))
    jitter_post = RNG.normal(0, 0.03, size=len(post))
    x_pre, x_post = 0.0, 1.0

    for idx in range(len(pre)):
        ax.plot([x_pre + jitter_pre[idx], x_post + jitter_post[idx]], [pre[idx], post[idx]], color="#bcb6a9", lw=1.1, alpha=0.9, zorder=1)

    ax.scatter(np.full_like(pre, x_pre) + jitter_pre, pre, s=42, color="#2d6a8e", edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter(np.full_like(post, x_post) + jitter_post, post, s=42, color="#c2543c", edgecolor="white", linewidth=0.5, zorder=3)
    ax.plot([x_pre - 0.08, x_pre + 0.08], [np.mean(pre), np.mean(pre)], color="#17384d", lw=2.0, zorder=4)
    ax.plot([x_post - 0.08, x_post + 0.08], [np.mean(post), np.mean(post)], color="#7a2d20", lw=2.0, zorder=4)
    ax.set_xticks([x_pre, x_post], ["Pre", "Post"])
    ax.set_ylabel(METRIC_LABELS[metric], labelpad=6)
    ax.set_title(panel_title, pad=10)
    ax.grid(axis="y", alpha=0.65)

    y_max = max(np.max(pre), np.max(post))
    y_min = min(np.min(pre), np.min(post))
    pad = 0.24 * max(y_max - y_min, 0.1)
    ax.set_ylim(y_min - 0.08 * max(y_max - y_min, 0.1), y_max + pad)
    add_significance_bar(ax, x_pre, x_post, y_max + 0.03 * pad, significance_label(stats_row["paired_perm_fdr_bh"]))
    ax.text(
        0.03,
        0.03,
        f"perm-FDR={stats_row['paired_perm_fdr_bh']:.3g}\ndz={stats_row['cohens_dz']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.7, linewidth=0),
    )


def save_cohort_summary_figure(wide_df, tests_df, out_png):
    set_plot_style()
    fig, axes = plt.subplots(2, len(PRIMARY_METRICS), figsize=(16.5, 8.8), sharex=False)
    cohorts = [("experimental", "Experimental"), ("control", "Control")]
    for row_idx, (cohort_key, cohort_label) in enumerate(cohorts):
        cohort_df = wide_df[wide_df["cohort"] == cohort_key]
        cohort_tests = tests_df[tests_df["cohort"] == cohort_key].set_index("metric")
        for col_idx, metric in enumerate(PRIMARY_METRICS):
            ax = axes[row_idx, col_idx]
            if cohort_df.empty or metric not in cohort_tests.index:
                ax.axis("off")
                continue
            draw_paired_panel(ax, cohort_df, metric, cohort_tests.loc[metric], PANEL_TITLES[metric])
            if col_idx > 0:
                ax.set_ylabel("")
        fig.text(0.015, 0.74 - row_idx * 0.44, cohort_label, rotation=90, va="center", ha="center", fontsize=13, weight="bold", color="#2f2f2f")
    fig.suptitle("Pre/Post calcium activity by cohort", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0.03, 0, 1, 0.97], w_pad=2.0, h_pad=2.0)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_global_summary_figure(wide_df, tests_df, out_png):
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.2))
    test_map = tests_df[tests_df["cohort"] == "all"].set_index("metric")
    for ax, metric in zip(axes.ravel(), PRIMARY_METRICS):
        draw_paired_panel(ax, wide_df, metric, test_map.loc[metric], PANEL_TITLES[metric])
    fig.suptitle("Overall paired calcium activity summary", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97], w_pad=2.0, h_pad=2.0)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def draw_delta_panel(ax, between_df, wide_df, metric):
    delta = wide_df[f"{metric}_post"] - wide_df[f"{metric}_pre"]
    exp = delta[wide_df["cohort"] == "experimental"].to_numpy(dtype=float)
    ctrl = delta[wide_df["cohort"] == "control"].to_numpy(dtype=float)
    positions = [0, 1]
    colors = ["#2d6a8e", "#b56576"]

    for pos, values, color in zip(positions, [exp, ctrl], colors):
        x = np.full(len(values), pos, dtype=float) + RNG.normal(0, 0.04, size=len(values))
        ax.scatter(x, values, s=46, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        parts = ax.violinplot(values, positions=[pos], widths=0.55, showmeans=False, showmedians=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.18)
            body.set_edgecolor("none")
        ax.plot([pos - 0.1, pos + 0.1], [np.mean(values), np.mean(values)], color=color, lw=2.2, zorder=4)

    row = between_df.set_index("metric").loc[metric]
    ax.axhline(0, color="#9e9686", lw=1.0, ls="--")
    ax.set_xticks(positions, ["Experimental", "Control"])
    ax.set_ylabel("Post - Pre")
    ax.set_title(PANEL_TITLES[metric], pad=10)
    ax.grid(axis="y", alpha=0.65)
    y_max = max(np.max(exp), np.max(ctrl))
    y_min = min(np.min(exp), np.min(ctrl))
    pad = 0.24 * max(y_max - y_min, 0.1)
    ax.set_ylim(y_min - 0.1 * max(y_max - y_min, 0.1), y_max + pad)
    add_significance_bar(ax, positions[0], positions[1], y_max + 0.03 * pad, significance_label(row["perm_fdr_bh"]))
    ax.text(
        0.03,
        0.03,
        f"perm-FDR={row['perm_fdr_bh']:.3g}\ng={row['hedges_g']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.7, linewidth=0),
    )


def save_delta_figure(wide_df, between_df, out_png):
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.2))
    for ax, metric in zip(axes.ravel(), PRIMARY_METRICS):
        draw_delta_panel(ax, between_df, wide_df, metric)
    fig.suptitle("Between-cohort comparison of pre-to-post change", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97], w_pad=2.0, h_pad=2.0)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_effect_figure(between_df, out_png):
    set_plot_style()
    subset = between_df[between_df["metric"].isin(PRIMARY_METRICS)].copy()
    subset["metric_label"] = subset["metric"].map(METRIC_LABELS)
    subset = subset.sort_values("delta_difference_mean")

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    y = np.arange(len(subset))
    ax.axvline(0, color="#9e9686", lw=1.0, ls="--")
    ax.errorbar(
        subset["delta_difference_mean"],
        y,
        xerr=[
            subset["delta_difference_mean"] - subset["delta_difference_ci95_low"],
            subset["delta_difference_ci95_high"] - subset["delta_difference_mean"],
        ],
        fmt="o",
        color="#3d405b",
        ecolor="#81b29a",
        elinewidth=2.0,
        capsize=4,
        markersize=7,
    )
    ax.set_yticks(y, subset["metric_label"])
    ax.set_xlabel("Experimental delta - Control delta")
    ax.set_title("Effect-size style summary of cohort differences")
    ax.grid(axis="x", alpha=0.65)
    for idx, row in enumerate(subset.itertuples(index=False)):
        ax.text(
            row.delta_difference_ci95_high,
            idx,
            f" {significance_label(row.perm_fdr_bh)}",
            va="center",
            ha="left",
            color="#2f2f2f",
            fontsize=10,
            weight="bold",
        )
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def build_results_summary(all_tests, between_tests):
    focus = between_tests[between_tests["metric"].isin(PRIMARY_METRICS)].copy()
    focus = focus.sort_values("perm_fdr_bh")
    summary_rows = []
    for row in focus.itertuples(index=False):
        summary_rows.append(
            {
                "metric": row.metric,
                "label": row.label,
                "experimental_delta_mean": row.experimental_delta_mean,
                "control_delta_mean": row.control_delta_mean,
                "delta_difference_mean": row.delta_difference_mean,
                "delta_difference_ci95": f"[{row.delta_difference_ci95_low:.4f}, {row.delta_difference_ci95_high:.4f}]",
                "perm_fdr_bh": row.perm_fdr_bh,
                "significance": significance_label(row.perm_fdr_bh),
                "hedges_g": row.hedges_g,
            }
        )

    within_rows = []
    for cohort in ["experimental", "control"]:
        subset = all_tests[(all_tests["cohort"] == cohort) & (all_tests["metric"].isin(PRIMARY_METRICS))].copy()
        subset = subset.sort_values("paired_perm_fdr_bh")
        for row in subset.itertuples(index=False):
            within_rows.append(
                {
                    "cohort": cohort,
                    "metric": row.metric,
                    "label": row.label,
                    "mean_change_post_minus_pre": row.mean_change_post_minus_pre,
                    "change_ci95": f"[{row.change_ci95_low:.4f}, {row.change_ci95_high:.4f}]",
                    "paired_perm_fdr_bh": row.paired_perm_fdr_bh,
                    "significance": significance_label(row.paired_perm_fdr_bh),
                    "cohens_dz": row.cohens_dz,
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(within_rows)


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    metrics_df = collect_movie_metrics(args.activity_dir)
    metrics_df.to_csv(os.path.join(args.out_dir, "movie_metrics.csv"), index=False)

    paired_df, wide_df = paired_table(metrics_df)
    paired_df.to_csv(os.path.join(args.out_dir, "paired_movie_metrics_long.csv"), index=False)
    wide_df.to_csv(os.path.join(args.out_dir, "paired_movie_metrics_wide.csv"), index=False)

    if wide_df.empty:
        print("No complete pre/post pairs found.")
        return

    test_frames = []
    overall_tests = run_paired_tests(wide_df, args)
    overall_tests["cohort"] = "all"
    test_frames.append(overall_tests)
    for cohort in sorted(wide_df["cohort"].unique()):
        cohort_df = wide_df[wide_df["cohort"] == cohort]
        if cohort_df.empty:
            continue
        cohort_tests = run_paired_tests(cohort_df, args)
        cohort_tests["cohort"] = cohort
        test_frames.append(cohort_tests)
    all_tests = pd.concat(test_frames, ignore_index=True)
    all_tests.to_csv(os.path.join(args.out_dir, "statistical_tests.csv"), index=False)

    between_tests = run_between_cohort_tests(wide_df, args)
    if not between_tests.empty:
        between_tests.to_csv(os.path.join(args.out_dir, "delta_between_cohorts.csv"), index=False)

    compact_between, compact_within = build_results_summary(all_tests, between_tests)
    compact_between.to_csv(os.path.join(args.out_dir, "results_summary_between_cohorts.csv"), index=False)
    compact_within.to_csv(os.path.join(args.out_dir, "results_summary_within_cohorts.csv"), index=False)

    save_global_summary_figure(wide_df, all_tests, os.path.join(args.out_dir, "pre_post_summary.png"))
    save_cohort_summary_figure(wide_df, all_tests, os.path.join(args.out_dir, "pre_post_by_cohort.png"))
    if not between_tests.empty:
        save_delta_figure(wide_df, between_tests, os.path.join(args.out_dir, "pre_post_deltas.png"))
        save_effect_figure(between_tests, os.path.join(args.out_dir, "between_cohort_effects.png"))

    print(f"Saved upgraded statistics to {args.out_dir}")


if __name__ == "__main__":
    main()
