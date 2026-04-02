import argparse
import glob
import math
import os
import re

import matplotlib
import numpy as np
import pandas as pd
import tifffile
from scipy import stats
from scipy.ndimage import gaussian_filter
from skimage import filters


matplotlib.use("Agg")
import matplotlib.pyplot as plt


RNG = np.random.default_rng(42)
CHANNEL_METRICS = [
    ("mean_intensity", "Mean intensity"),
    ("p95_intensity", "95th percentile intensity"),
    ("integrated_intensity", "Integrated intensity"),
    ("positive_fraction", "Positive area fraction"),
    ("positive_mean", "Mean positive intensity"),
]
OVERLAP_METRICS = [
    ("overlap_jaccard", "Green/red overlap (Jaccard)"),
    ("green_overlap_fraction", "Green overlap fraction"),
    ("red_overlap_fraction", "Red overlap fraction"),
    ("green_red_corr", "Green/red pixel correlation"),
    ("merged_union_fraction", "Merged positive fraction"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze red/green reference channels and merged channel images across cohorts."
    )
    parser.add_argument("--corrected-dir", default="./corrected")
    parser.add_argument("--out-dir", default="./activity/channel_stats")
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
    return "static"


def detect_channel(name):
    lower = name.lower()
    if "green" in lower:
        return "green"
    if "red" in lower:
        return "red"
    return None


def detect_cohort(group_dir):
    group_dir = (group_dir or "").replace("\\", "/")
    if (
        group_dir.endswith("_C")
        or group_dir == "2025_03_05_C"
        or "/2025_03_05_C" in f"/{group_dir}"
        or group_dir == "gcamp_ctrl"
        or "/gcamp_ctrl" in f"/{group_dir}"
    ):
        return "control"
    return "experimental"


def is_channel_movie(path):
    lower = os.path.basename(path).lower()
    if f"{os.sep}_tmp_for_caiman{os.sep}" in os.path.abspath(path):
        return False
    return any(token in lower for token in ["green", "red"])


def mouse_id_from_name(name):
    match = re.search(r"(m\d+)", name.lower())
    if match:
        return match.group(1)
    cleaned = re.sub(r"(green|red|pre|post|_rigid|outside|gfp|opsin)", "", name.lower())
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    return cleaned or name.lower()


def discover_channel_movies(corrected_dir):
    patterns = [
        os.path.join(corrected_dir, "**", "*_rigid.tif"),
        os.path.join(corrected_dir, "**", "*_rigid.TIF"),
    ]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern, recursive=True))
    root_abs = os.path.abspath(corrected_dir)
    items = []
    for path in sorted(set(paths)):
        if not is_channel_movie(path):
            continue
        rel = os.path.relpath(path, start=root_abs)
        rel_dir = os.path.dirname(rel).replace("\\", "/")
        movie_name = os.path.splitext(os.path.basename(rel))[0]
        items.append(
            {
                "path": path,
                "group_dir": rel_dir,
                "movie_name": movie_name,
                "stage": detect_stage(movie_name),
                "channel": detect_channel(movie_name),
                "mouse_id": mouse_id_from_name(movie_name),
            }
        )
    return pd.DataFrame(items)


def load_image(path):
    arr = tifffile.imread(path).astype(np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr.max(axis=0)
    return arr


def crop_pair(a, b):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def robust_rescale(image):
    lo, hi = np.percentile(image, [1, 99.5])
    if np.isclose(lo, hi):
        return np.zeros_like(image, dtype=np.float32)
    scaled = np.clip((image - lo) / (hi - lo), 0, 1)
    return scaled.astype(np.float32)


def positive_mask(image):
    smoothed = gaussian_filter(image, sigma=1.0)
    threshold = filters.threshold_otsu(smoothed) if np.any(smoothed > 0) else 1.0
    return smoothed >= threshold, float(threshold)


def image_metrics(image):
    scaled = robust_rescale(image)
    mask, threshold = positive_mask(scaled)
    positive = scaled[mask]
    return {
        "mean_intensity": float(np.mean(scaled)),
        "p95_intensity": float(np.percentile(scaled, 95)),
        "integrated_intensity": float(np.sum(scaled)),
        "positive_fraction": float(mask.mean()),
        "positive_mean": float(np.mean(positive)) if positive.size else 0.0,
        "threshold": threshold,
        "n_pixels": int(scaled.size),
    }, scaled, mask


def paired_permutation_p(a, b, n_permutations):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    if len(diff) < 2:
        return np.nan
    observed = abs(np.mean(diff))
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
    return float((np.sum(statistics >= observed - 1e-12) + 1) / (n_permutations + 1))


def independent_permutation_p(group_a, group_b, n_permutations):
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    observed = abs(np.mean(a) - np.mean(b))
    pooled = np.concatenate([a, b])
    n_a = len(a)
    stats_perm = []
    for _ in range(n_permutations):
        perm = RNG.permutation(pooled)
        stats_perm.append(abs(np.mean(perm[:n_a]) - np.mean(perm[n_a:])))
    stats_perm = np.asarray(stats_perm, dtype=float)
    return float((np.sum(stats_perm >= observed - 1e-12) + 1) / (n_permutations + 1))


def bootstrap_ci(values, n_bootstrap):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    means = []
    for _ in range(n_bootstrap):
        sample = RNG.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


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


def build_channel_table(items_df):
    rows = []
    for row in items_df.itertuples(index=False):
        image = load_image(row.path)
        metrics, scaled, mask = image_metrics(image)
        rows.append(
            {
                "group_dir": row.group_dir,
                "cohort": detect_cohort(row.group_dir),
                "mouse_id": row.mouse_id,
                "stage": row.stage,
                "channel": row.channel,
                "movie_name": row.movie_name,
                "path": row.path,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def build_pair_table(items_df):
    pair_rows = []
    keys = ["group_dir", "mouse_id", "stage"]
    for key, sub in items_df.groupby(keys):
        green_items = sub[sub["channel"] == "green"]
        red_items = sub[sub["channel"] == "red"]
        if green_items.empty or red_items.empty:
            continue
        green_images = [load_image(path) for path in green_items["path"]]
        red_images = [load_image(path) for path in red_items["path"]]
        green = np.mean([robust_rescale(img) for img in green_images], axis=0)
        red = np.mean([robust_rescale(img) for img in red_images], axis=0)
        green, red = crop_pair(green, red)
        merged = np.clip(green + red, 0, None)

        green_mask, _ = positive_mask(green)
        red_mask, _ = positive_mask(red)
        merged_mask, _ = positive_mask(merged)
        intersection = green_mask & red_mask
        union = green_mask | red_mask

        corr = np.nan
        if np.std(green) > 0 and np.std(red) > 0:
            corr = float(np.corrcoef(green.ravel(), red.ravel())[0, 1])

        green_metrics, _, _ = image_metrics(green)
        red_metrics, _, _ = image_metrics(red)
        merged_metrics, _, _ = image_metrics(merged)
        group_dir, mouse_id, stage = key
        base = {
            "group_dir": group_dir,
            "cohort": detect_cohort(group_dir),
            "mouse_id": mouse_id,
            "stage": stage,
            "overlap_jaccard": float(intersection.sum() / max(union.sum(), 1)),
            "green_overlap_fraction": float(intersection.sum() / max(green_mask.sum(), 1)),
            "red_overlap_fraction": float(intersection.sum() / max(red_mask.sum(), 1)),
            "green_red_corr": corr,
            "merged_union_fraction": float(merged_mask.mean()),
        }
        for prefix, metrics in [("green", green_metrics), ("red", red_metrics), ("merged", merged_metrics)]:
            for metric, _ in CHANNEL_METRICS:
                base[f"{prefix}_{metric}"] = metrics[metric]
        pair_rows.append(base)
    return pd.DataFrame(pair_rows)


def collapse_to_mouse(df, value_cols):
    if df.empty:
        return df
    keys = ["cohort", "group_dir", "mouse_id"]
    return df.groupby(keys, as_index=False)[value_cols].mean()


def run_between_cohort_tests(mouse_df, prefixed_metrics, args):
    rows = []
    for metric, label in prefixed_metrics:
        exp = mouse_df.loc[mouse_df["cohort"] == "experimental", metric].to_numpy(dtype=float)
        ctrl = mouse_df.loc[mouse_df["cohort"] == "control", metric].to_numpy(dtype=float)
        if len(exp) < 2 or len(ctrl) < 2:
            continue
        try:
            u_stat, u_p = stats.mannwhitneyu(exp, ctrl, alternative="two-sided")
        except ValueError:
            u_stat, u_p = np.nan, np.nan
        diff = exp - np.mean(ctrl)
        ci_low, ci_high = bootstrap_ci(diff, args.n_bootstrap)
        rows.append(
            {
                "metric": metric,
                "label": label,
                "n_experimental": int(len(exp)),
                "n_control": int(len(ctrl)),
                "experimental_mean": float(np.mean(exp)),
                "control_mean": float(np.mean(ctrl)),
                "mean_difference": float(np.mean(exp) - np.mean(ctrl)),
                "difference_ci95_low": ci_low,
                "difference_ci95_high": ci_high,
                "mannwhitney_u": float(u_stat) if not np.isnan(u_stat) else np.nan,
                "mannwhitney_p": float(u_p) if not np.isnan(u_p) else np.nan,
                "perm_p": independent_permutation_p(exp, ctrl, args.n_permutations),
                "hedges_g": hedges_g(exp, ctrl),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["mannwhitney_fdr_bh"] = benjamini_hochberg(out["mannwhitney_p"].to_numpy())
        out["perm_fdr_bh"] = benjamini_hochberg(out["perm_p"].to_numpy())
    return out


def run_paired_channel_tests(mouse_pair_df, args):
    rows = []
    channel_pairs = [("green", "red"), ("green", "merged"), ("red", "merged")]
    for metric, label in CHANNEL_METRICS:
        for left, right in channel_pairs:
            left_col = f"{left}_{metric}"
            right_col = f"{right}_{metric}"
            values = mouse_pair_df[[left_col, right_col]].dropna()
            if len(values) < 2:
                continue
            left_vals = values[left_col].to_numpy(dtype=float)
            right_vals = values[right_col].to_numpy(dtype=float)
            diff = left_vals - right_vals
            try:
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(left_vals, right_vals, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                wilcoxon_stat, wilcoxon_p = np.nan, np.nan
            ci_low, ci_high = bootstrap_ci(diff, args.n_bootstrap)
            rows.append(
                {
                    "metric": metric,
                    "label": label,
                    "comparison": f"{left}_vs_{right}",
                    "n_pairs": int(len(values)),
                    "left_mean": float(np.mean(left_vals)),
                    "right_mean": float(np.mean(right_vals)),
                    "mean_difference": float(np.mean(diff)),
                    "difference_ci95_low": ci_low,
                    "difference_ci95_high": ci_high,
                    "wilcoxon_stat": float(wilcoxon_stat) if not np.isnan(wilcoxon_stat) else np.nan,
                    "wilcoxon_p": float(wilcoxon_p) if not np.isnan(wilcoxon_p) else np.nan,
                    "perm_p": paired_permutation_p(left_vals, right_vals, args.n_permutations),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["wilcoxon_fdr_bh"] = benjamini_hochberg(out["wilcoxon_p"].to_numpy())
        out["perm_fdr_bh"] = benjamini_hochberg(out["perm_p"].to_numpy())
    return out


def add_significance_bar(ax, x0, x1, y, label, color="#2f2f2f"):
    h = 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0] + 1e-9)
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], color=color, lw=1.2, clip_on=False)
    ax.text((x0 + x1) / 2, y + h * 1.15, label, ha="center", va="bottom", color=color, fontsize=10, weight="bold")


def save_channel_cohort_figure(mouse_pair_df, between_df, out_png):
    if mouse_pair_df.empty or between_df.empty:
        return
    set_plot_style()
    focus = [
        ("green_mean_intensity", "Green mean intensity"),
        ("red_mean_intensity", "Red mean intensity"),
        ("merged_mean_intensity", "Merged mean intensity"),
        ("green_positive_fraction", "Green positive fraction"),
        ("red_positive_fraction", "Red positive fraction"),
        ("merged_positive_fraction", "Merged positive fraction"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    test_map = between_df.set_index("metric")
    for ax, (metric, title) in zip(axes.ravel(), focus):
        exp = mouse_pair_df.loc[mouse_pair_df["cohort"] == "experimental", metric].to_numpy(dtype=float)
        ctrl = mouse_pair_df.loc[mouse_pair_df["cohort"] == "control", metric].to_numpy(dtype=float)
        values = [exp, ctrl]
        colors = ["#2d6a8e", "#b56576"]
        for pos, vals, color in zip([0, 1], values, colors):
            x = np.full(len(vals), pos, dtype=float) + RNG.normal(0, 0.04, size=len(vals))
            ax.scatter(x, vals, s=46, color=color, edgecolor="white", linewidth=0.5, zorder=3)
            if len(vals):
                parts = ax.violinplot(vals, positions=[pos], widths=0.55, showmeans=False, showmedians=False, showextrema=False)
                for body in parts["bodies"]:
                    body.set_facecolor(color)
                    body.set_alpha(0.18)
                    body.set_edgecolor("none")
                ax.plot([pos - 0.1, pos + 0.1], [np.mean(vals), np.mean(vals)], color=color, lw=2.0, zorder=4)
        ax.set_xticks([0, 1], ["Experimental", "Control"])
        ax.set_title(title, pad=10)
        ax.grid(axis="y", alpha=0.65)
        if metric in test_map.index and len(exp) and len(ctrl):
            row = test_map.loc[metric]
            y_max = max(np.max(exp), np.max(ctrl))
            y_min = min(np.min(exp), np.min(ctrl))
            pad = 0.24 * max(y_max - y_min, 0.1)
            ax.set_ylim(y_min - 0.08 * max(y_max - y_min, 0.1), y_max + pad)
            add_significance_bar(ax, 0, 1, y_max + 0.03 * pad, significance_label(row["perm_fdr_bh"]))
            ax.text(
                0.03,
                0.03,
                f"perm-FDR={row['perm_fdr_bh']:.3g}\ng={row['hedges_g']:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.7, linewidth=0),
            )
    fig.suptitle("Red/green channel intensity by cohort", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97], w_pad=2.0, h_pad=2.0)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_overlap_figure(mouse_pair_df, overlap_df, out_png):
    if mouse_pair_df.empty or overlap_df.empty:
        return
    set_plot_style()
    focus = OVERLAP_METRICS[:4]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10))
    test_map = overlap_df.set_index("metric")
    for ax, (metric, title) in zip(axes.ravel(), focus):
        exp = mouse_pair_df.loc[mouse_pair_df["cohort"] == "experimental", metric].to_numpy(dtype=float)
        ctrl = mouse_pair_df.loc[mouse_pair_df["cohort"] == "control", metric].to_numpy(dtype=float)
        values = [exp, ctrl]
        colors = ["#2d6a8e", "#b56576"]
        for pos, vals, color in zip([0, 1], values, colors):
            x = np.full(len(vals), pos, dtype=float) + RNG.normal(0, 0.04, size=len(vals))
            ax.scatter(x, vals, s=46, color=color, edgecolor="white", linewidth=0.5, zorder=3)
            if len(vals):
                parts = ax.violinplot(vals, positions=[pos], widths=0.55, showmeans=False, showmedians=False, showextrema=False)
                for body in parts["bodies"]:
                    body.set_facecolor(color)
                    body.set_alpha(0.18)
                    body.set_edgecolor("none")
                ax.plot([pos - 0.1, pos + 0.1], [np.mean(vals), np.mean(vals)], color=color, lw=2.0, zorder=4)
        ax.set_xticks([0, 1], ["Experimental", "Control"])
        ax.set_title(title, pad=10)
        ax.grid(axis="y", alpha=0.65)
        if metric in test_map.index and len(exp) and len(ctrl):
            row = test_map.loc[metric]
            y_max = max(np.max(exp), np.max(ctrl))
            y_min = min(np.min(exp), np.min(ctrl))
            pad = 0.24 * max(y_max - y_min, 0.1)
            ax.set_ylim(y_min - 0.08 * max(y_max - y_min, 0.1), y_max + pad)
            add_significance_bar(ax, 0, 1, y_max + 0.03 * pad, significance_label(row["perm_fdr_bh"]))
            ax.text(
                0.03,
                0.03,
                f"perm-FDR={row['perm_fdr_bh']:.3g}\ng={row['hedges_g']:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.7, linewidth=0),
            )
    fig.suptitle("Green/red overlap and colocalization by cohort", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97], w_pad=2.0, h_pad=2.0)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_paired_channel_figure(mouse_pair_df, paired_df, out_png):
    if mouse_pair_df.empty or paired_df.empty:
        return
    set_plot_style()
    focus = CHANNEL_METRICS[:4]
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.8))
    pair_labels = [("green", "red"), ("green", "merged"), ("red", "merged")]
    for ax, (metric, title) in zip(axes.ravel(), focus):
        plotted = mouse_pair_df[[f"green_{metric}", f"red_{metric}", f"merged_{metric}"]].dropna()
        if plotted.empty:
            ax.axis("off")
            continue
        positions = [0, 1, 2]
        colors = ["#4c956c", "#b23a48", "#6d597a"]
        for row in plotted.itertuples(index=False):
            vals = [row[0], row[1], row[2]]
            ax.plot(positions, vals, color="#c6c0b3", lw=1.0, alpha=0.8, zorder=1)
        for idx, (pos, color, col) in enumerate(zip(positions, colors, [f"green_{metric}", f"red_{metric}", f"merged_{metric}"])):
            vals = plotted[col].to_numpy(dtype=float)
            ax.scatter(np.full(len(vals), pos) + RNG.normal(0, 0.03, size=len(vals)), vals, s=42, color=color, edgecolor="white", linewidth=0.5, zorder=3)
            ax.plot([pos - 0.09, pos + 0.09], [np.mean(vals), np.mean(vals)], color=color, lw=2.0, zorder=4)
        ax.set_xticks(positions, ["Green", "Red", "Merged"])
        ax.set_title(title, pad=10)
        ax.grid(axis="y", alpha=0.65)
        subset = paired_df[paired_df["metric"] == metric].set_index("comparison")
        ymax = plotted.max().max()
        ymin = plotted.min().min()
        spread = max(ymax - ymin, 0.1)
        pad = 0.58 * spread
        ax.set_ylim(ymin - 0.10 * spread, ymax + pad)
        bar_y = ymax + 0.08 * spread
        bar_step = 0.16 * spread
        for x0, x1, key in [(0, 1, "green_vs_red"), (0, 2, "green_vs_merged"), (1, 2, "red_vs_merged")]:
            if key not in subset.index:
                continue
            add_significance_bar(ax, x0, x1, bar_y, significance_label(subset.loc[key, "perm_fdr_bh"]))
            bar_y += bar_step
        ax.text(
            0.03,
            0.03,
            f"n={len(plotted)} paired mice",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.7, linewidth=0),
        )
    fig.suptitle("Within-mouse comparison of green, red, and merged channel signal", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97], w_pad=2.0, h_pad=2.0)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def build_interpretation(channel_between_df, overlap_between_df):
    lines = []
    if not channel_between_df.empty:
        sig = channel_between_df[channel_between_df["perm_fdr_bh"] < 0.05].sort_values("perm_fdr_bh")
        if not sig.empty:
            row = sig.iloc[0]
            direction = "higher" if row["mean_difference"] > 0 else "lower"
            lines.append(
                f"Strongest cohort difference was {row['label'].lower()} ({row['metric']}), with experimental mice showing {direction} values than controls (perm-FDR={row['perm_fdr_bh']:.3g})."
            )
        else:
            lines.append("No channel-intensity metric crossed the FDR < 0.05 threshold between cohorts.")
    if not overlap_between_df.empty:
        sig = overlap_between_df[overlap_between_df["perm_fdr_bh"] < 0.05].sort_values("perm_fdr_bh")
        if not sig.empty:
            row = sig.iloc[0]
            direction = "higher" if row["mean_difference"] > 0 else "lower"
            lines.append(
                f"The clearest overlap effect was {row['label'].lower()}, with experimental mice showing {direction} green/red colocalization than controls (perm-FDR={row['perm_fdr_bh']:.3g})."
            )
        else:
            lines.append("Green/red overlap metrics did not show an FDR-significant cohort split.")
    return "\n".join(lines)


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    items_df = discover_channel_movies(args.corrected_dir)
    if items_df.empty:
        print("No corrected green/red channel images found.")
        return

    channel_df = build_channel_table(items_df)
    channel_df.to_csv(os.path.join(args.out_dir, "channel_image_metrics_long.csv"), index=False)

    pair_df = build_pair_table(items_df)
    pair_df.to_csv(os.path.join(args.out_dir, "channel_pair_metrics_long.csv"), index=False)
    if pair_df.empty:
        print("No matched green/red channel pairs found.")
        return

    mouse_channel_cols = []
    for prefix in ["green", "red", "merged"]:
        mouse_channel_cols.extend([f"{prefix}_{metric}" for metric, _ in CHANNEL_METRICS])
    overlap_cols = [metric for metric, _ in OVERLAP_METRICS]
    mouse_pair_df = collapse_to_mouse(pair_df, mouse_channel_cols + overlap_cols)
    mouse_pair_df.to_csv(os.path.join(args.out_dir, "channel_pair_metrics_mouse_level.csv"), index=False)

    channel_metric_map = []
    for prefix, prefix_label in [("green", "Green"), ("red", "Red"), ("merged", "Merged")]:
        for metric, label in CHANNEL_METRICS:
            channel_metric_map.append((f"{prefix}_{metric}", f"{prefix_label} {label}"))
    overlap_metric_map = [(metric, label) for metric, label in OVERLAP_METRICS]

    between_channels = run_between_cohort_tests(mouse_pair_df, channel_metric_map, args)
    between_channels.to_csv(os.path.join(args.out_dir, "channel_between_cohorts.csv"), index=False)
    between_overlap = run_between_cohort_tests(mouse_pair_df, overlap_metric_map, args)
    between_overlap.to_csv(os.path.join(args.out_dir, "channel_overlap_between_cohorts.csv"), index=False)
    paired_channels = run_paired_channel_tests(mouse_pair_df, args)
    paired_channels.to_csv(os.path.join(args.out_dir, "channel_within_mouse_tests.csv"), index=False)

    save_channel_cohort_figure(mouse_pair_df, between_channels, os.path.join(args.out_dir, "channel_intensity_by_cohort.png"))
    save_overlap_figure(mouse_pair_df, between_overlap, os.path.join(args.out_dir, "channel_overlap_by_cohort.png"))
    save_paired_channel_figure(mouse_pair_df, paired_channels, os.path.join(args.out_dir, "channel_within_mouse.png"))

    with open(os.path.join(args.out_dir, "channel_interpretation.txt"), "w", encoding="utf-8") as handle:
        handle.write(build_interpretation(between_channels, between_overlap) + "\n")

    print(f"Saved channel statistics to {args.out_dir}")


if __name__ == "__main__":
    main()
