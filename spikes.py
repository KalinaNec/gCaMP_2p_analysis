import argparse
import glob
import os

import imageio.v2 as imageio
import matplotlib
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, resample
from skimage import measure, morphology, segmentation


matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract GCaMP traces, detect events, and render GIFs for a few large high-spiking cells."
    )
    parser.add_argument("--movie-dir", default="./corrected")
    parser.add_argument("--mask-dir", default="./cellpose")
    parser.add_argument("--out-dir", default="./activity")
    parser.add_argument("--fr", type=float, default=1.0)
    parser.add_argument("--baseline-q", type=float, default=20.0)
    parser.add_argument("--baseline-win-s", type=float, default=30.0)
    parser.add_argument("--neuropil-inner-px", type=int, default=3)
    parser.add_argument("--neuropil-outer-px", type=int, default=12)
    parser.add_argument("--neuropil-scale", type=float, default=0.7)
    parser.add_argument("--peak-z", type=float, default=2.5)
    parser.add_argument("--peak-prominence", type=float, default=0.8)
    parser.add_argument("--min-event-dist-s", type=float, default=2.0)
    parser.add_argument("--smooth-sigma", type=float, default=1.2)
    parser.add_argument("--top-gif-cells", type=int, default=3)
    parser.add_argument("--gif-fps", type=float, default=4.0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def running_percentile(x, win, q):
    win = max(5, int(win))
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x, dtype=np.float32)
    for idx in range(len(x)):
        out[idx] = np.percentile(xp[idx : idx + win], q)
    return out


def robust_z(x, eps=1e-9):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return (x - med) / (1.4826 * mad)


def discover_movies(movie_dir, mask_dir, out_dir):
    patterns = [
        os.path.join(movie_dir, "**", "*_rigid.tif"),
        os.path.join(movie_dir, "**", "*_rigid.TIF"),
    ]
    movie_paths = []
    for pattern in patterns:
        movie_paths.extend(glob.glob(pattern, recursive=True))

    movie_abs = os.path.abspath(movie_dir)
    mask_abs = os.path.abspath(mask_dir)
    out_abs = os.path.abspath(out_dir)
    unique = sorted(set(movie_paths))
    kept = []
    for path in unique:
        path_abs = os.path.abspath(path)
        if os.path.commonpath([path_abs, out_abs]) == out_abs:
            continue
        if os.path.commonpath([path_abs, mask_abs]) == mask_abs:
            continue
        if f"{os.sep}_tmp_for_caiman{os.sep}" in path_abs:
            continue
        kept.append((path, os.path.relpath(path, start=movie_abs)))
    return kept


def load_movie(movie_path):
    movie = tifffile.imread(movie_path).astype(np.float32)
    if movie.ndim == 2:
        movie = movie[None, :, :]
    return movie


def build_neuropil_mask(mask_labels, roi_id, inner_px, outer_px):
    roi = mask_labels == roi_id
    if not np.any(roi):
        return roi

    outer = morphology.binary_dilation(roi, morphology.disk(outer_px))
    inner = morphology.binary_dilation(roi, morphology.disk(inner_px))
    ring = outer & ~inner & (mask_labels == 0)
    return ring


def extract_traces(movie, masks, args):
    n_rois = int(masks.max())
    all_cells = masks > 0
    raw = np.zeros((n_rois, movie.shape[0]), dtype=np.float32)
    neuropil = np.zeros_like(raw)
    corrected = np.zeros_like(raw)
    stats = []

    for roi_id in range(1, n_rois + 1):
        roi = masks == roi_id
        if not np.any(roi):
            continue

        ring = build_neuropil_mask(masks, roi_id, args.neuropil_inner_px, args.neuropil_outer_px)
        if ring.sum() < max(roi.sum(), 20):
            # When local packing is tight, fall back to a simpler surrounding ring.
            fallback = morphology.binary_dilation(roi, morphology.disk(args.neuropil_outer_px))
            ring = fallback & ~roi & ~all_cells

        raw_trace = movie[:, roi].mean(axis=1)
        if np.any(ring):
            neuropil_trace = movie[:, ring].mean(axis=1)
        else:
            neuropil_trace = np.zeros(movie.shape[0], dtype=np.float32)

        corrected_trace = raw_trace - args.neuropil_scale * neuropil_trace
        corrected_trace -= np.percentile(corrected_trace, 1)
        corrected_trace = np.clip(corrected_trace, 1e-3, None)

        raw[roi_id - 1] = raw_trace
        neuropil[roi_id - 1] = neuropil_trace
        corrected[roi_id - 1] = corrected_trace

        props = measure.regionprops(roi.astype(np.uint8))[0]
        stats.append(
            {
                "roi_id": roi_id,
                "area_px": int(roi.sum()),
                "centroid_y": float(props.centroid[0]),
                "centroid_x": float(props.centroid[1]),
                "neuropil_area_px": int(ring.sum()),
            }
        )

    return raw, neuropil, corrected, pd.DataFrame(stats)


def compute_dff(corrected, args):
    win = max(int(args.baseline_win_s * args.fr), 5)
    baseline = np.vstack([running_percentile(trace, win, args.baseline_q) for trace in corrected])
    baseline_floor = np.maximum(np.median(corrected, axis=1, keepdims=True) * 0.2, 1.0)
    baseline = np.maximum(baseline, baseline_floor)
    dff = (corrected - baseline) / baseline
    smooth = gaussian_filter1d(dff, sigma=args.smooth_sigma, axis=1, mode="nearest")
    zscore = np.vstack([robust_z(trace) for trace in smooth])
    return baseline, dff, smooth, zscore


def detect_events(dff, smooth, zscore, args):
    min_dist = max(int(args.min_event_dist_s * args.fr), 1)
    events = {}
    rows = []
    summary = []

    for idx in range(dff.shape[0]):
        roi_name = f"ROI_{idx + 1}"
        peaks, props = find_peaks(
            zscore[idx],
            height=args.peak_z,
            prominence=args.peak_prominence,
            distance=min_dist,
        )
        peaks = peaks.astype(int)
        events[roi_name] = peaks

        prominences = props.get("prominences", np.zeros(len(peaks), dtype=float))
        heights = props.get("peak_heights", np.zeros(len(peaks), dtype=float))

        for peak, prominence, height in zip(peaks, prominences, heights):
            rows.append(
                {
                    "roi": roi_name,
                    "frame": int(peak),
                    "time_s": float(peak / args.fr),
                    "dff": float(dff[idx, peak]),
                    "smooth_dff": float(smooth[idx, peak]),
                    "zscore": float(height),
                    "prominence": float(prominence),
                }
            )

        summary.append(
            {
                "roi": roi_name,
                "event_count": int(len(peaks)),
                "event_rate_hz": float(len(peaks) / max(dff.shape[1] / args.fr, 1e-6)),
                "peak_dff_max": float(dff[idx, peaks].max()) if len(peaks) else 0.0,
                "peak_dff_mean": float(dff[idx, peaks].mean()) if len(peaks) else 0.0,
                "trace_std": float(np.std(dff[idx])),
            }
        )

    return events, pd.DataFrame(rows), pd.DataFrame(summary)


def save_trace_tables(raw, neuropil, corrected, baseline, dff, out_dir):
    roi_names = [f"ROI_{idx + 1}" for idx in range(dff.shape[0])]
    pd.DataFrame(dff.T, columns=roi_names).to_csv(os.path.join(out_dir, "traces_dff.csv"), index=False)
    pd.DataFrame(corrected.T, columns=roi_names).to_csv(
        os.path.join(out_dir, "traces_corrected.csv"), index=False
    )
    pd.DataFrame(raw.T, columns=roi_names).to_csv(os.path.join(out_dir, "traces_raw.csv"), index=False)
    pd.DataFrame(neuropil.T, columns=roi_names).to_csv(
        os.path.join(out_dir, "traces_neuropil.csv"), index=False
    )
    pd.DataFrame(baseline.T, columns=roi_names).to_csv(
        os.path.join(out_dir, "traces_baseline.csv"), index=False
    )


def save_event_tables(events, events_long, out_dir):
    max_len = max((len(v) for v in events.values()), default=0)
    wide = {
        key: np.pad(value.astype(float), (0, max_len - len(value)), constant_values=np.nan)
        for key, value in events.items()
    }
    pd.DataFrame(wide).to_csv(os.path.join(out_dir, "events.csv"), index=False)
    events_long.to_csv(os.path.join(out_dir, "events_long.csv"), index=False)


def save_summary_plot(dff, events, out_png, fr):
    n_show = min(30, dff.shape[0])
    order = np.argsort([len(events[f"ROI_{idx + 1}"]) for idx in range(dff.shape[0])])[::-1][:n_show]
    t = np.arange(dff.shape[1]) / fr

    fig, ax = plt.subplots(figsize=(14, 9))
    offset = 4.0
    for row, idx in enumerate(order):
        trace = robust_z(dff[idx])
        trace = np.clip(trace, -4, 8) + row * offset
        ax.plot(t, trace, linewidth=1.0, color="#1f77b4")
        peaks = events[f"ROI_{idx + 1}"]
        if len(peaks):
            ax.scatter(t[peaks], trace[peaks], color="crimson", s=15)
        ax.text(t[-1] + 0.5 / fr, trace[-1], f"ROI_{idx + 1}", fontsize=8, va="center")

    ax.set_title("Top calcium traces ranked by event count")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_heatmap(dff, out_png):
    order = np.argsort(np.max(dff, axis=1))[::-1]
    clipped = np.clip(dff[order], *np.percentile(dff, [2, 98]))
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(clipped, aspect="auto", cmap="magma")
    ax.set_title("Delta F/F heatmap")
    ax.set_xlabel("Frame")
    ax.set_ylabel("ROI ranked by peak")
    fig.colorbar(im, ax=ax, label="Delta F/F")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_eeg_like_plot(dff, events, out_png, fr, n_show=12):
    if dff.size == 0:
        return

    ranking = np.argsort([len(events[f"ROI_{idx + 1}"]) for idx in range(dff.shape[0])])[::-1][: min(n_show, dff.shape[0])]
    traces = []
    for idx in ranking:
        trace = gaussian_filter1d(dff[idx], sigma=1.0)
        traces.append(robust_z(trace))

    stack = np.vstack(traces)
    eeg_like = np.mean(stack, axis=0)
    if eeg_like.size > 600:
        eeg_like = resample(eeg_like, 600)
        t = np.linspace(0, dff.shape[1] / fr, eeg_like.size, endpoint=False)
    else:
        t = np.arange(eeg_like.size) / fr

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(t, eeg_like, color="black", linewidth=1.4)

    peak_times = []
    peak_vals = []
    for idx in ranking:
        peaks = events[f"ROI_{idx + 1}"] / fr
        peak_times.extend(peaks.tolist())
    if len(peak_times) > 0:
        peak_times = np.asarray(peak_times, dtype=float)
        interp_vals = np.interp(peak_times, t, eeg_like)
        ax.scatter(peak_times, interp_vals, color="crimson", s=18, alpha=0.65, label="ROI events")

    ax.axhline(0, color="0.7", linewidth=0.8)
    ax.set_title("EEG-like population activity")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean robust z-score")
    if len(peak_times) > 0:
        ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def select_gif_rois(roi_summary, movie, masks, top_n):
    if roi_summary.empty:
        return []

    ranked = roi_summary.copy()
    visibility_scores = []
    movie_mean = movie.mean(axis=0)
    for row in ranked.itertuples(index=False):
        roi = masks == int(row.roi_id)
        if not np.any(roi):
            visibility_scores.append(0.0)
            continue
        # Prefer cells that are not only active, but also easy to see in the movie.
        ring = morphology.binary_dilation(roi, morphology.disk(8)) & ~roi
        roi_signal = float(movie_mean[roi].mean())
        ring_signal = float(movie_mean[ring].mean()) if np.any(ring) else float(np.median(movie_mean))
        visibility_scores.append(max(roi_signal - ring_signal, 0.0))

    ranked["visibility_raw"] = visibility_scores
    ranked["area_score"] = ranked["area_px"] / max(ranked["area_px"].max(), 1.0)
    ranked["spike_score"] = ranked["peak_dff_max"] / max(ranked["peak_dff_max"].max(), 1.0)
    ranked["event_score"] = ranked["event_count"] / max(ranked["event_count"].max(), 1.0)
    ranked["visibility_score"] = ranked["visibility_raw"] / max(ranked["visibility_raw"].max(), 1.0)
    ranked["gif_priority"] = (
        0.35 * ranked["spike_score"]
        + 0.25 * ranked["area_score"]
        + 0.20 * ranked["visibility_score"]
        + 0.20 * ranked["event_score"]
    )
    ranked = ranked.sort_values(
        ["gif_priority", "peak_dff_max", "visibility_raw", "area_px", "event_count"],
        ascending=False,
    )
    return ranked["roi_id"].head(top_n).astype(int).tolist()


def normalize_movie_for_gif(movie):
    lo, hi = np.percentile(movie, (1, 99.5))
    scaled = np.clip((movie - lo) / max(hi - lo, 1e-6), 0, 1)
    return (scaled * 255).astype(np.uint8)


def choose_anchor(center_x, center_y, width, height):
    horiz = -1 if center_x > width * 0.55 else 1
    vert = -1 if center_y > height * 0.55 else 1
    text_x = np.clip(center_x + horiz * width * 0.18, width * 0.08, width * 0.92)
    text_y = np.clip(center_y + vert * height * 0.18, height * 0.10, height * 0.90)
    return text_x, text_y


def render_roi_gif(movie, masks, dff, events, roi_id, out_path, args):
    roi = masks == roi_id
    if not np.any(roi):
        return

    boundary = segmentation.find_boundaries(roi, mode="outer")
    yx = np.argwhere(roi)
    center_y, center_x = yx.mean(axis=0)
    text_x, text_y = choose_anchor(center_x, center_y, movie.shape[2], movie.shape[1])
    event_frames = set(events[f"ROI_{roi_id}"].tolist())
    t = np.arange(dff.shape[1]) / args.fr
    movie_u8 = normalize_movie_for_gif(movie)

    frames = []
    for frame_idx in range(movie.shape[0]):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})

        axes[0].imshow(movie_u8[frame_idx], cmap="gray", vmin=0, vmax=255)
        axes[0].imshow(np.ma.masked_where(~boundary, boundary), cmap="spring", alpha=0.9)
        marker_color = "red" if frame_idx in event_frames else "gold"
        axes[0].annotate(
            f"ROI {roi_id}",
            xy=(center_x, center_y),
            xytext=(text_x, text_y),
            color="white",
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5, linewidth=0),
            arrowprops=dict(arrowstyle="->", color=marker_color, lw=1.7, shrinkA=3, shrinkB=4),
        )
        if frame_idx in event_frames:
            axes[0].text(
                6,
                18,
                "SPIKE",
                color="red",
                fontsize=11,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, linewidth=0),
            )
        axes[0].set_title(f"Frame {frame_idx}")
        axes[0].axis("off")

        axes[1].plot(t, dff[roi_id - 1], color="0.82", linewidth=1.2)
        axes[1].plot(t[: frame_idx + 1], dff[roi_id - 1, : frame_idx + 1], color="#1f77b4", linewidth=1.8)
        peak_frames = events[f"ROI_{roi_id}"]
        if len(peak_frames):
            axes[1].scatter(t[peak_frames], dff[roi_id - 1, peak_frames], color="crimson", s=24)
        axes[1].annotate(
            "",
            xy=(t[frame_idx], dff[roi_id - 1, frame_idx]),
            xytext=(t[frame_idx], dff[roi_id - 1, frame_idx] + 0.12 * max(np.std(dff[roi_id - 1]), 1.0)),
            arrowprops=dict(arrowstyle="-|>", color=marker_color, lw=1.8),
            zorder=5,
        )
        axes[1].set_title(f"ROI {roi_id} trace | events={len(event_frames)}")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Delta F/F")
        right = t[-1] if len(t) > 1 else t[0] + 1.0 / max(args.fr, 1e-6)
        axes[1].set_xlim(t[0], right)
        ymin = np.percentile(dff[roi_id - 1], 1)
        ymax = np.percentile(dff[roi_id - 1], 99.5)
        if np.isclose(ymin, ymax):
            ymax = ymin + 1.0
        axes[1].set_ylim(ymin - 0.1 * abs(ymax - ymin), ymax + 0.15 * abs(ymax - ymin))

        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())
        plt.close(fig)

    imageio.mimsave(out_path, frames, duration=1.0 / args.gif_fps, loop=0)


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    movie_items = discover_movies(args.movie_dir, args.mask_dir, args.out_dir)
    if not movie_items:
        print("No motion-corrected movies found.")
        return

    for movie_path, rel in movie_items:
        rel_dir = os.path.dirname(rel)
        base = os.path.splitext(os.path.basename(rel))[0]
        name = base.replace("_rigid", "")
        mask_path = os.path.join(args.mask_dir, rel_dir, f"{base}_masks.tif")
        out_dir = os.path.join(args.out_dir, rel_dir, name)
        gif_dir = os.path.join(out_dir, "gifs")
        summary_path = os.path.join(out_dir, "roi_summary.csv")

        if not os.path.exists(mask_path):
            print(f"SKIP (no masks): {rel}")
            continue
        if args.skip_existing and os.path.exists(summary_path):
            print(f"SKIP (exists): {rel}")
            continue

        ensure_dir(out_dir)
        ensure_dir(gif_dir)

        print(f"Processing: {rel}")
        movie = load_movie(movie_path)
        masks = tifffile.imread(mask_path).astype(np.int32)

        if masks.shape != movie.shape[1:]:
            print(f"  SKIP: mask shape {masks.shape} does not match movie shape {movie.shape[1:]}")
            continue
        if int(masks.max()) == 0:
            print("  SKIP: empty mask")
            continue

        raw, neuropil, corrected, roi_stats = extract_traces(movie, masks, args)
        baseline, dff, smooth, zscore = compute_dff(corrected, args)
        events, events_long, event_summary = detect_events(dff, smooth, zscore, args)

        event_summary = event_summary.copy()
        event_summary["roi_id"] = event_summary["roi"].str.replace("ROI_", "", regex=False).astype(int)
        roi_summary = roi_stats.merge(event_summary, on="roi_id", how="left")
        roi_summary = roi_summary.sort_values(["event_count", "peak_dff_max"], ascending=False)

        save_trace_tables(raw, neuropil, corrected, baseline, dff, out_dir)
        save_event_tables(events, events_long, out_dir)
        roi_summary.to_csv(summary_path, index=False)
        save_summary_plot(dff, events, os.path.join(out_dir, "stacked_traces.png"), args.fr)
        save_heatmap(dff, os.path.join(out_dir, "heatmap.png"))
        save_eeg_like_plot(dff, events, os.path.join(out_dir, "eeg_like.png"), args.fr)

        # Keep only a few representative cells per movie.
        top_roi_ids = select_gif_rois(roi_summary, movie, masks, args.top_gif_cells)
        for roi_id in top_roi_ids:
            render_roi_gif(
                movie,
                masks,
                dff,
                events,
                int(roi_id),
                os.path.join(gif_dir, f"ROI_{int(roi_id):03d}.gif"),
                args,
            )

        print(f"  saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
