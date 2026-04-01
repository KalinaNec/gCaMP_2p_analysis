import argparse
import glob
import os

import matplotlib
import numpy as np
import pandas as pd
import tifffile
from cellpose import models
from skimage import exposure, filters, measure, morphology, segmentation


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect GCaMP somatic ROIs from motion-corrected TIFF movies."
    )
    parser.add_argument("--input-dir", default="./corrected")
    parser.add_argument("--output-dir", default="./cellpose")
    parser.add_argument("--diameter", type=float, default=12.0)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=-0.5)
    parser.add_argument("--model-type", default="cyto3")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--min-area", type=int, default=30)
    parser.add_argument("--max-area", type=int, default=1200)
    parser.add_argument("--min-solidity", type=float, default=0.7)
    parser.add_argument("--max-eccentricity", type=float, default=0.97)
    parser.add_argument("--projection", choices=("mean", "median", "max"), default="mean")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_movies(input_dir, output_dir):
    patterns = [
        os.path.join(input_dir, "**", "*_rigid.tif"),
        os.path.join(input_dir, "**", "*_rigid.TIF"),
    ]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern, recursive=True))

    out_abs = os.path.abspath(output_dir)
    unique = sorted(set(paths))
    return [
        path
        for path in unique
        if os.path.commonpath([os.path.abspath(path), out_abs]) != out_abs
        and f"{os.sep}_tmp_for_caiman{os.sep}" not in os.path.abspath(path)
    ]


def load_projection(movie_path, projection):
    movie = tifffile.imread(movie_path).astype(np.float32)
    if movie.ndim == 2:
        return movie
    if projection == "max":
        return movie.max(axis=0)
    if projection == "median":
        return np.median(movie, axis=0)
    return movie.mean(axis=0)


def preprocess_projection(image):
    p_low, p_high = np.percentile(image, (1, 99.8))
    clipped = np.clip(image, p_low, p_high)
    scaled = exposure.rescale_intensity(clipped, in_range=(p_low, p_high), out_range=(0, 1))
    background = filters.gaussian(scaled, sigma=8, preserve_range=True)
    enhanced = np.clip(scaled - 0.7 * background, 0, None)
    enhanced = exposure.equalize_adapthist(enhanced, clip_limit=0.02)
    return enhanced.astype(np.float32)


def relabel_and_filter_masks(raw_masks, min_area, max_area, min_solidity, max_eccentricity):
    filtered = np.zeros_like(raw_masks, dtype=np.int32)
    rows = []
    next_id = 1

    for region in measure.regionprops(raw_masks):
        area = int(region.area)
        if area < min_area or area > max_area:
            continue
        if region.solidity < min_solidity:
            continue
        if region.eccentricity > max_eccentricity:
            continue

        min_row, min_col, max_row, max_col = region.bbox
        if min_row == 0 or min_col == 0:
            continue
        if max_row >= raw_masks.shape[0] or max_col >= raw_masks.shape[1]:
            continue

        filtered[raw_masks == region.label] = next_id
        rows.append(
            {
                "roi_id": next_id,
                "source_label": int(region.label),
                "area_px": area,
                "centroid_y": float(region.centroid[0]),
                "centroid_x": float(region.centroid[1]),
                "eccentricity": float(region.eccentricity),
                "solidity": float(region.solidity),
                "bbox_min_row": int(min_row),
                "bbox_min_col": int(min_col),
                "bbox_max_row": int(max_row),
                "bbox_max_col": int(max_col),
            }
        )
        next_id += 1

    return filtered, pd.DataFrame(rows)


def place_labels(stats_df, shape):
    if stats_df.empty:
        return {}

    h, w = shape
    placements = {}
    occupied = []
    label_w = max(18, int(0.05 * w))
    label_h = max(14, int(0.03 * h))
    # Push labels away from dense soma clusters and pick the least-overlapping spot.
    directions = [
        np.array([-1.0, -1.0]),
        np.array([1.0, -1.0]),
        np.array([1.0, 1.0]),
        np.array([-1.0, 1.0]),
        np.array([0.0, -1.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([-1.0, 0.0]),
    ]

    for row in stats_df.sort_values(["centroid_y", "centroid_x"]).itertuples(index=False):
        cx = float(row.centroid_x)
        cy = float(row.centroid_y)
        bbox = np.array([row.bbox_min_col, row.bbox_min_row, row.bbox_max_col, row.bbox_max_row], dtype=float)

        best = None
        best_score = None
        bbox_radius = max(abs(cx - bbox[0]), abs(cx - bbox[2]), abs(cy - bbox[1]), abs(cy - bbox[3]))
        base_radius = max(label_w * 0.9, label_h * 1.2, bbox_radius + 12.0)
        for radius_scale in (1.4, 2.0, 2.8, 3.6, 4.6):
            radius = base_radius * radius_scale
            for direction in directions:
                norm = np.linalg.norm(direction)
                if norm == 0:
                    continue
                unit = direction / norm
                lx = np.clip(cx + unit[0] * radius, label_w / 2, w - label_w / 2)
                ly = np.clip(cy + unit[1] * radius, label_h / 2, h - label_h / 2)
                rect = np.array([lx - label_w / 2, ly - label_h / 2, lx + label_w / 2, ly + label_h / 2], dtype=float)

                overlap_penalty = 0.0
                if rect[0] < bbox[2] + 4 and rect[2] > bbox[0] - 4 and rect[1] < bbox[3] + 4 and rect[3] > bbox[1] - 4:
                    overlap_penalty += 1e6

                for other in occupied:
                    if rect[0] < other[2] and rect[2] > other[0] and rect[1] < other[3] and rect[3] > other[1]:
                        overlap_penalty += 1e6

                score = overlap_penalty + abs(lx - cx) + abs(ly - cy)
                if best_score is None or score < best_score:
                    best = (lx, ly, rect)
                    best_score = score

        if best is None:
            best = (cx, max(label_h / 2, cy - base_radius), np.array([cx - label_w / 2, cy - base_radius - label_h / 2, cx + label_w / 2, cy - base_radius + label_h / 2], dtype=float))

        placements[int(row.roi_id)] = (best[0], best[1])
        occupied.append(best[2])

    return placements


def save_overlay(preprocessed, masks, stats_df, out_png, title):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(preprocessed, cmap="gray")
    boundary = segmentation.find_boundaries(masks, mode="outer")
    ax.imshow(np.ma.masked_where(~boundary, boundary), cmap="autumn", alpha=0.8)
    placements = place_labels(stats_df, masks.shape)

    for row in stats_df.itertuples(index=False):
        label_x, label_y = placements.get(int(row.roi_id), (row.centroid_x, row.centroid_y))
        ax.annotate(
            str(row.roi_id),
            xy=(row.centroid_x, row.centroid_y),
            xytext=(label_x, label_y),
            color="cyan",
            fontsize=7,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.35, linewidth=0),
            arrowprops=dict(arrowstyle="-", color="cyan", lw=0.7, alpha=0.8, shrinkA=2, shrinkB=4),
        )

    ax.set_title(f"{title} | ROIs={len(stats_df)}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    model = models.CellposeModel(model_type=args.model_type, gpu=args.gpu)
    input_abs = os.path.abspath(args.input_dir)
    movie_paths = find_movies(args.input_dir, args.output_dir)

    if not movie_paths:
        print("No motion-corrected movies found.")
        return

    for movie_path in movie_paths:
        rel = os.path.relpath(movie_path, start=input_abs)
        rel_dir = os.path.dirname(rel)
        base = os.path.splitext(os.path.basename(rel))[0]

        out_dir = os.path.join(args.output_dir, rel_dir)
        ensure_dir(out_dir)

        mask_path = os.path.join(out_dir, f"{base}_masks.tif")
        stats_path = os.path.join(out_dir, f"{base}_roi_stats.csv")
        overlay_path = os.path.join(out_dir, f"{base}_overlay.png")

        if args.skip_existing and os.path.exists(mask_path) and os.path.exists(stats_path):
            print(f"SKIP (exists): {rel}")
            continue

        print(f"Cellpose ROI detection: {rel}")
        projection = load_projection(movie_path, args.projection)
        preprocessed = preprocess_projection(projection)

        masks, *_ = model.eval(
            preprocessed,
            diameter=args.diameter,
            channels=[0, 0],
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
        )

        # remove_small_objects is only useful when Cellpose returned multiple labels
        if int(np.max(masks)) > 1:
            masks = morphology.remove_small_objects(masks.astype(np.int32), min_size=args.min_area)
        else:
            masks = masks.astype(np.int32)
        masks, stats_df = relabel_and_filter_masks(
            masks,
            min_area=args.min_area,
            max_area=args.max_area,
            min_solidity=args.min_solidity,
            max_eccentricity=args.max_eccentricity,
        )

        tifffile.imwrite(mask_path, masks.astype(np.int32))
        stats_df.to_csv(stats_path, index=False)
        save_overlay(preprocessed, masks, stats_df, overlay_path, base)
        print(f"  saved {mask_path} with {len(stats_df)} ROIs")


if __name__ == "__main__":
    main()
