import os, glob, time, re
import numpy as np
import pandas as pd
import tifffile
from aicspylibczi import CziFile

from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as caiman_params
from caiman.mmapping import load_memmap

INPUT_DIR  = "./"
OUTPUT_DIR = "./corrected"
FR = 1

MAX_SHIFTS = (4, 4)
BORDER_NAN = "copy"
SKIP_EXISTING = True

ONLY_PREPOST = True
Z_STRATEGY = "max"


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def is_prepost(path: str) -> bool:
    name = os.path.basename(path).lower()
    return bool(re.search(r"(pre|post|green|red)\.(tif|tiff|czi)$", name))


def find_inputs_recursive(root: str):
    patterns = [
        os.path.join(root, "**", "*.tif"),
        os.path.join(root, "**", "*.tiff"),
        os.path.join(root, "**", "*.czi"),
        os.path.join(root, "**", "*.TIF"),
        os.path.join(root, "**", "*.TIFF"),
        os.path.join(root, "**", "*.CZI"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(set(files))

    if ONLY_PREPOST:
        files = [f for f in files if is_prepost(f)]

    out_abs = os.path.abspath(OUTPUT_DIR)
    kept = []
    for f in files:
        f_abs = os.path.abspath(f)
        if os.path.commonpath([f_abs, out_abs]) == out_abs:
            continue
        kept.append(f)
    files = kept

    return files


def rel_out_path(in_path: str, out_root: str, in_root: str, suffix="_rigid.tif") -> str:
    rel = os.path.relpath(in_path, start=os.path.abspath(in_root))
    rel_dir = os.path.dirname(rel)
    base = os.path.splitext(os.path.basename(rel))[0]
    out_dir = os.path.join(out_root, rel_dir)
    ensure_dir(out_dir)
    return os.path.join(out_dir, f"{base}{suffix}")


def find_mmap_path(mc):
    preferred = ["fname_tot_els", "fname_tot_rig", "fname_tot", "mmap_file", "fname_tot_rig1"]
    for attr in preferred:
        if hasattr(mc, attr):
            val = getattr(mc, attr)
            if isinstance(val, str):
                candidates = [val]
            elif isinstance(val, (list, tuple)):
                candidates = [x for x in val if isinstance(x, str)]
            else:
                candidates = []
            for p in candidates:
                if isinstance(p, str) and p.endswith(".mmap") and os.path.exists(p):
                    return p

    for attr in dir(mc):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(mc, attr)
        except Exception:
            continue
        if isinstance(val, str) and val.endswith(".mmap") and os.path.exists(val):
            return val
        if isinstance(val, (list, tuple)):
            for x in val:
                if isinstance(x, str) and x.endswith(".mmap") and os.path.exists(x):
                    return x

    raise RuntimeError("Could not find produced .mmap file on MotionCorrect object.")


def load_mmap_movie(mmap_file: str):
    Yr, dims, T = load_memmap(mmap_file)
    dims = tuple(dims)
    if Yr.shape[0] == int(np.prod(dims)):
        movie = Yr.T.reshape((T, *dims), order="F")
    else:
        movie = Yr.reshape((T, *dims), order="F")
    return movie


def mean_sharpness(img):
    gy, gx = np.gradient(img.astype(np.float32))
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def to_TYX_stack(arr):
    a = np.asarray(arr)
    a = np.squeeze(a)

    if a.ndim < 3:
        raise ValueError(f"Expected >=3D array for a movie, got shape {a.shape}")

    yx = a.shape[-2], a.shape[-1]
    leading = int(np.prod(a.shape[:-2]))
    out = a.reshape((leading, *yx))
    return out


def read_czi_movie_TYX(path: str, z_strategy="first"):
    czi = CziFile(path)

    dims = czi.dims
    size = czi.size

    if isinstance(size, dict):
        size_dict = size
    else:
        dims_clean = [d for d in dims if d.isalpha()]
        if len(size) < len(dims_clean):
            dims_clean = list(dims)[:len(size)]
        size_dict = {d: int(size[i]) for i, d in enumerate(dims_clean)}

    index = {}
    for d, n in size_dict.items():
        if d in ("X", "Y"):
            continue
        if n and n > 0:
            index[d] = 0

    T = int(size_dict.get("T", 1))

    frames = []
    for t in range(T):
        if "T" in size_dict:
            index["T"] = t

        plane = czi.read_image(**index)[0]
        plane = np.asarray(plane)
        plane = np.squeeze(plane)

        if plane.ndim > 2:
            y, x = plane.shape[-2], plane.shape[-1]
            plane = plane.reshape((-1, y, x))
            plane = plane.max(axis=0) if z_strategy == "max" else plane[0]

        if plane.ndim != 2:
            raise ValueError(f"CZI frame is not 2D after reduction: shape={plane.shape}")

        frames.append(plane)

    movie = np.stack(frames, axis=0)
    return movie


def read_movie_any(in_path: str, z_strategy="first"):
    ext = os.path.splitext(in_path)[1].lower()
    if ext == ".czi":
        movie = read_czi_movie_TYX(in_path, z_strategy=z_strategy)
        return movie, movie.dtype, f"{movie.shape} (decoded from CZI)"

    raw = tifffile.imread(in_path)

    if raw.ndim == 2:
        mov = raw[None, :, :]
        return mov, raw.dtype, str(getattr(raw, "shape", None))

    mov = to_TYX_stack(raw)
    return mov, raw.dtype, str(getattr(raw, "shape", None))


def write_multipage_tiff(path: str, movie: np.ndarray):
    movie = np.asarray(movie)
    if movie.ndim != 3:
        raise ValueError(f"Expected (T,Y,X), got {movie.shape}")

    ensure_dir(os.path.dirname(path))

    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        for t in range(movie.shape[0]):
            tw.write(movie[t], photometric="minisblack")


def main():
    print("CWD:", os.getcwd(), flush=True)
    print("INPUT_DIR abs:", os.path.abspath(INPUT_DIR), flush=True)

    ensure_dir(OUTPUT_DIR)
    tmp_root = os.path.join(OUTPUT_DIR, "_tmp_for_caiman")
    ensure_dir(tmp_root)

    files = find_inputs_recursive(INPUT_DIR)
    print(f"Found {len(files)} input files", flush=True)

    rows = []

    for i, in_path in enumerate(files, 1):
        rel = os.path.relpath(in_path, start=os.path.abspath(INPUT_DIR))
        out_tif = rel_out_path(in_path, OUTPUT_DIR, INPUT_DIR, suffix="_rigid.tif")

        if SKIP_EXISTING and os.path.exists(out_tif):
            print(f"[{i}/{len(files)}] SKIP (exists): {rel}", flush=True)
            continue

        print(f"\n[{i}/{len(files)}] START {rel}", flush=True)
        t0 = time.time()

        try:
            movie, dtype_in, raw_shape_str = read_movie_any(in_path, z_strategy=Z_STRATEGY)
            if movie.ndim != 3:
                raise ValueError(f"Movie is not (T,Y,X). Got shape={movie.shape}")

            if movie.shape[0] < 2:
                frame = movie[0]
                tifffile.imwrite(out_tif, frame.astype(dtype_in, copy=False), photometric="minisblack")
                dt = time.time() - t0
                print(f"  NOTE: T=1 frame -> saved without motion correction: {out_tif}", flush=True)
                rows.append({
                    "rel_file": rel,
                    "in_file": in_path,
                    "out_tif": out_tif,
                    "dtype": str(dtype_in),
                    "raw_shape": raw_shape_str,
                    "note": "T=1, skipped motion correction",
                    "seconds": dt
                })
                continue

            mean_raw = movie.mean(axis=0)
            sharp_raw = mean_sharpness(mean_raw)

            ext = os.path.splitext(in_path)[1].lower()
            if ext == ".czi":
                tmp_tif = rel_out_path(in_path, tmp_root, INPUT_DIR, suffix=".tif")
                write_multipage_tiff(tmp_tif, movie)
                mc_input = tmp_tif
            else:
                mc_input = in_path

            params_dict = {
                "data": {"fnames": [mc_input], "fr": FR},
                "motion": {
                    "pw_rigid": False,
                    "max_shifts": MAX_SHIFTS,
                    "shifts_opencv": True,
                    "border_nan": BORDER_NAN
                }
            }
            opts = caiman_params.CNMFParams(params_dict=params_dict)

            mc = MotionCorrect(
                [mc_input], dview=None,
                max_shifts=opts.get("motion", "max_shifts"),
                pw_rigid=opts.get("motion", "pw_rigid"),
                shifts_opencv=opts.get("motion", "shifts_opencv"),
                border_nan=opts.get("motion", "border_nan"),
            )
            mc.motion_correct(save_movie=True)

            mmap_file = find_mmap_path(mc)
            corr = load_mmap_movie(mmap_file)

            mean_corr = corr.mean(axis=0)
            sharp_corr = mean_sharpness(mean_corr)

            if dtype_in == np.uint8:
                corr_out = np.clip(corr, 0, 255).astype(np.uint8)
            elif dtype_in == np.uint16:
                corr_out = np.clip(corr, 0, 65535).astype(np.uint16)
            else:
                corr_out = corr.astype(np.float32)

            tifffile.imwrite(out_tif, corr_out, photometric="minisblack")

            caiman_max = np.nan
            caiman_mean = np.nan
            if hasattr(mc, "shifts_rig"):
                sh = np.array(mc.shifts_rig, dtype=np.float32)
                if sh.ndim == 2 and sh.shape[1] >= 2:
                    mags = np.sqrt((sh[:, :2] ** 2).sum(axis=1))
                    caiman_max = float(mags.max())
                    caiman_mean = float(mags.mean())

            dt = time.time() - t0
            print(f"  saved: {out_tif}", flush=True)
            print(f"  raw_shape={raw_shape_str} -> corr_shape={corr_out.shape}", flush=True)
            print(
                f"  sharp raw={sharp_raw:.3f} corr={sharp_corr:.3f} "
                f"(+{(sharp_corr/(sharp_raw+1e-9)-1)*100:.2f}%)",
                flush=True
            )
            print(f"  shifts mean={caiman_mean:.2f}px max={caiman_max:.2f}px", flush=True)
            print(f"  time: {dt:.1f}s", flush=True)

            rows.append({
                "rel_file": rel,
                "in_file": in_path,
                "out_tif": out_tif,
                "mmap_file": mmap_file,
                "used_for_mc": mc_input,
                "dtype": str(dtype_in),
                "raw_shape": raw_shape_str,
                "sharp_raw": sharp_raw,
                "sharp_corr": sharp_corr,
                "sharp_gain_percent": (sharp_corr / (sharp_raw + 1e-9) - 1.0) * 100.0,
                "caiman_mean_shift_px": caiman_mean,
                "caiman_max_shift_px": caiman_max,
                "seconds": dt
            })

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            rows.append({"rel_file": rel, "in_file": in_path, "error": str(e)})

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT_DIR, "motion_correction_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved summary CSV: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
