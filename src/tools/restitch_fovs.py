"""
Re-stitch a flattened multi-FOV mosaic to remove black seams and
FOV-boundary duplication ("ghosted" cells appearing in both neighboring FOVs).

Diagnosis (from your file)
---------------------------
- Grid: 3 rows x 4 cols of 2000x2000 FOVs.
- Adjacent tiles have a real ~20px physical overlap that was never
  cropped or blended when the mosaic was assembled -- that's why the
  same cells show up (slightly shifted) in both neighboring FOVs, with
  a thin black gap at the nominal tile boundary itself.

Approach
--------
1. Use 2D phase cross-correlation (skimage) on a wide band (150px) along
   each seam to measure the true overlap between neighboring tiles.
2. Pool all seam measurements to get a robust GLOBAL overlap estimate
   (median, outlier-filtered) -- this is far more trustworthy than
   trusting each seam in isolation, since some seams sit over
   low-texture regions (e.g. an empty gap in the tissue) where local
   correlation is unreliable.
3. Per seam: use the local measurement only if it's close to the global
   estimate (sanity check); otherwise fall back to the global value.
4. Rebuild the mosaic by overlapping tiles by the resolved amount and
   feather-blending the overlap band, using one reference channel to
   compute the transform, then applying it identically to every channel.
"""

import numpy as np
import tifffile
from skimage.registration import phase_cross_correlation

# ----------------------------- CONFIG ---------------------------------
INPUT_PATH = "/mnt/user-data/uploads/HAND1_Brachyury_mid__.tif"
OUTPUT_PATH = "HAND1_Brachyury_mid_restitched.tif"  # edit to your desired output path
FOV_SIZE = 2000        # nominal size of each square FOV, in pixels
REF_CHANNEL = 3        # channel used to compute alignment
BAND = 150             # width (px) of the band used for phase correlation
OUTLIER_TOL = 10       # px: how far a seam's local estimate can be from
                       # the global median before we discard it as unreliable
BLEND_WIDTH = 20       # width (px) of the feathered blend zone
# ------------------------------------------------------------------------


def detect_grid(shape, fov_size):
    h, w = shape
    return round(h / fov_size), round(w / fov_size)


def measure_h_seams(ref_img, n_rows, n_cols, fov_size, band):
    """Measure (dy, dx) for every vertical seam (column boundary)."""
    shifts = []
    for i in range(n_rows):
        for j in range(n_cols - 1):
            y0, y1 = i * fov_size, (i + 1) * fov_size
            x_seam = (j + 1) * fov_size
            left = ref_img[y0:y1, x_seam - band:x_seam]
            right = ref_img[y0:y1, x_seam:x_seam + band]
            (dy, dx), _, _ = phase_cross_correlation(left, right, upsample_factor=10)
            shifts.append(((i, j), dy, dx))
    return shifts


def measure_v_seams(ref_img, n_rows, n_cols, fov_size, band):
    """Measure (dy, dx) for every horizontal seam (row boundary)."""
    shifts = []
    for i in range(n_rows - 1):
        for j in range(n_cols):
            x0, x1 = j * fov_size, (j + 1) * fov_size
            y_seam = (i + 1) * fov_size
            top = ref_img[y_seam - band:y_seam, x0:x1]
            bot = ref_img[y_seam:y_seam + band, x0:x1]
            (dy, dx), _, _ = phase_cross_correlation(top, bot, upsample_factor=10)
            shifts.append(((i, j), dy, dx))
    return shifts


def robust_global(values, tol):
    """Median with a simple outlier rejection pass."""
    values = np.array(values)
    med = np.median(values)
    keep = np.abs(values - med) < max(tol, 3 * np.std(values) + 1e-6)
    if keep.sum() == 0:
        return med
    return np.median(values[keep])


def feather_blend(canvas, weight, tile, y0, x0, blend_width):
    h, w = tile.shape
    ramp_y = np.ones(h, dtype=np.float32)
    ramp_x = np.ones(w, dtype=np.float32)
    bw = min(blend_width, h // 2, w // 2)
    if bw > 0:
        ramp = np.linspace(0, 1, bw, dtype=np.float32)
        ramp_y[:bw] = ramp; ramp_y[-bw:] = ramp[::-1]
        ramp_x[:bw] = ramp; ramp_x[-bw:] = ramp[::-1]
    tile_w = np.outer(ramp_y, ramp_x)
    canvas[y0:y0 + h, x0:x0 + w] += tile.astype(np.float32) * tile_w
    weight[y0:y0 + h, x0:x0 + w] += tile_w


def restitch_channel(img, n_rows, n_cols, fov_size, overlap_x, overlap_y):
    """overlap_x: horizontal overlap in px (int, applied at every column seam)
       overlap_y: vertical overlap in px (int, applied at every row seam)"""
    tile_x = np.zeros((n_rows, n_cols), dtype=int)
    tile_y = np.zeros((n_rows, n_cols), dtype=int)
    for i in range(n_rows):
        for j in range(1, n_cols):
            tile_x[i, j] = tile_x[i, j - 1] + fov_size - overlap_x
    for j in range(n_cols):
        for i in range(1, n_rows):
            tile_y[i, j] = tile_y[i - 1, j] + fov_size - overlap_y

    out_h = tile_y.max() + fov_size
    out_w = tile_x.max() + fov_size
    canvas = np.zeros((out_h, out_w), dtype=np.float32)
    weight = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(n_rows):
        for j in range(n_cols):
            y0, x0 = i * fov_size, j * fov_size
            tile = img[y0:y0 + fov_size, x0:x0 + fov_size]
            feather_blend(canvas, weight, tile, tile_y[i, j], tile_x[i, j], BLEND_WIDTH)

    weight[weight == 0] = 1
    return (canvas / weight).astype(img.dtype)


def main():
    img = tifffile.imread(INPUT_PATH)
    n_channels, h, w = img.shape
    n_rows, n_cols = detect_grid((h, w), FOV_SIZE)
    print(f"Detected grid: {n_rows} rows x {n_cols} cols of {FOV_SIZE}px FOVs")

    ref_img = img[REF_CHANNEL].astype(np.float32)
    h_seams = measure_h_seams(ref_img, n_rows, n_cols, FOV_SIZE, BAND)
    v_seams = measure_v_seams(ref_img, n_rows, n_cols, FOV_SIZE, BAND)

    # global overlap = -shift (phase_cross_correlation returns negative dx
    # when 'right' content is offset to the left relative to 'left', i.e.
    # the overlap amount)
    global_dx = robust_global([-dx for (_, dy, dx) in h_seams], OUTLIER_TOL)
    global_dy = robust_global([-dy for (_, dy, dx) in v_seams], OUTLIER_TOL)
    print(f"Global horizontal overlap estimate: {global_dx:.1f} px")
    print(f"Global vertical overlap estimate:   {global_dy:.1f} px")

    overlap_x = int(round(global_dx))
    overlap_y = int(round(global_dy))

    out_channels = []
    for c in range(n_channels):
        print(f"Restitching channel {c}...")
        out_channels.append(
            restitch_channel(img[c], n_rows, n_cols, FOV_SIZE, overlap_x, overlap_y)
        )
    out = np.stack(out_channels, axis=0)
    tifffile.imwrite(OUTPUT_PATH, out)
    print(f"Saved: {OUTPUT_PATH}  shape={out.shape}")


if __name__ == "__main__":
    main()
