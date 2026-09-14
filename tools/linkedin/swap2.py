import cv2
import numpy as np
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from faceswap import make_detector, landmarks, delaunay_indices, warp_triangle, FACE_OVAL

INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_BROW = [70, 63, 105, 66, 107, 46, 53, 52, 65, 55]
RIGHT_BROW = [300, 293, 334, 296, 336, 276, 283, 282, 295, 285]


def color_transfer(src, dst, mask, chroma=0.35, luma=0.85):
    """Fit the source face to the target's lighting without repainting his
    complexion: luminance is matched to the scene, chroma only partially so
    his own skin tone survives."""
    m = mask > 10
    if m.sum() < 50:
        return src
    s = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = s.copy()
    for c in range(3):
        w = luma if c == 0 else chroma
        sm, ss = s[:, :, c][m].mean(), s[:, :, c][m].std() + 1e-6
        dm, ds = d[:, :, c][m].mean(), d[:, :, c][m].std() + 1e-6
        full = (s[:, :, c] - sm) * (min(ds / ss, 2.0)) + dm
        out[:, :, c] = s[:, :, c] * (1 - w) + full * w
    # clamp blown highlights/crushed shadows to the target's own tonal range,
    # otherwise flash-lit selfie specular hot spots punch through as white halos
    lo, hi = np.percentile(d[:, :, 0][m], [1.5, 98.5])
    out[:, :, 0] = np.clip(out[:, :, 0], lo, hi)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def deshine(img, mask, strength=0.85):
    """Flatten specular hot spots (flash-lit oily skin) inside mask."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    k = max(11, (int(min(img.shape[:2]) * 0.035) // 2) * 2 + 1)
    base = cv2.medianBlur(L.astype(np.uint8), k).astype(np.float32)
    excess = np.clip(L - base - 4.0, 0, None)
    m = (mask > 10).astype(np.float32)
    m = cv2.GaussianBlur(m, (21, 21), 0)
    lab[:, :, 0] = L - excess * strength * m
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def lower_forehead(pts, ld, keep=0.55):
    """Pull the top of the face hull down off the hairline so the blend seam
    lands on flat forehead skin instead of at the hair boundary."""
    brow_y = float(np.mean(ld[LEFT_BROW + RIGHT_BROW][:, 1]))
    out = pts.copy()
    above = out[:, 1] < brow_y
    out[above, 1] = brow_y - (brow_y - out[above, 1]) * keep
    return out


def poly_mask(shape, pts, dilate=0):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.int32(pts)), 255)
    if dilate:
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    return m


def glasses_layer(dst, ld):
    """Isolate the dark wire frames of the target's eyewear (frames only:
    not the eyes, not the eyebrows, not hair)."""
    face_w = np.linalg.norm(ld[234] - ld[454])
    d = max(3, int(face_w * 0.11))
    band = poly_mask(dst.shape, ld[LEFT_EYE + RIGHT_EYE], dilate=d)
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    skin = np.median(gray[band > 0])
    dark = ((gray.astype(np.float32) < skin * 0.62) & (band > 0)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    keep = np.zeros_like(dark)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        fill = a / float(max(bw * bh, 1))
        # frames are long and wiry; hair/brows come out as solid blobs
        if a > 40 and fill < 0.45:
            keep[lab == i] = 255

    block = poly_mask(dst.shape, ld[LEFT_EYE], dilate=5) | poly_mask(dst.shape, ld[RIGHT_EYE], dilate=5)
    block |= poly_mask(dst.shape, ld[LEFT_BROW], dilate=7) | poly_mask(dst.shape, ld[RIGHT_BROW], dilate=7)
    keep[block > 0] = 0
    return cv2.GaussianBlur(keep, (3, 3), 0)


def run(src_path, dst_path, out_path, keep_mouth=True, keep_glasses=False, remove_glasses=False,
        erode=10, feather=25, forehead=0.55, debug=False):
    src = cv2.imread(src_path)
    dst = cv2.imread(dst_path)
    det = make_detector()
    ls, ld = landmarks(src, det), landmarks(dst, det)
    if ls is None or ld is None:
        raise SystemExit("landmark detection failed")

    # 1. geometry: warp source face onto target landmark layout
    tris = delaunay_indices(ld, dst.shape[:2])
    canvas = np.zeros_like(dst, np.float32)
    for a, b, c in tris:
        warp_triangle(src.astype(np.float32), canvas, [ls[a], ls[b], ls[c]], [ld[a], ld[b], ld[c]])
    warped = np.clip(canvas, 0, 255).astype(np.uint8)

    # 2. mask: face oval (top pulled off the hairline), eroded + feathered
    oval = lower_forehead(ld[FACE_OVAL].copy(), ld, keep=forehead)
    mask = poly_mask(dst.shape, oval)
    mask = cv2.erode(mask, np.ones((erode, erode), np.uint8))

    if keep_mouth:
        mouth = poly_mask(dst.shape, ld[INNER_LIPS], dilate=3)
        mask[mouth > 0] = 0

    # 3. kill flash highlights, then colour match to the target's own face
    warped = deshine(warped, mask)
    warped = color_transfer(warped, dst, mask)

    # 4. feathered alpha composite
    k = feather if feather % 2 else feather + 1
    alpha = (cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0)[:, :, None]
    out = (warped.astype(np.float32) * alpha + dst.astype(np.float32) * (1 - alpha))
    out = np.clip(out, 0, 255).astype(np.uint8)

    # 5. put the eyewear back on top
    if keep_glasses:
        g = glasses_layer(dst, ld).astype(np.float32)[:, :, None] / 255.0
        out = np.clip(out.astype(np.float32) * (1 - g) + dst.astype(np.float32) * g, 0, 255).astype(np.uint8)
    elif remove_glasses:
        # frame arms sit outside the swapped face region — paint them out
        face_w = np.linalg.norm(ld[234] - ld[454])
        band = poly_mask(dst.shape, ld[LEFT_EYE + RIGHT_EYE], dilate=int(face_w * 0.45))
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        skin = np.median(gray[poly_mask(dst.shape, oval) > 0])
        leftover = ((gray.astype(np.float32) < skin * 0.62) & (band > 0) &
                    (alpha[:, :, 0] < 0.6)).astype(np.uint8) * 255
        hair = poly_mask(dst.shape, oval)
        leftover[hair > 0] = 0
        leftover = cv2.dilate(leftover, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(leftover, 8)
        clean = np.zeros_like(leftover)
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if a > 25 and a / float(max(bw * bh, 1)) < 0.5:
                clean[lab == i] = 255
        clean = cv2.dilate(clean, np.ones((3, 3), np.uint8))
        out = cv2.inpaint(out, clean, 6, cv2.INPAINT_TELEA)

    cv2.imwrite(out_path, out)
    if debug:
        cv2.imwrite(out_path.replace(".png", "_mask.png"), mask)
    print("wrote", out_path, "tris", len(tris))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("out")
    ap.add_argument("--mouth", action="store_true")
    ap.add_argument("--glasses", action="store_true")
    ap.add_argument("--noglasses", action="store_true")
    ap.add_argument("--erode", type=int, default=10)
    ap.add_argument("--feather", type=int, default=25)
    ap.add_argument("--forehead", type=float, default=0.55)
    a = ap.parse_args()
    run(a.src, a.dst, a.out, keep_mouth=a.mouth, keep_glasses=a.glasses, remove_glasses=a.noglasses,
        erode=a.erode, feather=a.feather, forehead=a.forehead)
