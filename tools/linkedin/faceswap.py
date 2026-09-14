import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL = __import__("os").environ.get("FACE_LANDMARKER", "face_landmarker.task")

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
             400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
             54, 103, 67, 109]


def make_detector():
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL),
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def landmarks(img, detector):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        return None
    h, w = img.shape[:2]
    return np.array([[lm.x * w, lm.y * h] for lm in res.face_landmarks[0]], dtype=np.float32)


def delaunay_indices(points, size):
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect)
    pts = [(float(p[0]), float(p[1])) for p in points]
    inserted = []
    for i, p in enumerate(pts):
        if 0 <= p[0] < size[1] and 0 <= p[1] < size[0]:
            subdiv.insert(p)
            inserted.append(i)
    P = np.array(pts, dtype=np.float64)
    tris = []
    for t in subdiv.getTriangleList():
        idx = []
        for k in range(3):
            v = np.array([t[k * 2], t[k * 2 + 1]], dtype=np.float64)
            d = np.hypot(P[:, 0] - v[0], P[:, 1] - v[1])
            j = int(np.argmin(d))
            if d[j] < 1.5:
                idx.append(j)
        if len(idx) == 3 and len(set(idx)) == 3:
            tris.append(tuple(idx))
    return tris


def warp_triangle(src, dst, t_src, t_dst):
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))
    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return
    t1 = [(t_src[i][0] - r1[0], t_src[i][1] - r1[1]) for i in range(3)]
    t2 = [(t_dst[i][0] - r2[0], t_dst[i][1] - r2[1]) for i in range(3)]

    src_crop = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    if src_crop.size == 0:
        return
    M = cv2.getAffineTransform(np.float32(t1), np.float32(t2))
    warped = cv2.warpAffine(src_crop, M, (r2[2], r2[3]), None,
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2), (1.0, 1.0, 1.0), cv2.LINE_AA, 0)

    region = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    if region.shape[:2] != warped.shape[:2]:
        return
    dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = region * (1 - mask) + warped * mask


def swap(src_path, dst_path, out_path, erode=6, feather=9, clone_mode=cv2.NORMAL_CLONE,
         mask_points=None, debug=False):
    src = cv2.imread(src_path)
    dst = cv2.imread(dst_path)
    det = make_detector()

    ls = landmarks(src, det)
    ld = landmarks(dst, det)
    if ls is None or ld is None:
        raise SystemExit(f"landmark detection failed  src={ls is not None} dst={ld is not None}")

    tris = delaunay_indices(ld, dst.shape[:2])

    warped_face = np.zeros_like(dst, dtype=np.float32)
    dstf = warped_face
    for (a, b, c) in tris:
        warp_triangle(src.astype(np.float32), dstf,
                      [ls[a], ls[b], ls[c]], [ld[a], ld[b], ld[c]])
    warped_face = np.clip(dstf, 0, 255).astype(np.uint8)

    idxs = mask_points if mask_points is not None else FACE_OVAL
    hull = cv2.convexHull(np.int32(ld[idxs]))
    mask = np.zeros(dst.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if erode:
        mask = cv2.erode(mask, np.ones((erode, erode), np.uint8), iterations=1)
    if feather:
        k = feather if feather % 2 == 1 else feather + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    r = cv2.boundingRect(hull)
    center = (r[0] + r[2] // 2, r[1] + r[3] // 2)
    out = cv2.seamlessClone(warped_face, dst, mask, center, clone_mode)

    cv2.imwrite(out_path, out)
    if debug:
        dbg = dst.copy()
        for p in ld:
            cv2.circle(dbg, (int(p[0]), int(p[1])), 1, (0, 255, 0), -1)
        cv2.polylines(dbg, [hull], True, (0, 0, 255), 2)
        cv2.imwrite(out_path.replace('.png', '_debug.png'), dbg)
    return out_path


if __name__ == "__main__":
    import sys
    swap(sys.argv[1], sys.argv[2], sys.argv[3])
    print("done", sys.argv[3])
