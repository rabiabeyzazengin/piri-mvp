import os
import cv2
import numpy as np

def order_points(pts: np.ndarray) -> np.ndarray:
    """Return points ordered as: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def four_point_warp(image: np.ndarray, pts4: np.ndarray) -> np.ndarray:
    rect = order_points(pts4)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    maxW = max(maxW, 2)
    maxH = max(maxH, 2)

    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH), flags=cv2.INTER_CUBIC)
    return warped


def rect_score(cnt4: np.ndarray, img_w: int, img_h: int) -> float:
    """Score a 4-point contour as 'paper-like'."""
    pts = cnt4.reshape(4, 2).astype(np.float32)
    area = cv2.contourArea(pts)
    if area <= 0:
        return -1.0

    img_area = float(img_w * img_h)
    area_ratio = area / img_area
    if area_ratio < 0.08 or area_ratio > 0.97:
        return -1.0

    border_margin = 0.01 
    xm = border_margin * img_w
    ym = border_margin * img_h
    touches = 0
    for (x, y) in pts:
        if x < xm or x > (img_w - xm) or y < ym or y > (img_h - ym):
            touches += 1
    border_penalty = 1.0 - (touches * 0.12) 

    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    w1 = np.linalg.norm(tr - tl)
    w2 = np.linalg.norm(br - bl)
    h1 = np.linalg.norm(bl - tl)
    h2 = np.linalg.norm(br - tr)
    w = (w1 + w2) / 2.0
    h = (h1 + h2) / 2.0
    if w < 2 or h < 2:
        return -1.0

    ar = w / h
    ar = ar if ar >= 1.0 else (1.0 / ar)  
    ar_target = 1.414
    ar_score = np.exp(-abs(ar - ar_target) * 1.6)

    x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
    rect_area = float(bw * bh) + 1e-6
    rectangularity = float(area) / rect_area
    rectangularity = np.clip(rectangularity, 0.0, 1.0)

    score = (
        (area_ratio ** 0.85)
        * (0.55 + 0.45 * rectangularity)
        * (0.55 + 0.45 * ar_score)
        * border_penalty
    )
    return float(score)


def best_paper_contour(image_bgr: np.ndarray) -> np.ndarray | None:
    """Try multiple preprocess pipelines and pick the best 4-point contour."""
    h, w = image_bgr.shape[:2]

    max_dim = 1400
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        small = cv2.resize(
            image_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image_bgr.copy()

    sh, sw = small.shape[:2]
    pipelines = []

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    edges = cv2.erode(edges, np.ones((3, 3), np.uint8), iterations=1)
    pipelines.append(edges)

    # Pipeline 2: stronger contrast + edges
    gray2 = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray2 = clahe.apply(gray2)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    edges2 = cv2.Canny(gray2, 30, 120)
    edges2 = cv2.dilate(edges2, np.ones((3, 3), np.uint8), iterations=2)
    pipelines.append(edges2)

    # Pipeline 3: adaptive threshold
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 6
    )
    thr = cv2.morphologyEx(
        thr, cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
        iterations=2
    )
    pipelines.append(thr)

    best = None
    best_score = -1.0

    for bin_img in pipelines:
        cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

        for c in cnts:
            peri = cv2.arcLength(c, True)
            if peri < 100:
                continue

            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) != 4:
                continue
            if not cv2.isContourConvex(approx):
                continue

            sc = rect_score(approx, sw, sh)
            if sc > best_score:
                best_score = sc
                best = approx

    if best is None:
        return None

    if scale != 1.0:
        best = (best.reshape(4, 2).astype(np.float32) / scale).astype(np.float32).reshape(4, 1, 2)

    return best

def force_portrait(img_bgr: np.ndarray) -> np.ndarray:
    """Make output always portrait (h >= w)."""
    h, w = img_bgr.shape[:2]
    if w > h:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    return img_bgr


def binarize_for_orientation(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 15
    )
    inv = 255 - th
    return inv


def find_blue_logo_centroid(img_bgr: np.ndarray):
    """Detect blue logo-ish region; return centroid if found."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 40, 40])
    upper = np.array([135, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 200:
        return None

    M = cv2.moments(c)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def choose_upright(img0: np.ndarray, img180: np.ndarray) -> np.ndarray:
    """
    Decide between 0 and 180 degrees:
    - Prefer direction where blue logo is closer to top-left
    - Fallback: text mass (COM) higher in the image
    """
    c0 = find_blue_logo_centroid(img0)
    c1 = find_blue_logo_centroid(img180)

    if c0 is not None or c1 is not None:
        def dist_tl(c, shape):
            if c is None:
                return 1e18
            x, y = c
            h, w = shape[:2]
            return (x / w) ** 2 + (y / h) ** 2

        d0 = dist_tl(c0, img0.shape)
        d1 = dist_tl(c1, img180.shape)
        return img0 if d0 <= d1 else img180

    def com_y(img):
        inv = binarize_for_orientation(img)
        ys, xs = np.nonzero(inv)
        if len(ys) == 0:
            return 0.5
        return float(ys.mean() / inv.shape[0])

    return img0 if com_y(img0) <= com_y(img180) else img180


def fix_180_if_needed(img_bgr: np.ndarray) -> np.ndarray:
    """Assumes image is already portrait; fix upside-down (0 vs 180)."""
    img0 = img_bgr
    img180 = cv2.rotate(img_bgr, cv2.ROTATE_180)
    return choose_upright(img0, img180)


def postprocess_scan(warped: np.ndarray) -> np.ndarray:
    """Light cleanup; keep color output but reduce gray background."""
    h, w = warped.shape[:2]

    pad = 6
    if h > 2 * pad and w > 2 * pad:
        warped = warped[pad:h - pad, pad:w - pad]

    warped = force_portrait(warped)
    warped = fix_180_if_needed(warped)
    return warped


def fix_orientation_when_no_paper(img_bgr: np.ndarray) -> np.ndarray:
    """
    If paper not found, still produce readable portrait output:
    - Force portrait
    - Fix 180 if needed
    """
    img_bgr = force_portrait(img_bgr)
    img_bgr = fix_180_if_needed(img_bgr)
    return img_bgr


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    INPUT_DIR = os.path.join(BASE_DIR, "input_images")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output_images")

    print(f"📌 INPUT  : {INPUT_DIR}")
    print(f"📌 OUTPUT : {OUTPUT_DIR}")

    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(
            f"❌ '{INPUT_DIR}' klasörü yok. Script ile aynı dizinde 'input_images' olmalı."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".webp")
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(exts)]

    if not files:
        print("❌ İşlenecek resim yok (input_images boş).")
        return

    ok = 0
    fail = 0

    for fname in files:
        in_path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(in_path)
        if img is None:
            print(f"⚠️ Okunamadı: {fname}")
            fail += 1
            continue

        contour4 = best_paper_contour(img)

        if contour4 is None:
            print(f"⚠️ Kağıt bulunamadı: {fname}  (portrait+180 düzeltme uygulanıp kaydedildi)")
            out = fix_orientation_when_no_paper(img)
            fail += 1
        else:
            out = four_point_warp(img, contour4)

            if out.shape[0] < 250 or out.shape[1] < 250:
                print(f"⚠️ Kağıt bulunamadı (warp çok küçük): {fname}  (portrait+180 düzeltme uygulanıp kaydedildi)")
                out = fix_orientation_when_no_paper(img)
                fail += 1
            else:
                out = postprocess_scan(out)
                ok += 1

        name, _ = os.path.splitext(fname)
        out_path = os.path.join(OUTPUT_DIR, f"{name}_scan.jpg")
        cv2.imwrite(out_path, out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    print(f"🎉 Bitti. Başarılı: {ok}, Başarısız: {fail}")
    print("✅ Tüm çıktılar output_images klasörüne yazıldı.")


if __name__ == "__main__":
    main()
