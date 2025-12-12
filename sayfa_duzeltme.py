import os
import glob
import cv2
import numpy as np

try:
    import pytesseract
    HAS_TESS = True
except Exception:
    HAS_TESS = False


# =========================
# AYARLAR
# =========================
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"
DEBUG_DIR = os.path.join(OUTPUT_DIR, "_debug")

SAVE_DEBUG = True

LANG = "tur+eng"
LANG_FALLBACK = "eng"

MAX_DETECT_SIDE = 1600          # sayfa tespiti için küçültme
WARP_MIN_SIZE = 900             # warptan sonra min boyut
PAD_QUAD_SCALE = 1.04           # sayfa köşelerine çok az genişletme
MIN_PAGE_AREA_RATIO = 0.18      # kağıt alanı / görüntü alanı

# Orientation
OSD_LOCK_CONF = 12.0            # OSD conf yüksekse rotasyonu kilitle
TRY_FINE_DESKEW = True
MAX_SKEW_DEG = 18

# Tight crop (masayı at)
TIGHT_CROP_PAD = 12


# =========================
# IO
# =========================
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_DEBUG:
        os.makedirs(DEBUG_DIR, exist_ok=True)

def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite_unicode(path: str, img):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    elif ext == ".png":
        ok, buf = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    else:
        ok, buf = cv2.imencode(".png", img)
    if not ok:
        return False
    buf.tofile(path)
    return True


# =========================
# GEOMETRİ / WARP
# =========================
def resize_max_side(img, max_side=MAX_DETECT_SIDE):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img, 1.0
    s = max_side / float(m)
    out = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return out, s

def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def quad_area(q):
    x = q[:, 0]
    y = q[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def expand_quad(quad, scale=PAD_QUAD_SCALE):
    q = order_points(quad)
    c = np.mean(q, axis=0)
    return c + (q - c) * scale

def clip_quad(quad, w, h):
    q = quad.copy()
    q[:, 0] = np.clip(q[:, 0], 0, w-1)
    q[:, 1] = np.clip(q[:, 1], 0, h-1)
    return q

def four_point_warp(img, quad):
    q = order_points(quad)
    (tl, tr, br, bl) = q

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    maxW = max(maxW, WARP_MIN_SIZE)
    maxH = max(maxH, WARP_MIN_SIZE)

    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, M, (maxW, maxH), flags=cv2.INTER_CUBIC)


# =========================
# SAYFA TESPİTİ (SAĞLAM)
# =========================
def build_paper_mask(img_bgr):
    """
    Kağıt genelde: düşük saturation + yüksek value.
    Bu maske masayı/objeleri eleyip kağıdı yakalar.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # daha agresif white-ish mask
    mask = cv2.inRange(hsv, (0, 0, 140), (180, 90, 255))

    # ışık değişimleri için: value yüksek alanları güçlendir
    vmask = cv2.inRange(v, 160, 255)
    mask = cv2.bitwise_and(mask, vmask)

    kernel = np.ones((9,9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask

def find_page_quad(img_small):
    h, w = img_small.shape[:2]
    mask = build_paper_mask(img_small)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    best = None

    for c in cnts[:8]:
        area = cv2.contourArea(c)
        if area < 2000:
            continue

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect).astype(np.float32)
        q = order_points(box)
        ar = quad_area(q) / float(w*h + 1e-9)

        if ar < MIN_PAGE_AREA_RATIO:
            continue

        # approxPolyDP dene: 4 köşe daha iyi olur
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            q2 = order_points(approx.reshape(4,2).astype(np.float32))
            ar2 = quad_area(q2) / float(w*h + 1e-9)
            if ar2 >= ar:
                q = q2
                ar = ar2

        best = q
        break

    return best, mask


# =========================
# ROTASYON / DESKEW
# =========================
def rotate_90(img, k):
    k %= 4
    if k == 0: return img
    if k == 1: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if k == 2: return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

def rotate_any(img, angle_deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def prep_for_text(img_bgr, upscale=2):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if upscale > 1:
        g = cv2.resize(g, (g.shape[1]*upscale, g.shape[0]*upscale), interpolation=cv2.INTER_CUBIC)

    g = cv2.GaussianBlur(g, (3,3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8))
    g = clahe.apply(g)

    thr = cv2.adaptiveThreshold(
        g, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 11
    )
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    return g, thr

def estimate_skew(img_bgr, max_deg=MAX_SKEW_DEG):
    _, thr = prep_for_text(img_bgr, upscale=1)
    lines = cv2.HoughLinesP(
        thr, 1, np.pi/180, threshold=120,
        minLineLength=int(min(img_bgr.shape[:2]) * 0.28),
        maxLineGap=20
    )
    if lines is None:
        return 0.0

    angles = []
    for x1,y1,x2,y2 in lines[:,0]:
        dx = x2-x1
        dy = y2-y1
        if abs(dx) < 10:
            continue
        ang = np.degrees(np.arctan2(dy, dx))
        if -max_deg <= ang <= max_deg:
            angles.append(ang)

    if not angles:
        return 0.0
    return float(np.median(angles))

def horizontalness_score(img_bgr):
    """
    OCR yoksa bile çalışan sinyal:
    Metin yataysa satır projeksiyonu daha “dalgalı” olur.
    """
    _, thr = prep_for_text(img_bgr, upscale=1)
    bw = (thr > 0).astype(np.uint8)
    row = bw.sum(axis=1).astype(np.float32)
    col = bw.sum(axis=0).astype(np.float32)
    return float(np.log1p(np.std(row)) - np.log1p(np.std(col)))

def ocr_char_score(img_bgr, lang):
    """
    Kelime az olsa bile: karakter ve conf istatistiğinden skor üret.
    """
    if not HAS_TESS:
        return 0.0, 0, 0.0

    g, thr = prep_for_text(img_bgr, upscale=2)

    cfg = "--psm 6"  # block of text
    data = pytesseract.image_to_data(thr, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)

    confs = []
    char_count = 0
    good_char = 0

    texts = data.get("text", [])
    conf_list = data.get("conf", [])

    for t, c in zip(texts, conf_list):
        t = (t or "").strip()
        if not t:
            continue
        try:
            cf = float(c)
        except:
            continue
        if cf < 0:
            continue

        # karakter bazlı say
        char_count += len(t)
        # harf/rakam oranı
        good_char += sum(ch.isalnum() for ch in t)
        confs.append(cf)

    mean_conf = float(np.mean(confs)) if confs else 0.0
    # karakter & good_char ile ölçekle (kelime yoksa bile ayırt eder)
    score = (mean_conf / 100.0) * np.log1p(char_count) * (0.4 + 0.6 * (good_char / max(char_count, 1)))
    return float(score), int(char_count), mean_conf

def osd_rotate_and_conf(img_bgr):
    """
    Tesseract OSD (model) — doğru çalışması için pre-processing + upscale
    """
    if not HAS_TESS:
        return None, 0.0
    try:
        _, thr = prep_for_text(img_bgr, upscale=2)
        osd = pytesseract.image_to_osd(thr, config="--psm 0")
        rot = None
        conf = 0.0
        for line in osd.splitlines():
            if "Rotate:" in line:
                rot = int(line.split(":")[1].strip())
            if "Orientation confidence:" in line:
                conf = float(line.split(":")[1].strip())
        return rot, conf
    except Exception:
        return None, 0.0

def choose_best_orientation(img_bgr):
    """
    1) OSD güvenliyse kilitle (düzgünü bozma)
    2) değilse brute force 0/90/180/270 + (fine deskew) + skor = OSD + OCR + yataylık
    """
    rot_osd, conf_osd = osd_rotate_and_conf(img_bgr)

    # 1) OSD kilit
    if rot_osd in [0,90,180,270] and conf_osd >= OSD_LOCK_CONF:
        out = rotate_90(img_bgr, (rot_osd//90) % 4)
        skew = estimate_skew(out) if TRY_FINE_DESKEW else 0.0
        out2 = rotate_any(out, -skew) if abs(skew) > 0.2 else out
        return out2, {
            "mode": "osd_lock",
            "osd_rot": rot_osd,
            "osd_conf": conf_osd,
            "skew": skew
        }

    # 2) brute force
    candidates = [0,90,180,270]
    # OSD varsa “öncelik” ver ama kilitleme yok
    if rot_osd in candidates:
        candidates = [rot_osd] + [d for d in candidates if d != rot_osd]

    best = None
    best_meta = None

    for deg in candidates:
        cand = rotate_90(img_bgr, (deg//90) % 4)

        skew = estimate_skew(cand) if TRY_FINE_DESKEW else 0.0
        if abs(skew) > 0.2:
            cand = rotate_any(cand, -skew)

        hscore = horizontalness_score(cand)

        # OCR iki dil dene
        o1, ch1, mc1 = (0.0, 0, 0.0)
        o2, ch2, mc2 = (0.0, 0, 0.0)
        if HAS_TESS:
            try:
                o1, ch1, mc1 = ocr_char_score(cand, LANG)
            except:
                pass
            if o1 <= 1e-6:
                try:
                    o2, ch2, mc2 = ocr_char_score(cand, LANG_FALLBACK)
                except:
                    pass

        ocr_best = max(o1, o2)
        char_best = ch1 if o1 >= o2 else ch2
        mean_conf = mc1 if o1 >= o2 else mc2

        # OSD destek puanı: OSD varsa ve bu deg’i öneriyorsa ek bonus
        osd_bonus = 0.0
        if rot_osd in [0,90,180,270]:
            if deg == rot_osd:
                osd_bonus = 0.15 + min(conf_osd, 20.0) / 100.0

        # Final skor:
        # - OCR güçlü ise daha fazla ağırlık
        # - OCR zayıf ise (char az) yataylık ağırlığı artar
        if char_best >= 20:
            score = (ocr_best * 1.1) + (hscore * 0.35) + osd_bonus
        else:
            score = (ocr_best * 0.35) + (hscore * 1.1) + osd_bonus

        # sayfa genelde portre -> küçük bonus
        hh, ww = cand.shape[:2]
        if hh >= ww:
            score += 0.08

        meta = {
            "mode": "search",
            "deg": deg,
            "skew": skew,
            "osd_rot": rot_osd,
            "osd_conf": conf_osd,
            "hscore": hscore,
            "ocr": ocr_best,
            "chars": char_best,
            "mean_conf": mean_conf,
            "score": score
        }

        if best is None or score > best_meta["score"]:
            best = cand
            best_meta = meta

    return best, best_meta


# =========================
# TIGHT CROP (warp sonrası)
# =========================
def tight_crop_to_page(img_bgr, pad=TIGHT_CROP_PAD):
    mask = build_paper_mask(img_bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img_bgr, mask

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    x2 = max(0, x-pad)
    y2 = max(0, y-pad)
    x3 = min(img_bgr.shape[1], x+w+pad)
    y3 = min(img_bgr.shape[0], y+h+pad)

    return img_bgr[y2:y3, x2:x3].copy(), mask


# =========================
# PIPELINE
# =========================
def process_one(path):
    img0 = imread_unicode(path)
    if img0 is None:
        print(f"❌ Okunamadı: {path}")
        return

    small, s = resize_max_side(img0, MAX_DETECT_SIDE)
    quad_small, mask0 = find_page_quad(small)

    base = os.path.splitext(os.path.basename(path))[0]

    if quad_small is None:
        # sayfa bulunamazsa: hiç kırpma yok, sadece rotasyon dene
        warped = img0.copy()
        if SAVE_DEBUG:
            imwrite_unicode(os.path.join(DEBUG_DIR, f"{base}_mask0.jpg"),
                            cv2.cvtColor(mask0, cv2.COLOR_GRAY2BGR))
        quad_info = "page=none"
    else:
        quad_orig = quad_small / float(s)
        h0, w0 = img0.shape[:2]
        q = expand_quad(quad_orig, PAD_QUAD_SCALE)
        q = clip_quad(q, w0, h0)

        warped = four_point_warp(img0, q)

        quad_info = f"page=ok area={quad_area(order_points(quad_small))/(small.shape[0]*small.shape[1]+1e-9):.2f}"

        if SAVE_DEBUG:
            dbg = img0.copy()
            qi = order_points(q).astype(int)
            cv2.polylines(dbg, [qi], True, (0,255,0), 6)
            imwrite_unicode(os.path.join(DEBUG_DIR, f"{base}_quad.jpg"), dbg)
            imwrite_unicode(os.path.join(DEBUG_DIR, f"{base}_mask0.jpg"),
                            cv2.cvtColor(mask0, cv2.COLOR_GRAY2BGR))
            imwrite_unicode(os.path.join(DEBUG_DIR, f"{base}_warp.jpg"), warped)

    # Orientation
    fixed, meta = choose_best_orientation(warped)

    # Tight crop (masa/kalem vs at)
    fixed2, mask1 = tight_crop_to_page(fixed, TIGHT_CROP_PAD)

    if SAVE_DEBUG:
        imwrite_unicode(os.path.join(DEBUG_DIR, f"{base}_mask1_tight.jpg"),
                        cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR))
        imwrite_unicode(os.path.join(DEBUG_DIR, f"{base}_fixed_before_tight.jpg"), fixed)

    out_path = os.path.join(OUTPUT_DIR, f"{base}_fixed.jpg")
    imwrite_unicode(out_path, fixed2)

    print(
        f"✅ {os.path.basename(path)} -> {os.path.basename(out_path)}"
        f" | {quad_info}"
        f" | mode={meta.get('mode')}"
        f" | deg={meta.get('deg', meta.get('osd_rot'))}"
        f" | osd=({meta.get('osd_rot')},{meta.get('osd_conf')})"
        f" | skew={meta.get('skew', 0.0):.2f}"
        f" | chars={meta.get('chars', 0)}"
        f" | conf={meta.get('mean_conf', 0.0):.1f}"
        f" | h={meta.get('hscore', 0.0):.3f}"
        f" | score={meta.get('score', 0.0):.3f}"
    )

def main():
    ensure_dirs()
    patterns = ["*.jpg","*.jpeg","*.png","*.webp","*.bmp","*.tif","*.tiff"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(INPUT_DIR, p)))

    if not files:
        print(f"⚠️ {INPUT_DIR} içinde görsel yok.")
        return

    for f in sorted(files):
        process_one(f)

    print("Bitti.")

if __name__ == "__main__":
    main()
