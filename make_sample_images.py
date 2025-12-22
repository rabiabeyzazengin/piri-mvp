import os
import cv2
import numpy as np

OUT_DIR = "samples"
os.makedirs(OUT_DIR, exist_ok=True)

def put_text(img, text, y):
    cv2.putText(img, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10,10,10), 2, cv2.LINE_AA)


base = np.full((1100, 800, 3), 230, dtype=np.uint8)  
paper = np.full((900, 650, 3), 255, dtype=np.uint8)  


x0, y0 = 75, 120
base[y0:y0+paper.shape[0], x0:x0+paper.shape[1]] = paper


for i, line in enumerate([
    "PIRI - Sample Exam Paper",
    "Student: EZGI AYNAL",
    "Q1) A   Q2) C   Q3) B   Q4) D",
    "Q5) A   Q6) B   Q7) D   Q8) C",
    "Score: ____ / 100"
]):
    put_text(base, line, 200 + i*80)

cv2.imwrite(os.path.join(OUT_DIR, "good.jpg"), base)


blur = cv2.GaussianBlur(base, (21, 21), 0)
cv2.imwrite(os.path.join(OUT_DIR, "blur.jpg"), blur)


dark = (base * 0.35).astype(np.uint8)
cv2.imwrite(os.path.join(OUT_DIR, "dark.jpg"), dark)


bright = cv2.convertScaleAbs(base, alpha=1.0, beta=60)
cv2.imwrite(os.path.join(OUT_DIR, "bright.jpg"), bright)


low_contrast = cv2.convertScaleAbs(base, alpha=0.6, beta=60)
cv2.imwrite(os.path.join(OUT_DIR, "low_contrast.jpg"), low_contrast)


cropped = base[:, 120:]  
cv2.imwrite(os.path.join(OUT_DIR, "cropped.jpg"), cropped)


h, w = base.shape[:2]
src = np.float32([[80, 120], [725, 120], [725, 1020], [80, 1020]])
dst = np.float32([[140, 160], [690, 90], [760, 990], [60, 1050]])
M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(base, M, (w, h), borderValue=(230,230,230))
cv2.imwrite(os.path.join(OUT_DIR, "warped.jpg"), warped)

print(f"✅ Sample images created under: {OUT_DIR}/")
