import os
import cv2
from photo_quality_check import check_photo_quality, map_issues_to_messages

FOLDER = "samples"

for fn in sorted(os.listdir(FOLDER)):
    if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(FOLDER, fn)
    img = cv2.imread(path)
    if img is None:
        print(f"[SKIP] {fn} (cannot read)")
        continue

    res = check_photo_quality(img)

    print("\n==============================")
    print("File:", fn)
    print("OK:", res.ok, "Score:", round(res.score, 2))
    print("Issues:", res.issues)
    if not res.ok:
        for m in map_issues_to_messages(res.issues):
            print(" -", m)
