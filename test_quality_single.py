import cv2
from photo_quality_check import check_photo_quality, map_issues_to_messages

IMAGE_PATH = "samples/blur.jpg" 

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise SystemExit(f"Image not found: {IMAGE_PATH}")

res = check_photo_quality(img)

print("OK:", res.ok)
print("Score:", res.score)
print("Issues:", res.issues)
print("Metrics:", res.metrics)

if not res.ok:
    print("\nUser messages:")
    for m in map_issues_to_messages(res.issues):
        print("-", m)
