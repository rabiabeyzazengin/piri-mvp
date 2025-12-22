import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_images")

sys.path.insert(0, BASE_DIR)
try:
    import sayfa_duzeltme as sd
except Exception as e:
    print("Cannot import sayfa_duzeltme:", e)
    sys.exit(1)


def process_file(path):
    img = sd.imread_unicode(path)
    if img is None:
        print(f"❌ Okunamadı: {path}")
        return

    pref = sd.prefer_upright_by_ocr(img)
    if not np.array_equal(pref, img):
        ok = sd.imwrite_unicode(path, pref)
        if ok:
            print(f"🔁 Döndürüldü: {os.path.basename(path)}")
        else:
            print(f"❌ Kaydedilemedi: {path}")
    else:
        print(f"✔️ Doğru yön: {os.path.basename(path)}")


if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_DIR):
        print("output_images klasörü bulunamadı:", OUTPUT_DIR)
        sys.exit(1)

    files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith('_fixed.jpg')]
    if not files:
        print("Düzleştirilmiş dosya bulunamadı.")
        sys.exit(0)

    for f in sorted(files):
        process_file(os.path.join(OUTPUT_DIR, f))

    print("Bitti.")
