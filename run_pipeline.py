#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _pick_python(piri_root: Path) -> str:
    """
    Önce piri-mvp-main içindeki venv'i dener, yoksa sistem python3 kullanır.
    """
    venv_py = piri_root / "venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return "python3"


def _run_quality_check(piri_root: Path, input_path: Path):
    """
    photo_quality_check.py varsa import ederek kalite raporu üretir.
    Yoksa "ok" döner (fail-open).
    """
    try:
        sys.path.insert(0, str(piri_root))
        from photo_quality_check import check_photo_quality, map_issues_to_messages  # type: ignore

        res = check_photo_quality(str(input_path))

        ok = bool(getattr(res, "ok", True))
        score = float(getattr(res, "score", 1.0))
        issues = list(getattr(res, "issues", []) or [])
        metrics = dict(getattr(res, "metrics", {}) or {})
        messages = map_issues_to_messages(issues)

        return {
            "ok": ok,
            "score": score,
            "issues": [str(x) for x in issues],
            "messages": [str(x) for x in messages],
            "metrics": metrics,
        }
    except Exception:
        return {
            "ok": True,
            "score": 1.0,
            "issues": [],
            "messages": [],
            "metrics": {},
        }


def _clear_dir(dir_path: Path):
    if not dir_path.exists():
        return
    for f in dir_path.glob("*"):
        if f.is_file():
            try:
                f.unlink()
            except Exception:
                pass


def _pick_latest_output(output_images: Path) -> Path:
    candidates = list(output_images.glob("*_scan.*"))
    if not candidates:
        candidates = list(output_images.glob("*_fixed.*"))
    if not candidates:
        raise RuntimeError("Çıktı bulunamadı. output_images içinde *_scan.* ya da *_fixed.* yok.")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _run_sayfa_duzeltme(piri_root: Path, python_bin: str, input_path: Path, output_dir: Path) -> Path:
    """
    sayfa_duzeltme.py batch çalışıyor:
    - piri_root/input_images/* alır
    - piri_root/output_images/*_scan.jpg üretir
    Tek dosya için:
    - input_images içine kopyalarız
    - sayfa_duzeltme.py'yi çalıştırırız
    - üretilen *_scan.jpg'i output_dir içine kopyalarız
    """
    input_images = piri_root / "input_images"
    output_images = piri_root / "output_images"
    _safe_mkdir(input_images)
    _safe_mkdir(output_images)
    _safe_mkdir(output_dir)

    # Debug için temizleme (istersen kaldır)
    _clear_dir(input_images)
    _clear_dir(output_images)

    # input'u kopyala
    dst_input = input_images / input_path.name
    shutil.copy2(str(input_path), str(dst_input))

    # sayfa_duzeltme.py çalıştır
    script = piri_root / "sayfa_duzeltme.py"
    if not script.exists():
        raise RuntimeError(f"sayfa_duzeltme.py bulunamadı: {script}")

    proc = subprocess.run(
        [python_bin, str(script)],
        cwd=str(piri_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"sayfa_duzeltme.py hata verdi.\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}")

    produced = _pick_latest_output(output_images)

    final_out = output_dir / produced.name
    shutil.copy2(str(produced), str(final_out))
    return final_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input image path")
    ap.add_argument("--output_dir", required=True, help="output directory")
    ap.add_argument("--jpeg_quality", default="95", help="(şimdilik kullanılmıyor)")
    args = ap.parse_args()

    piri_root = Path(__file__).resolve().parent
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    python_bin = _pick_python(piri_root)

    quality = _run_quality_check(piri_root, input_path)

    out_path = _run_sayfa_duzeltme(piri_root, python_bin, input_path, output_dir)

    payload = {
        "ok": bool(quality.get("ok", True)),
        "score": float(quality.get("score", 1.0)),
        "issues": quality.get("issues", []),
        "messages": quality.get("messages", []),
        "metrics": quality.get("metrics", {}),
        "output_path": str(out_path),
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
