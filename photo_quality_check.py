import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class QualityResult:
    ok: bool
    score: float
    issues: List[str]
    metrics: Dict[str, Any]


def variance_of_laplacian(gray: np.ndarray) -> float:
    """Blur tespiti için yaygın metrik: Laplacian variance."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_contrast(gray: np.ndarray) -> tuple[float, float]:
    """Parlaklık (mean) ve kontrast (std)."""
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    return mean, std


def find_paper_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Kağıt konturu tespiti (basit):
    - Edge + kontur
    - 4 köşeye yakın bir kontur bulmaya çalışır
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx  
    return None


def contour_area_ratio(image: np.ndarray, contour: np.ndarray) -> float:
    """Kağıt kadraja sığmış mı? Alan oranı ile kabaca kontrol."""
    img_area = image.shape[0] * image.shape[1]
    c_area = cv2.contourArea(contour)
    return float(c_area / img_area) if img_area else 0.0


def check_photo_quality(
    image_bgr: np.ndarray,
    blur_threshold: float = 120.0,
    min_brightness: float = 60.0,
    max_brightness: float = 210.0,
    min_contrast: float = 25.0,
    min_paper_area_ratio: float = 0.35
) -> QualityResult:
    """
    Basit kalite kontrol:
    - blur (laplacian var)
    - brightness/contrast
    - paper contour + alan oranı
    """
    issues: List[str] = []
    metrics: Dict[str, Any] = {}

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 1) Blur
    blur_score = variance_of_laplacian(gray)
    metrics["blur_laplacian_var"] = blur_score
    if blur_score < blur_threshold:
        issues.append("BLUR_LOW")

    # 2) Brightness / contrast
    mean_b, std_c = brightness_contrast(gray)
    metrics["brightness_mean"] = mean_b
    metrics["contrast_std"] = std_c

    if mean_b < min_brightness:
        issues.append("LIGHT_TOO_DARK")
    if mean_b > max_brightness:
        issues.append("LIGHT_TOO_BRIGHT")
    if std_c < min_contrast:
        issues.append("CONTRAST_LOW")

    # 3) Paper in frame
    paper = find_paper_contour(image_bgr)
    metrics["paper_found"] = paper is not None

    if paper is None:
        issues.append("PAPER_NOT_DETECTED")
        area_ratio = 0.0
    else:
        area_ratio = contour_area_ratio(image_bgr, paper)
        metrics["paper_area_ratio"] = area_ratio
        if area_ratio < min_paper_area_ratio:
            issues.append("PAPER_TOO_SMALL_OR_CROPPED")

    ok = len(issues) == 0


    score = 1.0
    if "BLUR_LOW" in issues:
        score -= 0.35
    if "LIGHT_TOO_DARK" in issues or "LIGHT_TOO_BRIGHT" in issues:
        score -= 0.25
    if "CONTRAST_LOW" in issues:
        score -= 0.15
    if "PAPER_NOT_DETECTED" in issues:
        score -= 0.35
    if "PAPER_TOO_SMALL_OR_CROPPED" in issues:
        score -= 0.25
    score = max(0.0, min(1.0, score))

    return QualityResult(ok=ok, score=score, issues=issues, metrics=metrics)


def map_issues_to_messages(issues: List[str]) -> List[str]:
    msg_map = {
        "BLUR_LOW": "Fotoğraf bulanık görünüyor. Lütfen telefonu sabit tutup yeniden çekiniz.",
        "LIGHT_TOO_DARK": "Işık düşük olduğu için metin okunamıyor. Daha aydınlık ortamda tekrar deneyiniz.",
        "LIGHT_TOO_BRIGHT": "Aşırı parlaklık/yansıma tespit edildi. Kağıdı ışığa göre konumlandırarak yeniden çekiniz.",
        "CONTRAST_LOW": "Kontrast düşük. Daha net görünmesi için daha iyi ışıkta yeniden çekiniz.",
        "PAPER_NOT_DETECTED": "Kağıt algılanamadı. Kağıdın tamamı kadrajda olacak şekilde yeniden deneyiniz.",
        "PAPER_TOO_SMALL_OR_CROPPED": "Kağıt kadraja tam sığmamış veya çok küçük. Daha yakından ve tam çerçevede çekiniz.",
    }
    return [msg_map.get(i, "Fotoğraf kalitesi uygun değil. Lütfen yeniden deneyiniz.") for i in issues]
