"""
preprocess.py — Document image preprocessing pipeline.

Stages
------
  1  Ingest         — colour mode normalisation (CMYK/RGBA → RGB)
  2  Quality Gate   — blur detection → unsharp mask
                      brightness check → CLAHE / gamma
                      noise estimation → NLM denoising
  3  Geometry       — coarse orientation (90°/180°/270° via projection variance)
                      fine deskew (Hough lines, <45°)
                      perspective correction (4-point homography, optional)
                      margin crop
  4  Binarisation   — shadow removal (divide by blurred background)
                      Sauvola adaptive threshold (optional, off by default)
                      stamp/watermark suppression (HSV masking, optional)
  5  Layout         — table region detection (H+V morphological lines)
                      table grid-line removal (inpainting, only on table regions)
  6  Final          — DPI normalisation to 300 DPI / 2480 px wide
                      super-resolution upscale (optional, off by default)

Entry points
------------
  preprocess_image(file_path, config=None)  -> str        temp PNG path
  preprocess_pdf_pages(file_path, ...)      -> List[str]  one temp PNG per page

The caller is responsible for deleting the returned temp file(s).
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class PreprocessConfig:
    # Stage 2 — Quality Gate
    blur_threshold: float = 80.0  # Laplacian variance below this → sharpen
    dark_threshold: float = 85.0  # Mean brightness below this → CLAHE
    bright_threshold: float = 195.0  # Mean brightness above this → gamma
    gamma_value: float = 0.7  # Gamma exponent for over-exposed images
    noise_threshold: float = 8.0  # Noise estimate above this → NLM denoise

    # Stage 3 — Geometry
    enable_orientation: bool = True  # Coarse 90°/180°/270° correction
    enable_deskew: bool = True  # Fine <45° deskew via Hough lines
    deskew_threshold: float = 0.5  # Skip deskew if |angle| < this (degrees)
    enable_perspective: bool = False  # 4-point warp (best for phone photos)
    enable_margin_crop: bool = True  # Crop excess white borders

    # Stage 4 — Binarisation
    enable_shadow_removal: bool = True  # Divide-by-background illumination fix
    shadow_blur_sigma: float = 50.0  # Gaussian sigma for background model
    enable_sauvola: bool = False  # Sauvola adaptive threshold (grayscale out)
    enable_stamp_removal: bool = False  # HSV colour masking to whiten stamps

    # Stage 5 -- Layout
    enable_table_grid_removal: bool = True

    # Stage 6 — Final
    target_width: int = 2480  # ~300 DPI for A4 portrait (upscale only)
    enable_super_resolution: bool = False  # EDSR / off by default


# ── Internal helpers ──────────────────────────────────────────────────────────


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _load_as_bgr(file_path: str) -> np.ndarray:
    """Load any PIL-supported image as BGR uint8, handling CMYK/RGBA/palette modes."""
    pil = Image.open(file_path)
    mode = pil.mode
    if mode == "CMYK":
        pil = pil.convert("RGB")
    elif mode in ("RGBA", "LA", "P"):
        pil = pil.convert("RGB")
    elif mode not in ("RGB", "L"):
        pil = pil.convert("RGB")
    arr = np.array(pil)
    if arr.ndim == 2:  # grayscale
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _load_bytes_as_bgr(image_bytes: bytes) -> np.ndarray:
    """Load image bytes as BGR uint8, including CMYK/RGBA/palette images."""
    if not image_bytes:
        raise ValueError("Image data is empty.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil:
            if pil.mode not in ("RGB", "L"):
                pil = pil.convert("RGB")
            arr = np.array(pil)
    except (OSError, ValueError) as exc:
        raise ValueError("Image data could not be decoded.") from exc
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ── Stage 2 — Quality Gate ────────────────────────────────────────────────────


def _detect_blur(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _unsharp_mask(
    img: np.ndarray, sigma: float = 1.5, amount: float = 1.5
) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def _mean_brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))


def _apply_clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge([lightness, a, b]), cv2.COLOR_LAB2BGR)


def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(img, table)


def _estimate_noise(gray: np.ndarray) -> float:
    """Noise estimate via mean absolute Laplacian energy (independent of blur variance)."""
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float(np.sqrt(np.mean(lap**2)) / 6.0)


def _denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(
        img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
    )


# ── Stage 3 — Geometry ────────────────────────────────────────────────────────


def _projection_variance(gray: np.ndarray) -> float:
    proj = np.sum(gray.astype(np.float64), axis=1)
    return float(np.var(proj))


def _coarse_orientation(img: np.ndarray) -> np.ndarray:
    """
    Test rotations at 0/90/180/270° and keep the one with the highest
    horizontal projection variance (text lines → high row-sum variance).
    """
    gray = _to_gray(img)
    best_k, best_var = 0, _projection_variance(gray)

    for k in (1, 2, 3):  # np.rot90 steps (CCW)
        rotated = np.rot90(gray, k=k)
        var = _projection_variance(rotated)
        if var > best_var:
            best_var, best_k = var, k

    if best_k == 0:
        return img

    angle = best_k * 90
    logger.info("  [preprocess] Coarse orientation: rotating %d°", angle)
    return np.ascontiguousarray(np.rot90(img, k=best_k))


def _deskew(img: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Detect dominant text-line angle via Hough lines and correct if significant."""
    gray = _to_gray(img)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    min_votes = max(img.shape[0], img.shape[1]) // 4
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=min_votes)
    if lines is None:
        return img

    angles = []
    for line in lines:
        rho, theta = line[0]
        # theta ∈ [0, π]; angle from horizontal = degrees(theta) - 90
        angle = float(np.degrees(theta)) - 90.0
        if abs(angle) <= 45.0:
            angles.append(angle)

    if len(angles) < 5:
        return img

    median_angle = float(np.median(angles))
    if abs(median_angle) < threshold:
        return img

    logger.info("  [preprocess] Deskew: correcting %.2f°", median_angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -median_angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left  (smallest x+y)
    rect[2] = pts[np.argmax(s)]  # bottom-right (largest x+y)
    rect[1] = pts[np.argmin(diff)]  # top-right (smallest x-y)
    rect[3] = pts[np.argmax(diff)]  # bottom-left (largest x-y)
    return rect


def _four_point_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = _order_points(pts)
    max_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    max_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(_order_points(pts), dst)
    return cv2.warpPerspective(img, M, (max_w, max_h))


def _perspective_correct(img: np.ndarray) -> np.ndarray:
    """Find the largest quadrilateral and warp it to a flat rectangle."""
    gray = _to_gray(img)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    img_area = img.shape[0] * img.shape[1]
    for c in contours:
        if cv2.contourArea(c) < img_area * 0.2:
            break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            logger.info("  [preprocess] Perspective correction applied")
            return _four_point_transform(img, approx.reshape(4, 2).astype(np.float32))

    return img  # no suitable quadrilateral found


def _margin_crop(img: np.ndarray, pad: int = 8) -> np.ndarray:
    """Crop to bounding box of non-white content, adding a small padding."""
    gray = _to_gray(img)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)
    ih, iw = img.shape[:2]
    x = max(0, x - pad)
    y = max(0, y - pad)
    x2 = min(iw, x + w + 2 * pad)
    y2 = min(ih, y + h + 2 * pad)

    # Only crop if removing >2 % on at least one edge
    if x < iw * 0.02 and y < ih * 0.02 and x2 > iw * 0.98 and y2 > ih * 0.98:
        return img

    logger.info("  [preprocess] Margin crop: %dx%d → %dx%d", iw, ih, x2 - x, y2 - y)
    return img[y:y2, x:x2]


# ── Stage 4 — Binarisation ────────────────────────────────────────────────────


def _shadow_removal(img: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    """Divide each channel by a blurred version of itself to flatten illumination."""
    img_f = img.astype(np.float32) + 1.0
    blurred = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma)
    result = np.clip((img_f / blurred) * 128.0, 0, 255).astype(np.uint8)
    return result


def _sauvola(img: np.ndarray, window: int = 25) -> np.ndarray:
    """Sauvola adaptive thresholding (requires scikit-image). Returns grayscale binary."""
    try:
        from skimage.filters import threshold_sauvola  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            "scikit-image is required for Sauvola thresholding.\n"
            "Install it with: pip install scikit-image"
        )
    gray = _to_gray(img)
    thresh = threshold_sauvola(gray, window_size=window)
    binary = ((gray > thresh) * 255).astype(np.uint8)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _stamp_suppression(img: np.ndarray) -> np.ndarray:
    """Whiten highly-saturated regions (coloured stamps / watermarks)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([180, 255, 255]))
    # Erode to avoid touching dark text next to a stamp
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    result = img.copy()
    result[mask > 0] = 255
    return result


# ── Stage 5 — Layout ──────────────────────────────────────────────────────────


def _detect_table_regions(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Return bounding rectangles of table-like regions detected by morphological
    H + V line extraction. Only regions where both orientations overlap are kept.
    """
    gray = _to_gray(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = img.shape[:2]
    h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 20), 1))
    v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 20)))

    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kern)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kern)

    # Only keep intersections (pixels where BOTH H and V lines exist)
    intersection = cv2.bitwise_and(h_lines, v_lines)
    # Dilate to merge nearby intersections into table regions
    dilate_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, h // 10))
    table_mask = cv2.dilate(intersection, dilate_kern)

    contours, _ = cv2.findContours(
        table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    tables = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > h * w * 0.01:  # at least 1 % of image
            tables.append(cv2.boundingRect(c))
    return tables


def _remove_table_gridlines(
    img: np.ndarray, table_rects: List[Tuple[int, int, int, int]]
) -> np.ndarray:
    """Inpaint H+V grid lines inside every detected table region."""
    result = img.copy()
    for x, y, tw, th in table_rects:
        roi = result[y : y + th, x : x + tw]
        gray_roi = _to_gray(roi)
        _, binary = cv2.threshold(
            gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, tw // 20), 1))
        v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, th // 20)))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kern)
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kern)
        lines_mask = cv2.add(h_lines, v_lines)

        repaired = cv2.inpaint(
            roi, lines_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
        )
        result[y : y + th, x : x + tw] = repaired
    return result


# ── Stage 6 — Final Enhancement ───────────────────────────────────────────────


def _normalise_dpi(img: np.ndarray, target_width: int = 2480) -> np.ndarray:
    """Upscale image width to target_width if smaller; downscale if 2× larger."""
    h, w = img.shape[:2]
    if w == target_width:
        return img
    if w > target_width * 2:
        # Mildly oversized — shrink to target to reduce token / memory usage
        new_h = int(h * target_width / w)
        logger.info(
            "  [preprocess] DPI normalise (downscale): %dx%d → %dx%d",
            w,
            h,
            target_width,
            new_h,
        )
        return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)
    if w < target_width:
        new_h = int(h * target_width / w)
        logger.info(
            "  [preprocess] DPI normalise (upscale): %dx%d → %dx%d",
            w,
            h,
            target_width,
            new_h,
        )
        return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_LANCZOS4)
    return img


# ── Master pipeline ───────────────────────────────────────────────────────────


def _run_pipeline(img: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Apply all preprocessing stages in order to a single BGR image."""

    # ── Stage 2: Quality Gate ─────────────────────────────────────────────────
    gray = _to_gray(img)

    blur_var = _detect_blur(gray)
    if blur_var < cfg.blur_threshold:
        logger.info("  [preprocess] Blur detected (var=%.1f) → unsharp mask", blur_var)
        img = _unsharp_mask(img)
        gray = _to_gray(img)

    brightness = _mean_brightness(gray)
    if brightness < cfg.dark_threshold:
        logger.info("  [preprocess] Dark image (mean=%.1f) → CLAHE", brightness)
        img = _apply_clahe(img)
        gray = _to_gray(img)
    elif brightness > cfg.bright_threshold:
        logger.info("  [preprocess] Over-exposed image (mean=%.1f) → gamma", brightness)
        img = _apply_gamma(img, cfg.gamma_value)
        gray = _to_gray(img)

    noise = _estimate_noise(gray)
    if noise > cfg.noise_threshold:
        logger.info("  [preprocess] Noisy image (est=%.1f) → NLM denoising", noise)
        img = _denoise(img)
        gray = _to_gray(img)

    # ── Stage 3: Geometry ─────────────────────────────────────────────────────
    if cfg.enable_orientation:
        img = _coarse_orientation(img)
        gray = _to_gray(img)

    if cfg.enable_deskew:
        img = _deskew(img, threshold=cfg.deskew_threshold)
        gray = _to_gray(img)

    if cfg.enable_perspective:
        img = _perspective_correct(img)
        gray = _to_gray(img)

    if cfg.enable_margin_crop:
        img = _margin_crop(img)
        gray = _to_gray(img)  # noqa: F841  (kept for symmetry)

    # ── Stage 4: Binarisation ─────────────────────────────────────────────────
    if cfg.enable_shadow_removal:
        logger.info("  [preprocess] Shadow removal")
        img = _shadow_removal(img, sigma=cfg.shadow_blur_sigma)

    if cfg.enable_sauvola:
        logger.info("  [preprocess] Sauvola binarisation")
        img = _sauvola(img)

    if cfg.enable_stamp_removal:
        logger.info("  [preprocess] Stamp/watermark suppression")
        img = _stamp_suppression(img)

    # ── Stage 5: Layout — table grid removal ──────────────────────────────────
    table_rects = _detect_table_regions(img) if cfg.enable_table_grid_removal else []
    if table_rects:
        logger.info(
            "  [preprocess] %d table region(s) → grid-line removal", len(table_rects)
        )
        img = _remove_table_gridlines(img, table_rects)

    # ── Stage 6: DPI normalisation ────────────────────────────────────────────
    img = _normalise_dpi(img, target_width=cfg.target_width)

    return img


# ── Public entry points ───────────────────────────────────────────────────────


def conservative_ocr_config() -> PreprocessConfig:
    """Return the non-destructive profile used for optional upload enhancement."""
    return PreprocessConfig(
        enable_perspective=False,
        enable_sauvola=False,
        enable_stamp_removal=False,
        enable_table_grid_removal=False,
    )


def enhanced_image_filename(filename: str) -> str:
    """Return a truthful PNG filename for an enhanced upload."""
    return f"{Path(filename).stem}-enhanced.png"


def upload_payload(
    original_bytes: bytes,
    original_filename: str,
    original_content_type: str,
    enhanced_bytes: Optional[bytes] = None,
) -> tuple[bytes, str, str]:
    """Choose the original upload or its enhanced PNG replacement."""
    if enhanced_bytes is None:
        return original_bytes, original_filename, original_content_type
    return enhanced_bytes, enhanced_image_filename(original_filename), "image/png"


def preprocess_image_bytes(
    image_bytes: bytes, config: Optional[PreprocessConfig] = None
) -> bytes:
    """Preprocess image bytes and return the result as PNG bytes."""
    img = _load_bytes_as_bgr(image_bytes)
    img = _run_pipeline(img, config or conservative_ocr_config())
    success, encoded = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Enhanced image could not be encoded as PNG.")
    return encoded.tobytes()


def preprocess_image(file_path: str, config: Optional[PreprocessConfig] = None) -> str:
    """
    Preprocess a single image file (JPEG, PNG, TIFF, BMP, WebP …).

    Returns the path to a temporary PNG file containing the preprocessed image.
    The **caller is responsible for deleting the temp file** when finished.
    """
    if config is None:
        config = PreprocessConfig()

    img = _load_as_bgr(file_path)
    img = _run_pipeline(img, config)

    fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="preproc_")
    os.close(fd)
    cv2.imwrite(tmp_path, img)
    logger.info("  [preprocess] Saved → %s", tmp_path)
    return tmp_path


def preprocess_pdf_pages(
    file_path: str, config: Optional[PreprocessConfig] = None, pdf_dpi: int = 200
) -> List[str]:
    """
    Convert every PDF page to an image, preprocess each one, and return a list
    of temporary PNG paths (one per page).

    Requires ``pdf2image`` (and Poppler).  Install with:
        pip install pdf2image
    """
    try:
        from pdf2image import convert_from_path  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            "pdf2image is not installed.\n"
            "Install it with: pip install pdf2image\n"
            "(Poppler must also be available on PATH.)"
        )

    if config is None:
        config = PreprocessConfig()

    pil_pages = convert_from_path(file_path, dpi=pdf_dpi)
    temp_paths: List[str] = []

    for i, page in enumerate(pil_pages):
        arr = np.array(page.convert("RGB"))
        img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        img = _run_pipeline(img, config)

        fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix=f"preproc_p{i + 1}_")
        os.close(fd)
        cv2.imwrite(tmp_path, img)
        temp_paths.append(tmp_path)
        logger.info("  [preprocess] Page %d/%d → %s", i + 1, len(pil_pages), tmp_path)

    return temp_paths
