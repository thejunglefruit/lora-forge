# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Dataset validation. numpy + Pillow only, no opencv (brittle on recent Python versions).

Per image: resolution, sharpness (variance of the Laplacian), near-duplicates
(perceptual dHash) and aspect ratio. Face detection is an OPTIONAL GPU-box
component (opencv/insightface); when it is missing the report says so clearly
instead of failing.

Rule: nothing moves on until a human has reviewed the flags.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

# Default thresholds (can be overridden from the registry / requirements card).
BLUR_VAR_THRESHOLD = 100.0      # var(Laplacian) below this -> probably blurry
DHASH_SIZE = 8                  # 64-bit hash
DUP_HAMMING_THRESHOLD = 5       # distance <= this -> near-duplicate
EXTREME_AR = (0.5, 2.0)         # w/h outside this range -> warning


@dataclass
class ImageReport:
    path: Path
    ok: bool = True
    width: int = 0
    height: int = 0
    aspect_ratio: float = 0.0
    blur_var: float = 0.0
    dhash: str = ""
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Filled in by face detection when it is available.
    faces: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.name,
            "ok": self.ok,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": round(self.aspect_ratio, 4),
            "blur_var": round(self.blur_var, 2),
            "dhash": self.dhash,
            "faces": self.faces,
            "flags": self.flags,
            "warnings": self.warnings,
        }


@dataclass
class DatasetReport:
    images: list[ImageReport]
    face_detection_available: bool
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.images)

    @property
    def flagged(self) -> list[ImageReport]:
        return [im for im in self.images if im.flags]

    @property
    def blocked(self) -> bool:
        """Something needs human review before going on."""
        return bool(self.flagged)


# ----------------------------------------------------------------------------- #
# Image primitives
# ----------------------------------------------------------------------------- #
def _to_gray_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float64)


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian (a sharpness proxy). 3x3 convolution in numpy."""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    # Manual 'valid' convolution (avoids depending on scipy).
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    out = (
        k[0, 1] * gray[:-2, 1:-1]
        + k[1, 0] * gray[1:-1, :-2]
        + k[1, 1] * gray[1:-1, 1:-1]
        + k[1, 2] * gray[1:-1, 2:]
        + k[2, 1] * gray[2:, 1:-1]
    )
    return float(out.var())


def dhash(img: Image.Image, size: int = DHASH_SIZE) -> str:
    """Perceptual difference hash. Returns size*size bits as hex."""
    small = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    arr = np.asarray(small, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = diff.flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:0{(size * size) // 4}x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


# ----------------------------------------------------------------------------- #
# Face detection (OPTIONAL, GPU box only)
# ----------------------------------------------------------------------------- #
def face_detector_available() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


def _detect_faces(path: Path) -> int:
    """Counts faces with opencv's Haar cascade. Only called when available."""
    import cv2

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    img = cv2.imread(str(path))
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    return len(faces)


# ----------------------------------------------------------------------------- #
# Main validation
# ----------------------------------------------------------------------------- #
def validate_image(path: Path, res_floor: int, blur_threshold: float = BLUR_VAR_THRESHOLD) -> ImageReport:
    rep = ImageReport(path=path)
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            rep.width, rep.height = im.size
            gray = _to_gray_array(im)
            rep.blur_var = laplacian_variance(gray)
            rep.dhash = dhash(im)
    except Exception as e:  # corrupt image / unsupported format
        rep.ok = False
        rep.flags.append(f"unreadable: {e}")
        return rep

    rep.aspect_ratio = rep.width / rep.height if rep.height else 0.0

    short_side = min(rep.width, rep.height)
    if short_side < res_floor:
        rep.flags.append(f"low resolution: short side {short_side}px < {res_floor}px")
        rep.ok = False

    if rep.blur_var < blur_threshold:
        rep.flags.append(f"possibly blurry: var(Laplacian)={rep.blur_var:.1f} < {blur_threshold:.0f}")

    ar = rep.aspect_ratio
    if ar and (ar < EXTREME_AR[0] or ar > EXTREME_AR[1]):
        rep.warnings.append(f"extreme aspect ratio: {ar:.2f} (backends bucket it, no cropping)")

    return rep


def validate_dataset(
    images: list[Path],
    res_floor: int,
    lora_type: str,
    requirements: dict[str, Any],
    detect_faces: bool = True,
) -> DatasetReport:
    reps: list[ImageReport] = []
    for p in images:
        reps.append(validate_image(p, res_floor))

    # Near-duplicates: pairwise dHash comparison.
    for i in range(len(reps)):
        if not reps[i].dhash:
            continue
        for j in range(i + 1, len(reps)):
            if not reps[j].dhash:
                continue
            if hamming(reps[i].dhash, reps[j].dhash) <= DUP_HAMMING_THRESHOLD:
                msg = f"near-duplicate of {reps[j].path.name}"
                if msg not in reps[i].flags:
                    reps[i].flags.append(msg)
                reps[j].flags.append(f"near-duplicate of {reps[i].path.name}")

    # Faces (optional).
    faces_available = detect_faces and face_detector_available()
    if faces_available:
        for rep in reps:
            if not rep.ok and "unreadable" in " ".join(rep.flags):
                continue
            try:
                rep.faces = _detect_faces(rep.path)
            except Exception:
                rep.faces = None

    # Rules specific to the character type.
    if lora_type == "character":
        for rep in reps:
            if faces_available and rep.faces is not None:
                if rep.faces == 0:
                    rep.flags.append("character: no face detected")
                elif rep.faces > 1:
                    rep.flags.append(f"character: {rep.faces} faces (one subject expected)")
        # The close-up ratio needs face bounding boxes: planned for a later phase.

    report = DatasetReport(images=reps, face_detection_available=faces_available)
    report.summary = _summarize(reps, lora_type, requirements, faces_available)
    return report


def _summarize(
    reps: list[ImageReport], lora_type: str, requirements: dict[str, Any], faces_available: bool
) -> dict[str, Any]:
    n = len(reps)
    imgs_min = requirements.get("imgs_min")
    imgs_rec = requirements.get("imgs_rec")
    summary: dict[str, Any] = {
        "count": n,
        "flagged": sum(1 for r in reps if r.flags),
        "warnings": sum(1 for r in reps if r.warnings),
        "faces_available": faces_available,
        "count_ok": imgs_min is None or n >= imgs_min,
    }
    if imgs_min is not None and n < imgs_min:
        summary["count_warning"] = f"only {n} images; recommended minimum {imgs_min} (ideal {imgs_rec})"
    return summary
