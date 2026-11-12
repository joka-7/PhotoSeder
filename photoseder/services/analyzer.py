"""Local image quality and scene analyzer (OpenCV + Pillow, fully offline).

Architecture:
    Analysis is split into two phases so thresholds can change without
    re-reading pixels:

    1. **Measure** (:meth:`OpenCvImageAnalyzer.analyze`, expensive): decode the
       image once, downscale it to a fixed working size and record raw
       measurements — Laplacian variance, luminance histogram statistics,
       face/eye boxes, the dHash and EXIF facts.
    2. **Score** (:meth:`OpenCvImageAnalyzer.rescore`, pure arithmetic):
       normalize the measurements into ``[0, 1]`` scores against the current
       :class:`AnalysisConfig`, classify the scene, map the overall score to
       stars and decide keep/reject. Fields locked by the user are untouched.

Algorithms:
    **Sharpness.** After a 3x3 Gaussian pre-blur (suppresses sensor noise that
    would otherwise read as detail), the Laplacian ``L = d2I/dx2 + d2I/dy2`` is
    computed on the grayscale working image. The image is divided into an
    8 x 8 grid and ``Var(L)`` is taken per tile; the 90th-percentile tile
    variance is the sharpness measurement. Using the sharpest tiles rather
    than the whole frame keeps a crisp subject on a deliberately blurred
    background (bokeh, macro) from being flagged as blurry. Because variance
    depends on resolution, it is always measured at a long edge of 1024 px.
    ``sharpness = min(1, variance / (3 * blur_threshold))``, so a photo exactly at
    the blur threshold scores 0.33 and one three times above it scores 1.0.

    **Exposure and contrast.** From the luminance histogram:
    ``clip`` penalizes the fraction of pixels at the extremes (<= 5 or
    >= 250) beyond a 5 % allowance; ``contrast = min(1, sigma / 50)`` uses the
    RMS contrast (standard deviation of luminance); ``brightness`` is 1 for a
    mean in ``[60, 190]`` and falls linearly to 0 at pure black / white.
    ``exposure = 0.40 * clip + 0.35 * contrast + 0.25 * brightness``.

    **Overall score.** ``0.55 * sharpness + 0.25 * exposure +
    0.10 * resolution + 0.10 * eyes_open``. When no face is detected the
    eyes term carries no information, so the remaining weights are
    renormalized (divided by 0.90) — otherwise a perfect landscape could
    never exceed 0.90.

    **Quality gates.** The weighted sum alone cannot reject a blurry photo
    that is well exposed (it would still score about 0.43). Therefore a photo
    whose Laplacian variance is below ``blur_threshold`` or whose exposure
    score is below ``exposure_gate`` has its overall score capped just below
    ``reject_threshold`` and is rejected with a specific reason.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from photoseder.domain.enums import Orientation, RejectionCategory, SceneType
from photoseder.domain.models import (
    AnalysisConfig,
    FaceBox,
    PhotoItem,
    QualityMetrics,
    new_id,
)
from photoseder.services.deduplicator import compute_dhash
from photoseder.services.interfaces import FaceDetector

try:  # Optional HEIC/HEIF decoding; absent by default.
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover - depends on the local environment
    pillow_heif = None

type Hasher = Callable[[np.ndarray], int]

WORKING_LONG_EDGE = 1024
"""Long edge (px) every image is downscaled to before measuring."""

TILE_GRID = 8
SHARP_REFERENCE_FACTOR = 3.0
"""Multiple of ``blur_threshold`` at which sharpness saturates at 1.0."""
SHARP_TILE_PERCENTILE = 90.0

WEIGHT_SHARPNESS = 0.55
WEIGHT_EXPOSURE = 0.25
WEIGHT_RESOLUTION = 0.10
WEIGHT_EYES = 0.10

STAR_THRESHOLDS: tuple[tuple[float, int], ...] = (
    (0.85, 5),
    (0.70, 4),
    (0.55, 3),
    (0.40, 2),
)

_EXIF_ORIENTATION = 0x0112
_EXIF_IFD_POINTER = 0x8769
_EXIF_DATETIME_ORIGINAL = 0x9003
_EXIF_DATETIME_DIGITIZED = 0x9004
_EXIF_DATETIME = 0x0132

_CLIP_LOW_LEVEL = 5
_CLIP_HIGH_LEVEL = 250
_CLIP_ALLOWANCE = 0.05
_CLIP_RANGE = 0.45
_CONTRAST_REFERENCE = 50.0
_BRIGHT_LOW = 60.0
_BRIGHT_HIGH = 190.0
_SQUARE_TOLERANCE = 0.05


class ImageReadError(ValueError):
    """Raised when a file cannot be decoded as an image."""


def star_rating_for(score: float) -> int:
    """Map an overall score in ``[0, 1]`` to a 1-5 star rating."""
    for threshold, stars in STAR_THRESHOLDS:
        if score >= threshold:
            return stars
    return 1


def classify_orientation(width: int, height: int) -> Orientation:
    """Classify display orientation; within 5 % of 1:1 counts as square."""
    ratio = width / height if height else 1.0
    if abs(ratio - 1.0) <= _SQUARE_TOLERANCE:
        return Orientation.SQUARE
    return Orientation.HORIZONTAL if ratio > 1.0 else Orientation.VERTICAL


def read_image_bgr(path: Path, long_edge_hint: int = 0) -> np.ndarray:
    """Decode an image into a BGR array, safely for non-ASCII paths.

    ``cv2.imread`` cannot open paths with non-ASCII characters (for example
    Hebrew folder names) on Windows, so the file is read into memory with
    ``np.fromfile`` and decoded with ``cv2.imdecode``. EXIF orientation is
    *not* applied here; see :func:`apply_exif_orientation`.

    Args:
        path: Image file.
        long_edge_hint: Known long edge in pixels. When large, JPEGs are
            decoded at 1/2 or 1/4 scale directly in the decoder, which is
            several times faster and sufficient for a 1024 px working copy.

    Returns:
        ``uint8`` array of shape ``(height, width, 3)`` in BGR order.

    Raises:
        ImageReadError: If neither OpenCV nor Pillow can decode the file.
    """
    flags = cv2.IMREAD_COLOR
    if long_edge_hint >= WORKING_LONG_EDGE * 4:
        flags = cv2.IMREAD_REDUCED_COLOR_4
    elif long_edge_hint >= WORKING_LONG_EDGE * 2:
        flags = cv2.IMREAD_REDUCED_COLOR_2
    flags |= cv2.IMREAD_IGNORE_ORIENTATION

    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ImageReadError(f"Cannot read file: {exc}") from exc
    image = cv2.imdecode(buffer, flags) if buffer.size else None
    if image is not None:
        return image

    # Fallback for formats OpenCV cannot decode (e.g. HEIC with pillow-heif).
    try:
        with Image.open(path) as pil_image:
            rgb = np.asarray(pil_image.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        hint = ""
        if path.suffix.lower() in {".heic", ".heif"} and pillow_heif is None:
            hint = " (install 'pillow-heif' to enable HEIC support)"
        raise ImageReadError(f"Unsupported or corrupt image{hint}") from exc
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def apply_exif_orientation(image: np.ndarray, orientation_tag: int) -> np.ndarray:
    """Rotate/flip pixels so the image is upright (EXIF orientation 1-8)."""
    match orientation_tag:
        case 2:
            return cv2.flip(image, 1)
        case 3:
            return cv2.rotate(image, cv2.ROTATE_180)
        case 4:
            return cv2.flip(image, 0)
        case 5:
            return cv2.transpose(image)
        case 6:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        case 7:
            return cv2.flip(cv2.transpose(image), -1)
        case 8:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        case _:
            return image


def resize_long_edge(image: np.ndarray, long_edge: int) -> np.ndarray:
    """Downscale so the long edge is at most ``long_edge`` (never upscales)."""
    height, width = image.shape[:2]
    scale = long_edge / max(height, width)
    if scale >= 1.0:
        return image
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def measure_sharpness(gray: np.ndarray) -> float:
    """Return the 90th-percentile tile variance of the Laplacian.

    Args:
        gray: Grayscale working image.

    Returns:
        ``Var(Laplacian)`` of the sharpest tiles; higher means sharper.
    """
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
    laplacian = cv2.Laplacian(smoothed, cv2.CV_64F)
    height, width = laplacian.shape
    tile_h, tile_w = height // TILE_GRID, width // TILE_GRID
    if tile_h < 4 or tile_w < 4:
        return float(laplacian.var())
    tiles = laplacian[: tile_h * TILE_GRID, : tile_w * TILE_GRID].reshape(
        TILE_GRID, tile_h, TILE_GRID, tile_w
    )
    variances = tiles.var(axis=(1, 3))
    return float(np.percentile(variances, SHARP_TILE_PERCENTILE))


def measure_exposure(gray: np.ndarray) -> tuple[float, float, float, float]:
    """Measure luminance statistics from the 256-bin histogram.

    Args:
        gray: Grayscale working image.

    Returns:
        ``(mean_luminance, rms_contrast, clipped_low, clipped_high)`` where
        the clipped values are pixel fractions at the dark / bright extremes.
    """
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = float(histogram.sum()) or 1.0
    probabilities = histogram / total
    levels = np.arange(256, dtype=np.float64)
    mean = float((probabilities * levels).sum())
    variance = float((probabilities * (levels - mean) ** 2).sum())
    clipped_low = float(probabilities[: _CLIP_LOW_LEVEL + 1].sum())
    clipped_high = float(probabilities[_CLIP_HIGH_LEVEL:].sum())
    return mean, variance**0.5, clipped_low, clipped_high


def exposure_score(metrics: QualityMetrics) -> float:
    """Combine clipping, contrast and brightness into a ``[0, 1]`` score."""
    excess = max(0.0, metrics.clipped_low - _CLIP_ALLOWANCE) + max(
        0.0, metrics.clipped_high - _CLIP_ALLOWANCE
    )
    clip = 1.0 - min(1.0, excess / _CLIP_RANGE)
    contrast = min(1.0, metrics.rms_contrast / _CONTRAST_REFERENCE)
    mean = metrics.mean_luminance
    if mean < _BRIGHT_LOW:
        brightness = mean / _BRIGHT_LOW
    elif mean > _BRIGHT_HIGH:
        brightness = (255.0 - mean) / (255.0 - _BRIGHT_HIGH)
    else:
        brightness = 1.0
    return 0.40 * clip + 0.35 * contrast + 0.25 * max(0.0, brightness)


def describe_exposure_problem(metrics: QualityMetrics) -> str:
    """Name the dominant exposure defect for a rejection reason."""
    too_bright = metrics.clipped_high > _CLIP_ALLOWANCE or (metrics.mean_luminance > _BRIGHT_HIGH)
    too_dark = metrics.clipped_low > _CLIP_ALLOWANCE or (metrics.mean_luminance < _BRIGHT_LOW)
    if too_bright and metrics.clipped_high >= metrics.clipped_low:
        return "Overexposed"
    if too_dark:
        return "Underexposed"
    return "Low contrast"


def classify_scene(
    metrics: QualityMetrics, orientation: Orientation, config: AnalysisConfig
) -> SceneType:
    """Apply the scene rules.

    * 3+ faces, or 1-2 faces covering less than ``portrait_face_ratio`` of
      the frame -> ``GROUP``.
    * 1-2 faces covering at least ``portrait_face_ratio`` -> ``PORTRAIT``.
    * No faces, horizontal and sharp -> ``LANDSCAPE``.
    * Everything else -> ``GENERAL``.
    """
    faces = metrics.face_count
    if faces >= 3:
        return SceneType.GROUP
    if faces >= 1:
        if metrics.face_area_ratio >= config.portrait_face_ratio:
            return SceneType.PORTRAIT
        return SceneType.GROUP
    if (
        orientation is Orientation.HORIZONTAL
        and metrics.sharpness >= config.landscape_min_sharpness
    ):
        return SceneType.LANDSCAPE
    return SceneType.GENERAL


class HaarFaceDetector:
    """Frontal face and eye detector based on OpenCV Haar cascades.

    Haar cascades are fast and ship with OpenCV, but they are prone to false
    positives. Mitigations applied here: histogram equalization, a strict
    ``minNeighbors``, a minimum face size relative to the frame, and searching
    for eyes only in the upper 60 % of each face box.

    ``cv2.CascadeClassifier`` is not safe for concurrent calls, so each
    thread lazily creates its own pair of classifiers.
    """

    def __init__(
        self,
        *,
        scale_factor: float = 1.1,
        face_min_neighbors: int = 6,
        eye_min_neighbors: int = 5,
        min_face_fraction: float = 0.06,
    ) -> None:
        """Configure the detector.

        Args:
            scale_factor: Image pyramid step of the cascade.
            face_min_neighbors: Overlapping detections required for a face.
            eye_min_neighbors: Overlapping detections required for an eye.
            min_face_fraction: Smallest face side as a fraction of the short
                image edge.
        """
        self._scale_factor = scale_factor
        self._face_min_neighbors = face_min_neighbors
        self._eye_min_neighbors = eye_min_neighbors
        self._min_face_fraction = min_face_fraction
        self._local = threading.local()

    def _classifiers(self) -> tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
        """Return this thread's (face, eye) classifiers, loading on first use."""
        cached = getattr(self._local, "classifiers", None)
        if cached is None:
            directory = Path(cv2.data.haarcascades)
            face = cv2.CascadeClassifier(str(directory / "haarcascade_frontalface_default.xml"))
            eye = cv2.CascadeClassifier(str(directory / "haarcascade_eye.xml"))
            if face.empty() or eye.empty():
                raise RuntimeError(
                    "OpenCV Haar cascade files could not be loaded from "
                    f"{directory}. Use an OpenCV 4.x build (OpenCV 5 no longer "
                    "ships them)."
                )
            cached = (face, eye)
            self._local.classifiers = cached
        return cached

    def detect(self, gray: np.ndarray) -> list[FaceBox]:
        """Detect faces and count the eyes found in each.

        Args:
            gray: Upright grayscale image.

        Returns:
            One :class:`FaceBox` per face, in normalized coordinates.
        """
        face_cascade, eye_cascade = self._classifiers()
        height, width = gray.shape[:2]
        equalized = cv2.equalizeHist(gray)
        min_side = max(32, int(self._min_face_fraction * min(height, width)))
        detections = face_cascade.detectMultiScale(
            equalized,
            scaleFactor=self._scale_factor,
            minNeighbors=self._face_min_neighbors,
            minSize=(min_side, min_side),
        )

        faces: list[FaceBox] = []
        for x, y, w, h in detections:
            x, y, w, h = int(x), int(y), int(w), int(h)
            upper_face = equalized[y : y + int(h * 0.6), x : x + w]
            eye_side = max(8, w // 10)
            eyes = eye_cascade.detectMultiScale(
                upper_face,
                scaleFactor=self._scale_factor,
                minNeighbors=self._eye_min_neighbors,
                minSize=(eye_side, eye_side),
                maxSize=(max(eye_side, w // 2), max(eye_side, w // 2)),
            )
            faces.append(
                FaceBox(
                    x=x / width,
                    y=y / height,
                    width=w / width,
                    height=h / height,
                    eyes_detected=min(len(eyes), 2),
                )
            )
        return faces


class OpenCvImageAnalyzer:
    """Measures and scores photos; see the module docstring for the math."""

    def __init__(
        self,
        config: AnalysisConfig | None = None,
        face_detector: FaceDetector | None = None,
        hasher: Hasher = compute_dhash,
    ) -> None:
        """Create an analyzer.

        Args:
            config: Thresholds used for the initial scoring in ``analyze``.
            face_detector: Face/eye detector; defaults to Haar cascades.
            hasher: Perceptual hash function applied to the working image.
        """
        self._config = config or AnalysisConfig()
        self._face_detector = face_detector or HaarFaceDetector()
        self._hasher = hasher

    def analyze(self, root: Path, rel_path: str) -> PhotoItem:
        """Measure one image and return a scored :class:`PhotoItem`.

        Args:
            root: Project folder.
            rel_path: Image location relative to ``root`` (POSIX separators).

        Returns:
            A new photo with measurements, scores and automatic decisions.

        Raises:
            ImageReadError: If the file cannot be decoded.
        """
        path = root / Path(*rel_path.split("/"))
        stat = path.stat()
        stored_width, stored_height, orientation_tag, captured_at = self._read_header(
            path, stat.st_mtime
        )

        image = read_image_bgr(path, long_edge_hint=max(stored_width, stored_height))
        image = apply_exif_orientation(image, orientation_tag)
        working = resize_long_edge(image, WORKING_LONG_EDGE)
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)

        if orientation_tag in (5, 6, 7, 8):
            width, height = stored_height, stored_width
        else:
            width, height = stored_width, stored_height

        mean, contrast, clipped_low, clipped_high = measure_exposure(gray)
        metrics = QualityMetrics(
            laplacian_variance=measure_sharpness(gray),
            mean_luminance=mean,
            rms_contrast=contrast,
            clipped_low=clipped_low,
            clipped_high=clipped_high,
            faces=self._face_detector.detect(gray),
        )
        photo = PhotoItem(
            photo_id=new_id(),
            rel_path=rel_path,
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            captured_at=captured_at,
            width=width,
            height=height,
            orientation=classify_orientation(width, height),
            metrics=metrics,
            dhash=self._hasher(gray),
        )
        self.rescore(photo, self._config)
        return photo

    def rescore(self, photo: PhotoItem, config: AnalysisConfig) -> None:
        """Recompute scores and automatic decisions from stored measurements.

        Resets any previous automatic rejection (including burst duplicates,
        which the deduplicator re-applies afterwards). Scene, rating and
        selection are left alone when the user locked them.

        Args:
            photo: Photo to update in place.
            config: Thresholds to score against.
        """
        metrics = photo.metrics
        metrics.sharpness = min(
            1.0,
            metrics.laplacian_variance
            / (SHARP_REFERENCE_FACTOR * max(config.blur_threshold, 1e-6)),
        )
        metrics.exposure = exposure_score(metrics)
        metrics.resolution = min(1.0, photo.megapixels / max(config.resolution_target_mp, 1e-6))
        if metrics.faces:
            metrics.eyes_open = sum(min(face.eyes_detected, 2) / 2 for face in metrics.faces) / len(
                metrics.faces
            )
            score = (
                WEIGHT_SHARPNESS * metrics.sharpness
                + WEIGHT_EXPOSURE * metrics.exposure
                + WEIGHT_RESOLUTION * metrics.resolution
                + WEIGHT_EYES * metrics.eyes_open
            )
        else:
            metrics.eyes_open = 1.0
            score = (
                WEIGHT_SHARPNESS * metrics.sharpness
                + WEIGHT_EXPOSURE * metrics.exposure
                + WEIGHT_RESOLUTION * metrics.resolution
            ) / (1.0 - WEIGHT_EYES)

        is_blurry = metrics.laplacian_variance < config.blur_threshold
        is_badly_exposed = metrics.exposure < config.exposure_gate
        if is_blurry or is_badly_exposed:
            score = min(score, max(0.0, config.reject_threshold - 0.01))
        metrics.overall_score = score

        if not photo.scene_locked:
            photo.scene = classify_scene(metrics, photo.orientation, config)
        if not photo.rating_locked:
            photo.star_rating = star_rating_for(score)
        if photo.selection_locked:
            return

        reason: str | None = None
        if is_badly_exposed:
            # Checked first: a badly exposed frame has no detail either, and
            # the exposure defect is the more useful explanation.
            problem = describe_exposure_problem(metrics)
            reason = f"{problem} (exposure {metrics.exposure:.2f})"
        elif is_blurry:
            reason = f"Blurry (sharpness {metrics.sharpness:.2f})"
        elif score < config.reject_threshold:
            reason = f"Low overall quality (score {score:.2f})"

        photo.is_selected = reason is None
        photo.rejection_reason = reason
        photo.rejection_category = (
            RejectionCategory.NONE if reason is None else RejectionCategory.QUALITY
        )

    @staticmethod
    def _read_header(path: Path, fallback_mtime: float) -> tuple[int, int, int, datetime]:
        """Read stored dimensions, EXIF orientation and capture time.

        Pillow opens files lazily, so this reads only the header.

        Args:
            path: Image file.
            fallback_mtime: File modification time used when EXIF has no date.

        Returns:
            ``(stored_width, stored_height, orientation_tag, captured_at)``.

        Raises:
            ImageReadError: If Pillow cannot identify the file.
        """
        try:
            with Image.open(path) as image:
                width, height = image.size
                exif = image.getexif()
                orientation_tag = exif.get(_EXIF_ORIENTATION, 1)
                exif_ifd = exif.get_ifd(_EXIF_IFD_POINTER)
                raw_date = (
                    exif_ifd.get(_EXIF_DATETIME_ORIGINAL)
                    or exif_ifd.get(_EXIF_DATETIME_DIGITIZED)
                    or exif.get(_EXIF_DATETIME)
                )
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            raise ImageReadError(f"Unsupported or corrupt image: {exc}") from exc

        if not isinstance(orientation_tag, int) or not 1 <= orientation_tag <= 8:
            orientation_tag = 1
        captured_at = _parse_exif_datetime(raw_date) or datetime.fromtimestamp(fallback_mtime)
        return int(width), int(height), orientation_tag, captured_at


def _parse_exif_datetime(raw: object) -> datetime | None:
    """Parse an EXIF ``YYYY:MM:DD HH:MM:SS`` string; ``None`` if invalid."""
    if isinstance(raw, bytes):
        raw = raw.decode("ascii", errors="ignore")
    if not isinstance(raw, str):
        return None
    text = raw.strip().strip("\x00")
    for pattern in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text[:19], pattern)
        except ValueError:
            continue
    return None
