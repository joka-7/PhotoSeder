"""Synthetic photo fixtures: deterministic images with controlled defects."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import piexif
import pytest
from PIL import Image

type PhotoFactory = Callable[..., Path]


def synth_image(seed: int, width: int = 1600, height: int = 1067) -> np.ndarray:
    """Build a textured RGB image: smooth colour blobs plus crisp detail."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(40, 215, size=(5, 6, 3), dtype=np.uint8)
    base = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    detail = rng.integers(-45, 45, size=(height // 4, width // 4, 1)).astype(np.float32)
    detail = cv2.resize(detail, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.clip(base.astype(np.float32) + detail[..., None], 0, 255).astype(np.uint8)


def save_jpeg(
    path: Path,
    pixels: np.ndarray,
    taken: datetime | None = None,
    orientation: int | None = None,
) -> Path:
    """Save RGB pixels as JPEG with optional EXIF date and orientation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exif: dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    if taken is not None:
        exif["Exif"][piexif.ExifIFD.DateTimeOriginal] = taken.strftime("%Y:%m:%d %H:%M:%S")
    if orientation is not None:
        exif["0th"][piexif.ImageIFD.Orientation] = orientation
    Image.fromarray(pixels).save(path, quality=92, exif=piexif.dump(exif))
    return path


@pytest.fixture
def make_photo(tmp_path: Path) -> PhotoFactory:
    """Factory writing a synthetic JPEG into the temporary project folder."""

    def factory(
        name: str,
        *,
        seed: int = 1,
        taken: datetime | None = None,
        blur: float = 0.0,
        gain: float = 1.0,
        offset: float = 0.0,
        size: tuple[int, int] = (1600, 1067),
        orientation: int | None = None,
    ) -> Path:
        pixels = synth_image(seed, *size)
        if blur:
            pixels = cv2.GaussianBlur(pixels, (0, 0), blur)
        if gain != 1.0 or offset:
            pixels = np.clip(pixels.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)
        return save_jpeg(tmp_path / name, pixels, taken, orientation)

    return factory


@pytest.fixture
def library(tmp_path: Path, make_photo: PhotoFactory) -> Path:
    """A small two-day library with every defect type, in a Hebrew folder."""
    day1 = datetime(2026, 10, 15, 9, 0, 0)
    make_photo("תמונות/a_sharp.jpg", seed=1, taken=day1)
    make_photo("תמונות/b_blurry.jpg", seed=2, taken=day1.replace(minute=5), blur=9)
    make_photo("תמונות/c_dark.jpg", seed=3, taken=day1.replace(minute=10), gain=0.06)
    make_photo("תמונות/d_bright.jpg", seed=4, taken=day1.replace(minute=15), gain=0.1, offset=245)
    # Burst: same scene two seconds apart, the second slightly soft.
    make_photo("תמונות/e_burst1.jpg", seed=5, taken=day1.replace(minute=20, second=0))
    make_photo("תמונות/e_burst2.jpg", seed=5, taken=day1.replace(minute=20, second=2), blur=0.6)
    # Same day, after a long gap -> new event.
    make_photo("תמונות/f_evening.jpg", seed=6, taken=day1.replace(hour=19))
    make_photo(
        "תמונות/g_vertical.jpg", seed=7, taken=day1.replace(hour=19, minute=3), size=(1067, 1600)
    )
    make_photo(
        "תמונות/h_vertical.jpg", seed=8, taken=day1.replace(hour=19, minute=6), size=(1067, 1600)
    )
    # Next day.
    for index in range(5):
        make_photo(
            f"תמונות/day2_{index}.jpg",
            seed=20 + index,
            taken=datetime(2026, 10, 16, 11, index * 4, 0),
        )
    return tmp_path / "תמונות"
