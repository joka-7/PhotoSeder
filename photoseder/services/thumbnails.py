"""Upright image loading, JPEG encoding and an on-disk thumbnail cache.

The Streamlit gallery would be unusably slow if it decoded full-resolution
photos on every rerun. :class:`ThumbnailCache` stores one small JPEG per photo
in ``<folder>/.photoseder_cache/thumbs/``. The cache key combines the stable
``photo_id`` with the file's modification time, so a thumbnail survives file
moves and is rebuilt only when the photo itself changes.

JPEG sources are decoded with Pillow's ``draft`` mode, which asks the decoder
for a 1/2, 1/4 or 1/8 scale image directly from the DCT coefficients — far
faster than decoding at full size and shrinking afterwards.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageOps

from photoseder.domain.models import PhotoItem

CACHE_DIRNAME = ".photoseder_cache"
THUMBNAIL_EDGE = 512


def load_upright_rgb(path: Path, max_edge: int) -> Image.Image:
    """Load an image as upright RGB, downscaled to ``max_edge``.

    Args:
        path: Image file (Unicode paths are supported by Pillow natively).
        max_edge: Maximum width and height of the result in pixels.

    Returns:
        An RGB image with the EXIF orientation applied.

    Raises:
        OSError: If the file cannot be read or decoded.
    """
    with Image.open(path) as image:
        image.draft("RGB", (max_edge, max_edge))
        upright = ImageOps.exif_transpose(image)
        upright.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        return upright.convert("RGB")


def encode_jpeg(image: Image.Image, quality: int) -> bytes:
    """Encode an RGB image as JPEG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def jpeg_data_uri(data: bytes) -> str:
    """Wrap JPEG bytes in a ``data:`` URI for embedding in HTML."""
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


class ThumbnailCache:
    """Creates and serves cached thumbnails for one project folder."""

    def __init__(self, root: Path, edge: int = THUMBNAIL_EDGE) -> None:
        """Bind the cache to a project folder.

        Args:
            root: Project folder.
            edge: Long edge of generated thumbnails in pixels.
        """
        self._root = root
        self._edge = edge
        self._directory = root / CACHE_DIRNAME / "thumbs"

    def path_for(self, photo: PhotoItem) -> Path | None:
        """Return the thumbnail file for a photo, creating it if needed.

        Args:
            photo: Photo to render.

        Returns:
            Path of the cached JPEG, or ``None`` if the source is unreadable.
        """
        target = self._directory / f"{photo.photo_id}_{photo.mtime_ns}.jpg"
        if target.is_file():
            return target
        try:
            image = load_upright_rgb(photo.absolute_path(self._root), self._edge)
            self._directory.mkdir(parents=True, exist_ok=True)
            for stale in self._directory.glob(f"{photo.photo_id}_*.jpg"):
                stale.unlink(missing_ok=True)
            target.write_bytes(encode_jpeg(image, quality=82))
        except (OSError, ValueError):
            return None
        return target

    def data_uri(self, photo: PhotoItem) -> str:
        """Return the thumbnail as a ``data:`` URI (empty if unreadable)."""
        path = self.path_for(photo)
        if path is None:
            return ""
        try:
            return jpeg_data_uri(path.read_bytes())
        except OSError:
            return ""
