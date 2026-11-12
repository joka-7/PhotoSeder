"""Lossless EXIF rating and keyword injection for JPEG files (``piexif``).

How it stays lossless:
    A JPEG file is a sequence of segments. EXIF metadata lives in the APP1
    segment, separate from the entropy-coded pixel data. ``piexif.insert``
    replaces only that segment and copies every other byte verbatim, so the
    image is never decoded or re-encoded and no quality is lost.

Tags written (all in the 0th IFD):
    * ``Rating`` (0x4746): 1-5 stars.
    * ``RatingPercent`` (0x4749): the Windows convention 1, 25, 50, 75, 99.
    * ``XPKeywords`` (0x9C9E): ``;``-separated keywords encoded as UTF-16LE
      with a terminating NUL, the encoding Windows Explorer expects and Linux
      tools (exiftool, digiKam) read.
    * ``ImageDescription`` (0x010E): a ``[PhotoSeder: ...]`` summary. An
      existing caption is preserved after the summary, and re-tagging replaces
      only the bracketed part.

Write safety:
    The new file is assembled in memory, written to a temporary file in the
    same directory and swapped in with ``os.replace``; an interrupted write
    can never leave a truncated original. The original access/modification
    times are restored afterwards.
"""

from __future__ import annotations

import io
import os
import re
from pathlib import Path

import piexif

from photoseder.domain.models import JPEG_EXTENSIONS, PhotoItem
from photoseder.services.interfaces import MetadataStatus, MetadataWriteResult

RATING_PERCENT: dict[int, int] = {1: 1, 2: 25, 3: 50, 4: 75, 5: 99}
QUALITY_TIERS: dict[int, str] = {
    5: "Hero_5Star",
    4: "Great_4Star",
    3: "Good_3Star",
    2: "Fair_2Star",
    1: "Weak_1Star",
}

_DESCRIPTION_PATTERN = re.compile(r"^\[PhotoSeder: [^\]]*\]\s*")
# EXIF tags whose values some cameras store with the wrong type, which makes
# ``piexif.dump`` fail. They carry no information PhotoSeder needs.
_FRAGILE_EXIF_TAGS = (
    piexif.ExifIFD.SceneType,
    piexif.ExifIFD.FileSource,
    piexif.ExifIFD.MakerNote,
)


def build_keywords(photo: PhotoItem, cluster_title: str | None) -> list[str]:
    """Return the keywords for a photo: scene, quality tier and event."""
    keywords = [photo.scene.label, QUALITY_TIERS[photo.star_rating]]
    if cluster_title:
        keywords.append(cluster_title)
    return keywords


def encode_xp_keywords(keywords: list[str]) -> bytes:
    """Encode keywords as a NUL-terminated UTF-16LE ``XPKeywords`` value."""
    return ";".join(keywords).encode("utf-16-le") + b"\x00\x00"


def decode_xp_keywords(raw: bytes | tuple[int, ...]) -> list[str]:
    """Decode an ``XPKeywords`` value back into a keyword list."""
    text = bytes(raw).decode("utf-16-le", errors="ignore").rstrip("\x00")
    return [keyword for keyword in text.split(";") if keyword]


class ExifMetadataWriter:
    """Writes star ratings and scene keywords into JPEG EXIF, losslessly."""

    def write(self, path: Path, photo: PhotoItem, cluster_title: str | None) -> MetadataWriteResult:
        """Write rating and keywords for one photo.

        Args:
            path: Absolute path of the image file.
            photo: Source of the rating and scene type.
            cluster_title: Event name added as a keyword, if any.

        Returns:
            ``WRITTEN`` on success, ``SKIPPED`` for non-JPEG files and
            ``FAILED`` (with the reason) when the file cannot be updated.
        """
        if path.suffix.lower() not in JPEG_EXTENSIONS:
            return MetadataWriteResult(
                MetadataStatus.SKIPPED, f"{path.suffix or 'file'}: EXIF needs JPEG"
            )
        try:
            original = path.read_bytes()
            stat = path.stat()
            exif_bytes = self._build_exif(original, photo, cluster_title)
            output = io.BytesIO()
            piexif.insert(exif_bytes, original, output)

            temporary = path.with_name(f".{path.name}.photoseder_tmp")
            try:
                temporary.write_bytes(output.getvalue())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        except (OSError, ValueError, KeyError, TypeError, piexif.InvalidImageDataError) as exc:
            return MetadataWriteResult(MetadataStatus.FAILED, str(exc))
        return MetadataWriteResult(MetadataStatus.WRITTEN)

    def _build_exif(self, jpeg_bytes: bytes, photo: PhotoItem, cluster_title: str | None) -> bytes:
        """Merge PhotoSeder's tags into the file's existing EXIF block."""
        try:
            exif = piexif.load(jpeg_bytes)
        except (ValueError, KeyError, TypeError, piexif.InvalidImageDataError):
            exif = {}
        for ifd in ("0th", "Exif", "GPS", "1st", "Interop"):
            exif.setdefault(ifd, {})
        exif.setdefault("thumbnail", None)

        keywords = build_keywords(photo, cluster_title)
        zeroth = exif["0th"]
        zeroth[piexif.ImageIFD.Rating] = photo.star_rating
        zeroth[piexif.ImageIFD.RatingPercent] = RATING_PERCENT[photo.star_rating]
        zeroth[piexif.ImageIFD.XPKeywords] = encode_xp_keywords(keywords)
        zeroth[piexif.ImageIFD.ImageDescription] = self._description(
            zeroth.get(piexif.ImageIFD.ImageDescription), photo, cluster_title
        )

        try:
            return piexif.dump(exif)
        except (ValueError, KeyError, TypeError):
            # Some cameras write malformed tags; drop the known offenders and
            # the embedded thumbnail, then retry once.
            for tag in _FRAGILE_EXIF_TAGS:
                exif["Exif"].pop(tag, None)
            exif["1st"] = {}
            exif["thumbnail"] = None
            return piexif.dump(exif)

    @staticmethod
    def _description(
        existing: bytes | str | None, photo: PhotoItem, cluster_title: str | None
    ) -> bytes:
        """Build ``ImageDescription``, preserving any user caption."""
        if isinstance(existing, bytes):
            existing = existing.decode("utf-8", errors="replace")
        caption = _DESCRIPTION_PATTERN.sub("", (existing or "").strip("\x00 "))
        parts = [
            f"Scene={photo.scene.label}",
            f"Tier={QUALITY_TIERS[photo.star_rating]}",
        ]
        if cluster_title:
            parts.append(f"Event={cluster_title.replace(']', ')')}")
        summary = f"[PhotoSeder: {'; '.join(parts)}]"
        return f"{summary} {caption}".strip().encode("utf-8")
