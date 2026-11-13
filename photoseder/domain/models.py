"""Domain models for PhotoSeder.

Design notes:
    * Every model is a ``@dataclass(slots=True)`` with explicit ``to_dict`` /
      ``from_dict`` methods, so the whole :class:`ProjectState` serializes to
      plain JSON without reflection or third-party libraries.
    * A photo is identified by a stable ``photo_id`` rather than by its path.
      The organizer may move files, so album pages, clusters and the undo
      journal reference the id while ``rel_path`` tracks the current location
      relative to the project root (POSIX separators, Unicode-safe).
    * Raw measurements (Laplacian variance, histogram statistics, face boxes)
      are stored next to the normalized scores. Changing a threshold in the UI
      therefore re-scores the library instantly without re-reading any image.
    * Manual decisions are protected by ``*_locked`` flags so automatic
      re-scoring never overwrites what the user chose.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from photoseder.domain.enums import (
    FileActionMode,
    LayoutTemplate,
    OrganizationStrategy,
    Orientation,
    RejectionCategory,
    SceneType,
)

type JsonDict = dict[str, Any]
type Fingerprint = tuple[int, int]
"""``(size_bytes, mtime_ns)`` — cheap change detector for cached analysis."""

STATE_SCHEMA_VERSION = 1

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
)
"""Lower-case file suffixes treated as photos (HEIC needs ``pillow-heif``)."""

JPEG_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg"})
"""Suffixes that support lossless EXIF injection via ``piexif``."""


def new_id() -> str:
    """Return a short random identifier (12 hex characters)."""
    return uuid.uuid4().hex[:12]


@dataclass(slots=True)
class AnalysisConfig:
    """User-tunable thresholds for scoring, deduplication and layout.

    Attributes:
        blur_threshold: Laplacian variance below which a photo is considered
            blurry. Sharpness is ``min(1, variance / (3 * blur_threshold))``,
            so the threshold itself maps to a sharpness score of 0.33.
        reject_threshold: Overall score below which a photo is rejected.
        exposure_gate: Exposure score below which a photo is rejected even if
            the weighted overall score would pass.
        resolution_target_mp: Megapixels that earn the full resolution score.
        portrait_face_ratio: Minimum total face area / image area for 1-2
            faces to count as a portrait instead of a group shot.
        landscape_min_sharpness: Minimum sharpness for a faceless horizontal
            photo to be classified as a landscape.
        burst_window_seconds: Maximum time gap between consecutive burst shots.
        burst_hamming_distance: Maximum dHash distance between burst shots.
        event_gap_hours: Time gap that starts a new event cluster.
        hero_threshold: Minimum overall score for a ``HERO_SINGLE`` page.
    """

    blur_threshold: float = 100.0
    reject_threshold: float = 0.40
    exposure_gate: float = 0.40
    resolution_target_mp: float = 8.0
    portrait_face_ratio: float = 0.04
    landscape_min_sharpness: float = 0.5
    burst_window_seconds: float = 15.0
    burst_hamming_distance: int = 6
    event_gap_hours: float = 3.0
    hero_threshold: float = 0.82

    def to_dict(self) -> JsonDict:
        """Serialize to a JSON-compatible dictionary."""
        return {name: getattr(self, name) for name in self.__slots__}

    @classmethod
    def from_dict(cls, data: JsonDict) -> AnalysisConfig:
        """Build a config from a dictionary, ignoring unknown keys."""
        return cls(**{k: v for k, v in data.items() if k in cls.__slots__})


@dataclass(slots=True)
class FaceBox:
    """One detected face in normalized image coordinates.

    Attributes:
        x: Left edge as a fraction of image width.
        y: Top edge as a fraction of image height.
        width: Box width as a fraction of image width.
        height: Box height as a fraction of image height.
        eyes_detected: Number of eyes found inside the face (capped at 2).
    """

    x: float
    y: float
    width: float
    height: float
    eyes_detected: int

    @property
    def area_ratio(self) -> float:
        """Face area as a fraction of the image area."""
        return self.width * self.height

    def to_dict(self) -> JsonDict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "width": round(self.width, 4),
            "height": round(self.height, 4),
            "eyes_detected": self.eyes_detected,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> FaceBox:
        """Deserialize from :meth:`to_dict` output."""
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            width=float(data["width"]),
            height=float(data["height"]),
            eyes_detected=int(data["eyes_detected"]),
        )


@dataclass(slots=True)
class QualityMetrics:
    """Raw measurements and the normalized scores derived from them.

    The ``laplacian_variance`` .. ``faces`` fields are *measurements* taken
    from pixels once. The remaining fields are *scores* in ``[0, 1]`` that the
    analyzer recomputes from the measurements whenever the config changes.
    """

    laplacian_variance: float = 0.0
    mean_luminance: float = 0.0
    rms_contrast: float = 0.0
    clipped_low: float = 0.0
    clipped_high: float = 0.0
    faces: list[FaceBox] = field(default_factory=list)

    sharpness: float = 0.0
    exposure: float = 0.0
    resolution: float = 0.0
    eyes_open: float = 1.0
    overall_score: float = 0.0

    @property
    def face_count(self) -> int:
        """Number of detected faces."""
        return len(self.faces)

    @property
    def face_area_ratio(self) -> float:
        """Total face bounding-box area divided by the image area."""
        return sum(face.area_ratio for face in self.faces)

    def to_dict(self) -> JsonDict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "laplacian_variance": round(self.laplacian_variance, 3),
            "mean_luminance": round(self.mean_luminance, 3),
            "rms_contrast": round(self.rms_contrast, 3),
            "clipped_low": round(self.clipped_low, 5),
            "clipped_high": round(self.clipped_high, 5),
            "faces": [face.to_dict() for face in self.faces],
            "sharpness": round(self.sharpness, 4),
            "exposure": round(self.exposure, 4),
            "resolution": round(self.resolution, 4),
            "eyes_open": round(self.eyes_open, 4),
            "overall_score": round(self.overall_score, 4),
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> QualityMetrics:
        """Deserialize from :meth:`to_dict` output."""
        return cls(
            laplacian_variance=float(data.get("laplacian_variance", 0.0)),
            mean_luminance=float(data.get("mean_luminance", 0.0)),
            rms_contrast=float(data.get("rms_contrast", 0.0)),
            clipped_low=float(data.get("clipped_low", 0.0)),
            clipped_high=float(data.get("clipped_high", 0.0)),
            faces=[FaceBox.from_dict(f) for f in data.get("faces", [])],
            sharpness=float(data.get("sharpness", 0.0)),
            exposure=float(data.get("exposure", 0.0)),
            resolution=float(data.get("resolution", 0.0)),
            eyes_open=float(data.get("eyes_open", 1.0)),
            overall_score=float(data.get("overall_score", 0.0)),
        )


@dataclass(slots=True)
class PhotoItem:
    """One analyzed photo and every decision made about it.

    Attributes:
        photo_id: Stable identifier that survives file moves.
        rel_path: Current location relative to the project root (POSIX form).
        size_bytes: File size at analysis time (fingerprint part 1).
        mtime_ns: File modification time in ns (fingerprint part 2).
        captured_at: EXIF ``DateTimeOriginal`` or the file mtime as fallback.
        width: Display width in pixels after EXIF orientation.
        height: Display height in pixels after EXIF orientation.
        orientation: Horizontal, vertical or square.
        metrics: Raw measurements and normalized scores.
        dhash: 64-bit difference hash used for burst detection.
        scene: Effective scene type (automatic unless ``scene_locked``).
        star_rating: Effective rating 1-5 (automatic unless ``rating_locked``).
        is_selected: ``True`` when the photo is a keeper.
        rejection_category: Why the photo is not a keeper.
        rejection_reason: Human-readable explanation of the rejection.
        burst_id: Identifier shared by all members of one burst, if any.
        cluster_id: Event cluster the photo belongs to.
        scene_locked: The user overrode the scene type.
        rating_locked: The user overrode the star rating.
        selection_locked: The user overrode keep/reject.
    """

    photo_id: str
    rel_path: str
    size_bytes: int
    mtime_ns: int
    captured_at: datetime
    width: int
    height: int
    orientation: Orientation
    metrics: QualityMetrics
    dhash: int = 0
    scene: SceneType = SceneType.GENERAL
    star_rating: int = 1
    is_selected: bool = True
    rejection_category: RejectionCategory = RejectionCategory.NONE
    rejection_reason: str | None = None
    burst_id: str | None = None
    cluster_id: str | None = None
    scene_locked: bool = False
    rating_locked: bool = False
    selection_locked: bool = False

    @property
    def fingerprint(self) -> Fingerprint:
        """``(size_bytes, mtime_ns)`` used to detect modified files."""
        return (self.size_bytes, self.mtime_ns)

    @property
    def name(self) -> str:
        """File name without directories."""
        return self.rel_path.rsplit("/", 1)[-1]

    @property
    def aspect_ratio(self) -> float:
        """Display width divided by display height."""
        return self.width / self.height if self.height else 1.0

    @property
    def megapixels(self) -> float:
        """Resolution in megapixels."""
        return self.width * self.height / 1_000_000

    def absolute_path(self, root: Path) -> Path:
        """Resolve the photo's current location under ``root``."""
        return root / Path(*self.rel_path.split("/"))

    def to_dict(self) -> JsonDict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "photo_id": self.photo_id,
            "rel_path": self.rel_path,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "captured_at": self.captured_at.isoformat(),
            "width": self.width,
            "height": self.height,
            "orientation": self.orientation.value,
            "metrics": self.metrics.to_dict(),
            "dhash": f"{self.dhash:016x}",
            "scene": self.scene.value,
            "star_rating": self.star_rating,
            "is_selected": self.is_selected,
            "rejection_category": self.rejection_category.value,
            "rejection_reason": self.rejection_reason,
            "burst_id": self.burst_id,
            "cluster_id": self.cluster_id,
            "scene_locked": self.scene_locked,
            "rating_locked": self.rating_locked,
            "selection_locked": self.selection_locked,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> PhotoItem:
        """Deserialize from :meth:`to_dict` output."""
        return cls(
            photo_id=str(data["photo_id"]),
            rel_path=str(data["rel_path"]),
            size_bytes=int(data["size_bytes"]),
            mtime_ns=int(data["mtime_ns"]),
            captured_at=datetime.fromisoformat(data["captured_at"]),
            width=int(data["width"]),
            height=int(data["height"]),
            orientation=Orientation(data["orientation"]),
            metrics=QualityMetrics.from_dict(data["metrics"]),
            dhash=int(data.get("dhash", "0"), 16),
            scene=SceneType(data.get("scene", SceneType.GENERAL.value)),
            star_rating=int(data.get("star_rating", 1)),
            is_selected=bool(data.get("is_selected", True)),
            rejection_category=RejectionCategory(
                data.get("rejection_category", RejectionCategory.NONE.value)
            ),
            rejection_reason=data.get("rejection_reason"),
            burst_id=data.get("burst_id"),
            cluster_id=data.get("cluster_id"),
            scene_locked=bool(data.get("scene_locked", False)),
            rating_locked=bool(data.get("rating_locked", False)),
            selection_locked=bool(data.get("selection_locked", False)),
        )


@dataclass(slots=True)
class EventCluster:
    """A chronological chapter of photos (one event or part of a day)."""

    cluster_id: str
    title: str
    start: datetime
    end: datetime
    photo_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "title": self.title,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "photo_ids": list(self.photo_ids),
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> EventCluster:
        """Deserialize from :meth:`to_dict` output."""
        return cls(
            cluster_id=str(data["cluster_id"]),
            title=str(data["title"]),
            start=datetime.fromisoformat(data["start"]),
            end=datetime.fromisoformat(data["end"]),
            photo_ids=[str(p) for p in data.get("photo_ids", [])],
        )


@dataclass(slots=True)
class AlbumPage:
    """One album page: an ordered list of photos and a layout template.

    Attributes:
        page_id: Stable identifier for UI widgets.
        template: CSS Grid layout used to render the page.
        photo_ids: Photos in slot order (slot 0 is the dominant slot).
        cluster_id: Chapter the page was generated from, if any.
        chapter_title: Heading printed on the first page of a chapter.
        template_locked: The user picked the template; automatic
            recalculation keeps it while it can still hold the photos.
    """

    page_id: str
    template: LayoutTemplate
    photo_ids: list[str] = field(default_factory=list)
    cluster_id: str | None = None
    chapter_title: str | None = None
    template_locked: bool = False

    def to_dict(self) -> JsonDict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "page_id": self.page_id,
            "template": self.template.value,
            "photo_ids": list(self.photo_ids),
            "cluster_id": self.cluster_id,
            "chapter_title": self.chapter_title,
            "template_locked": self.template_locked,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> AlbumPage:
        """Deserialize from :meth:`to_dict` output."""
        return cls(
            page_id=str(data["page_id"]),
            template=LayoutTemplate(data["template"]),
            photo_ids=[str(p) for p in data.get("photo_ids", [])],
            cluster_id=data.get("cluster_id"),
            chapter_title=data.get("chapter_title"),
            template_locked=bool(data.get("template_locked", False)),
        )


@dataclass(slots=True)
class FileTransaction:
    """One planned or executed file operation.

    Attributes:
        photo_id: Photo the operation applies to.
        source: Absolute source path.
        destination: Absolute destination path.
        mode: Move, copy or hardlink.
        executed: ``True`` once the operation succeeded on disk.
        note: Planner or executor remark (e.g. a fallback or an error).
        dest_inode: Inode of the destination right after execution; undo only
            deletes a copy/hardlink that is still this exact file.
        dest_size: Size of the destination right after execution.
    """

    photo_id: str
    source: str
    destination: str
    mode: FileActionMode
    executed: bool = False
    note: str = ""
    dest_inode: int = 0
    dest_size: int = 0

    def to_dict(self) -> JsonDict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "photo_id": self.photo_id,
            "source": self.source,
            "destination": self.destination,
            "mode": self.mode.value,
            "executed": self.executed,
            "note": self.note,
            "dest_inode": self.dest_inode,
            "dest_size": self.dest_size,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> FileTransaction:
        """Deserialize from :meth:`to_dict` output."""
        return cls(
            photo_id=str(data["photo_id"]),
            source=str(data["source"]),
            destination=str(data["destination"]),
            mode=FileActionMode(data["mode"]),
            executed=bool(data.get("executed", False)),
            note=str(data.get("note", "")),
            dest_inode=int(data.get("dest_inode", 0)),
            dest_size=int(data.get("dest_size", 0)),
        )


@dataclass(slots=True)
class OperationBatch:
    """A journaled group of file transactions that is undone as one unit.

    Attributes:
        batch_id: Identifier of the batch.
        created_at: When the batch was planned.
        strategy: Organization strategy that produced the batch.
        mode: Requested file action mode.
        transactions: Operations in execution order.
        created_dirs: Directories created by the batch (absolute paths), so
            undo can remove them again when they are empty.
        dry_run: ``True`` when nothing was written to disk.
    """

    batch_id: str
    created_at: datetime
    strategy: OrganizationStrategy
    mode: FileActionMode
    transactions: list[FileTransaction] = field(default_factory=list)
    created_dirs: list[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def executed_count(self) -> int:
        """Number of transactions that succeeded on disk."""
        return sum(1 for t in self.transactions if t.executed)

    def to_dict(self) -> JsonDict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "batch_id": self.batch_id,
            "created_at": self.created_at.isoformat(),
            "strategy": self.strategy.value,
            "mode": self.mode.value,
            "transactions": [t.to_dict() for t in self.transactions],
            "created_dirs": list(self.created_dirs),
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> OperationBatch:
        """Deserialize from :meth:`to_dict` output."""
        return cls(
            batch_id=str(data["batch_id"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            strategy=OrganizationStrategy(data["strategy"]),
            mode=FileActionMode(data["mode"]),
            transactions=[FileTransaction.from_dict(t) for t in data.get("transactions", [])],
            created_dirs=[str(d) for d in data.get("created_dirs", [])],
        )


@dataclass(slots=True)
class ProjectState:
    """Everything PhotoSeder knows about one folder; persisted as JSON.

    Attributes:
        root: Absolute project folder (not serialized; implied by location).
        config: Thresholds the scores were computed with.
        photos: Analyzed photos keyed by ``photo_id``.
        clusters: Event clusters in chronological order.
        pages: Album pages in print order.
        unreadable: Files that could not be decoded, ``rel_path -> reason``.
        updated_at: Last time the state was saved.
    """

    root: Path
    config: AnalysisConfig = field(default_factory=AnalysisConfig)
    photos: dict[str, PhotoItem] = field(default_factory=dict)
    clusters: list[EventCluster] = field(default_factory=list)
    pages: list[AlbumPage] = field(default_factory=list)
    unreadable: dict[str, str] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=datetime.now)

    def photos_by_path(self) -> dict[str, PhotoItem]:
        """Index photos by their current relative path."""
        return {photo.rel_path: photo for photo in self.photos.values()}

    def chronological(self) -> list[PhotoItem]:
        """All photos sorted by capture time, then path (deterministic)."""
        return sorted(self.photos.values(), key=lambda p: (p.captured_at, p.rel_path))

    def cluster_by_id(self, cluster_id: str | None) -> EventCluster | None:
        """Return the cluster with the given id, if it exists."""
        return next((c for c in self.clusters if c.cluster_id == cluster_id), None)

    def to_dict(self) -> JsonDict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "updated_at": self.updated_at.isoformat(),
            "config": self.config.to_dict(),
            "photos": [photo.to_dict() for photo in self.photos.values()],
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "pages": [page.to_dict() for page in self.pages],
            "unreadable": dict(self.unreadable),
        }

    @classmethod
    def from_dict(cls, root: Path, data: JsonDict) -> ProjectState:
        """Deserialize from :meth:`to_dict` output.

        Raises:
            ValueError: If the schema version is not supported.
        """
        version = data.get("schema_version")
        if version != STATE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported state schema version: {version!r}")
        photos = [PhotoItem.from_dict(p) for p in data.get("photos", [])]
        return cls(
            root=root,
            config=AnalysisConfig.from_dict(data.get("config", {})),
            photos={photo.photo_id: photo for photo in photos},
            clusters=[EventCluster.from_dict(c) for c in data.get("clusters", [])],
            pages=[AlbumPage.from_dict(p) for p in data.get("pages", [])],
            unreadable={str(k): str(v) for k, v in data.get("unreadable", {}).items()},
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )
