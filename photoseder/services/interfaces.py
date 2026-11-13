"""Service contracts (structural ``Protocol`` types).

The orchestrator depends only on these protocols, never on concrete classes
(dependency inversion). Any implementation with matching method signatures can
be injected into :class:`photoseder.orchestrator.pipeline.Pipeline` — for
example a YuNet-based ``FaceDetector`` or a PDF ``Exporter`` — without changing
the pipeline or the UI.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from photoseder.domain.enums import (
    FileActionMode,
    LayoutTemplate,
    OrganizationStrategy,
    PageSize,
)
from photoseder.domain.models import (
    AlbumPage,
    AnalysisConfig,
    EventCluster,
    FaceBox,
    FileTransaction,
    Fingerprint,
    OperationBatch,
    PhotoItem,
    ProjectState,
)

type ProgressCallback = Callable[[int, int, str], None]
"""``(done, total, current_item)`` reporter used by long-running operations."""


class MetadataStatus(StrEnum):
    """Outcome of writing metadata into one file."""

    WRITTEN = "written"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(slots=True)
class MetadataWriteResult:
    """Result of one metadata write attempt."""

    status: MetadataStatus
    message: str = ""


@dataclass(slots=True)
class ScanDiff:
    """Difference between the cached state and the files currently on disk.

    Attributes:
        new: Relative paths that were never analyzed.
        modified: Relative paths whose fingerprint changed.
        removed: Photo ids whose file no longer exists anywhere.
        moved: ``photo_id -> new relative path`` for files that were moved or
            renamed outside PhotoSeder (same name and fingerprint).
        unchanged: Number of files served from the cache.
    """

    new: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    moved: dict[str, str] = field(default_factory=dict)
    unchanged: int = 0


@dataclass(slots=True)
class ExportOptions:
    """Rendering options for the album exporter.

    Attributes:
        title: Album title printed on the cover page.
        page_size: Print format.
        embed_images: Embed photos as base64 (self-contained file) instead of
            linking to the original files by relative path.
        crop_to_fill: Fill each grid cell (``object-fit: cover``) instead of
            letterboxing the whole photo (``contain``).
        max_edge_px: Long edge of embedded images in pixels.
        include_cover: Render a cover page before the first album page.
    """

    title: str = "PhotoSeder Album"
    page_size: PageSize = PageSize.A4_LANDSCAPE
    embed_images: bool = True
    crop_to_fill: bool = True
    max_edge_px: int = 2400
    include_cover: bool = True


@runtime_checkable
class FaceDetector(Protocol):
    """Finds faces (and eyes) in an upright grayscale image."""

    def detect(self, gray: np.ndarray) -> list[FaceBox]:
        """Return detected faces in normalized image coordinates."""
        ...


@runtime_checkable
class Analyzer(Protocol):
    """Measures image quality and classifies photos."""

    def analyze(self, root: Path, rel_path: str) -> PhotoItem:
        """Read one image and return a fully measured and scored photo."""
        ...

    def rescore(self, photo: PhotoItem, config: AnalysisConfig) -> None:
        """Recompute scores and automatic decisions from stored measurements."""
        ...


@runtime_checkable
class Deduplicator(Protocol):
    """Detects burst duplicates and partitions photos into events."""

    def mark_bursts(self, photos: Sequence[PhotoItem], config: AnalysisConfig) -> int:
        """Reject all but the best photo of each burst; return rejected count."""
        ...

    def cluster_events(
        self, photos: Sequence[PhotoItem], config: AnalysisConfig
    ) -> list[EventCluster]:
        """Partition photos into chronological event clusters."""
        ...


@runtime_checkable
class Organizer(Protocol):
    """Performs reversible physical file organization."""

    def plan(
        self,
        photos: Sequence[PhotoItem],
        clusters: Sequence[EventCluster],
        strategy: OrganizationStrategy,
        mode: FileActionMode,
    ) -> list[FileTransaction]:
        """Compute the file operations for a strategy without touching disk."""
        ...

    def organize(
        self,
        photos: Sequence[PhotoItem],
        clusters: Sequence[EventCluster],
        strategy: OrganizationStrategy,
        mode: FileActionMode,
        *,
        dry_run: bool = False,
    ) -> OperationBatch:
        """Plan and (unless ``dry_run``) execute and journal a batch."""
        ...

    def undo_last_operation(self) -> OperationBatch | None:
        """Revert the most recent journaled batch, if any."""
        ...

    def can_undo(self) -> bool:
        """Return ``True`` when the journal holds a batch to undo."""
        ...

    def linked_copies(self) -> set[str]:
        """Absolute paths of copies/hardlinks created by journaled batches."""
        ...


@runtime_checkable
class MetadataWriter(Protocol):
    """Writes ratings and keywords into image files."""

    def write(self, path: Path, photo: PhotoItem, cluster_title: str | None) -> MetadataWriteResult:
        """Write rating and keywords for one photo."""
        ...


@runtime_checkable
class LayoutPlanner(Protocol):
    """Arranges photos into album pages."""

    def plan_pages(
        self,
        photos: Sequence[PhotoItem],
        config: AnalysisConfig,
        *,
        max_photos_per_page: int,
        cluster: EventCluster | None = None,
    ) -> list[AlbumPage]:
        """Lay out a chronological photo sequence as album pages."""
        ...

    def choose_template(self, photos: Sequence[PhotoItem]) -> LayoutTemplate:
        """Pick the best template for the photos currently on one page."""
        ...

    def recalculate_page(self, page: AlbumPage, photos_by_id: dict[str, PhotoItem]) -> None:
        """Re-pick a page's template after its photos changed."""
        ...


@runtime_checkable
class Exporter(Protocol):
    """Renders album pages to a printable document."""

    def render(
        self,
        pages: Sequence[AlbumPage],
        photos_by_id: dict[str, PhotoItem],
        root: Path,
        options: ExportOptions,
        *,
        output_dir: Path | None = None,
    ) -> str:
        """Return the album as an HTML document string."""
        ...

    def render_preview(
        self,
        page: AlbumPage,
        photos_by_id: dict[str, PhotoItem],
        options: ExportOptions,
        image_source: Callable[[PhotoItem], str],
    ) -> str:
        """Return one page as an HTML document sized to fit its viewport."""
        ...

    def export(
        self,
        pages: Sequence[AlbumPage],
        photos_by_id: dict[str, PhotoItem],
        root: Path,
        options: ExportOptions,
        output_path: Path,
    ) -> Path:
        """Write the album to ``output_path`` and return it."""
        ...


@runtime_checkable
class StateStore(Protocol):
    """Persists :class:`ProjectState` and detects changed files."""

    def load(self, root: Path) -> ProjectState | None:
        """Load the cached state for a folder, if a valid one exists."""
        ...

    def save(self, state: ProjectState) -> None:
        """Persist the state atomically."""
        ...

    def discover(self, root: Path, excluded: set[str]) -> dict[str, Fingerprint]:
        """Find image files under ``root`` as ``rel_path -> fingerprint``."""
        ...

    def diff(self, state: ProjectState, discovered: dict[str, Fingerprint]) -> ScanDiff:
        """Compare cached photos with the files found on disk."""
        ...
