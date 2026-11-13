"""High-level coordinator for the PhotoSeder workflows.

:class:`Pipeline` is the only object the UI talks to. It owns the
:class:`ProjectState` for one folder and wires the services together through
the protocols in :mod:`photoseder.services.interfaces` (constructor
injection), so each service can be replaced or mocked independently.

Workflows:
    * **Step by step** — ``scan_and_analyze`` -> manual curation
      (``set_selected`` / ``set_scene`` / ``set_rating``) ->
      ``preview_organization`` / ``execute_organization`` /
      ``undo_last_organization`` -> ``write_metadata`` ->
      ``generate_chapter`` per event -> page editing -> ``export_album``.
    * **Auto-Pilot** — :meth:`Pipeline.run_autopilot` chains scan, curation,
      album generation and export in one call.

Consistency rules enforced here:
    * ``curate`` always runs in the order *rescore -> burst culling -> event
      clustering -> album sync*, which makes it idempotent.
    * A photo that stops being a keeper is removed from the album; emptied
      pages are dropped and affected templates recalculated.
    * After the organizer moves files (or an undo moves them back) and after
      the EXIF writer changes a file, the cached path / fingerprint is
      updated so the next scan does not re-analyze anything.
    * Every mutating method saves the state.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from photoseder.domain.enums import (
    FileActionMode,
    LayoutTemplate,
    OrganizationStrategy,
    RejectionCategory,
    SceneType,
)
from photoseder.domain.models import (
    AlbumPage,
    AnalysisConfig,
    EventCluster,
    FileTransaction,
    OperationBatch,
    PhotoItem,
    ProjectState,
    new_id,
)
from photoseder.services.analyzer import OpenCvImageAnalyzer
from photoseder.services.deduplicator import BurstDeduplicator
from photoseder.services.exporter import HtmlAlbumExporter
from photoseder.services.interfaces import (
    Analyzer,
    Deduplicator,
    Exporter,
    ExportOptions,
    LayoutPlanner,
    MetadataStatus,
    MetadataWriter,
    Organizer,
    ProgressCallback,
    StateStore,
)
from photoseder.services.layout_planner import MAX_PHOTOS_PER_PAGE, SmartLayoutPlanner
from photoseder.services.metadata_writer import ExifMetadataWriter
from photoseder.services.organizer import FolderOrganizer
from photoseder.services.state_manager import JsonStateStore

DEFAULT_ALBUM_FILENAME = "PhotoSeder_Album.html"
MANUAL_REJECTION_REASON = "Rejected manually"


@dataclass(slots=True)
class ScanReport:
    """Outcome of one scan.

    Attributes:
        analyzed: Files that were (re)analyzed in this scan.
        cached: Files served from the cache without touching pixels.
        moved: Files re-linked after being moved outside PhotoSeder.
        removed: Cached photos whose file disappeared.
        errors: ``rel_path -> reason`` for files that could not be decoded.
    """

    analyzed: int = 0
    cached: int = 0
    moved: int = 0
    removed: int = 0
    errors: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class MetadataReport:
    """Outcome of writing EXIF metadata for the whole project."""

    written: int = 0
    skipped: int = 0
    failed: int = 0
    failures: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class DashboardStats:
    """Numbers shown on the curation dashboard."""

    total: int = 0
    keepers: int = 0
    rejected_quality: int = 0
    rejected_duplicates: int = 0
    rejected_manual: int = 0
    heroes: int = 0
    scenes: dict[SceneType, int] = field(default_factory=dict)


class Pipeline:
    """Coordinates analysis, curation, organization and album building."""

    def __init__(
        self,
        root: Path,
        *,
        analyzer: Analyzer | None = None,
        deduplicator: Deduplicator | None = None,
        organizer: Organizer | None = None,
        metadata_writer: MetadataWriter | None = None,
        layout_planner: LayoutPlanner | None = None,
        exporter: Exporter | None = None,
        state_store: StateStore | None = None,
        max_workers: int | None = None,
    ) -> None:
        """Open a project folder, loading its cached state if present.

        Args:
            root: Folder containing the photos.
            analyzer: Image analyzer (default: OpenCV + Haar cascades).
            deduplicator: Burst and event detector (default: dHash).
            organizer: File organizer (default: journaled folder organizer).
            metadata_writer: EXIF writer (default: piexif).
            layout_planner: Page layout engine.
            exporter: Album renderer (default: HTML / CSS Grid).
            state_store: State persistence (default: JSON in the folder).
            max_workers: Analysis threads; defaults to the CPU count (max 8).

        Raises:
            NotADirectoryError: If ``root`` is not an existing directory.
        """
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Not a folder: {root}")
        self._root = root
        self._analyzer: Analyzer = analyzer or OpenCvImageAnalyzer()
        self._deduplicator: Deduplicator = deduplicator or BurstDeduplicator()
        self._organizer: Organizer = organizer or FolderOrganizer(root)
        self._metadata_writer: MetadataWriter = metadata_writer or ExifMetadataWriter()
        self._layout_planner: LayoutPlanner = layout_planner or SmartLayoutPlanner()
        self._exporter: Exporter = exporter or HtmlAlbumExporter()
        self._state_store: StateStore = state_store or JsonStateStore()
        self._max_workers = max_workers or min(8, os.cpu_count() or 2)
        self._state = self._state_store.load(root) or ProjectState(root=root)

    # ------------------------------------------------------------ accessors

    @property
    def root(self) -> Path:
        """Absolute project folder."""
        return self._root

    @property
    def state(self) -> ProjectState:
        """The live project state (mutate it only through pipeline methods)."""
        return self._state

    @property
    def has_cached_state(self) -> bool:
        """``True`` when the folder already has analyzed photos."""
        return bool(self._state.photos)

    def save(self) -> None:
        """Persist the current state."""
        self._state_store.save(self._state)

    # ----------------------------------------------------- scan and curate

    def scan_and_analyze(self, progress: ProgressCallback | None = None) -> ScanReport:
        """Synchronize the state with the folder and analyze what changed.

        Only new or modified files are decoded; everything else comes from
        the cache. Analysis runs on a thread pool — OpenCV releases the GIL
        during decoding, filtering and cascade detection, so threads scale
        without the pickling and start-up cost of processes.

        Args:
            progress: Optional ``(done, total, rel_path)`` reporter.

        Returns:
            Counts of analyzed, cached, moved and removed files plus errors.
        """
        state = self._state
        discovered = self._state_store.discover(self._root, self._organizer.linked_copies())
        diff = self._state_store.diff(state, discovered)
        report = ScanReport(cached=diff.unchanged, moved=len(diff.moved), removed=len(diff.removed))

        for photo_id in diff.removed:
            state.photos.pop(photo_id, None)
        for photo_id, rel_path in diff.moved.items():
            state.photos[photo_id].rel_path = rel_path

        by_path = state.photos_by_path()
        pending = [*diff.new, *diff.modified]
        state.unreadable = {
            path: reason
            for path, reason in state.unreadable.items()
            if path in discovered and path not in pending
        }

        total = len(pending)
        if total:
            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                futures = {
                    pool.submit(self._analyzer.analyze, self._root, rel_path): rel_path
                    for rel_path in pending
                }
                for done, future in enumerate(as_completed(futures), start=1):
                    rel_path = futures[future]
                    previous = by_path.get(rel_path)
                    try:
                        photo = future.result()
                    except (OSError, ValueError, RuntimeError) as exc:
                        report.errors[rel_path] = str(exc)
                        if previous is not None:
                            state.photos.pop(previous.photo_id, None)
                    else:
                        if previous is not None:
                            self._inherit_identity(photo, previous)
                        state.photos[photo.photo_id] = photo
                        report.analyzed += 1
                    if progress is not None:
                        progress(done, total, rel_path)

        state.unreadable.update(report.errors)
        self.curate()
        return report

    @staticmethod
    def _inherit_identity(photo: PhotoItem, previous: PhotoItem) -> None:
        """Carry the id and manual decisions of a re-analyzed photo over."""
        photo.photo_id = previous.photo_id
        if previous.scene_locked:
            photo.scene, photo.scene_locked = previous.scene, True
        if previous.rating_locked:
            photo.star_rating, photo.rating_locked = previous.star_rating, True
        if previous.selection_locked:
            photo.selection_locked = True
            photo.is_selected = previous.is_selected
            photo.rejection_category = previous.rejection_category
            photo.rejection_reason = previous.rejection_reason

    def curate(self) -> None:
        """Re-apply all automatic decisions and save.

        Order matters: scores first (resets automatic rejections), then burst
        culling (needs fresh scores), then event clustering, then the album
        is reconciled with the new keeper set.
        """
        state = self._state
        photos = list(state.photos.values())
        for photo in photos:
            self._analyzer.rescore(photo, state.config)
        self._deduplicator.mark_bursts(photos, state.config)
        state.clusters = self._deduplicator.cluster_events(photos, state.config)
        self._sync_album()
        self.save()

    def update_config(self, config: AnalysisConfig) -> None:
        """Apply new thresholds; no image is re-read."""
        self._state.config = config
        self.curate()

    def stats(self) -> DashboardStats:
        """Compute the dashboard numbers from the current state."""
        photos = list(self._state.photos.values())
        categories = Counter(p.rejection_category for p in photos if not p.is_selected)
        keepers = [p for p in photos if p.is_selected]
        scenes = Counter(p.scene for p in keepers)
        return DashboardStats(
            total=len(photos),
            keepers=len(keepers),
            rejected_quality=categories[RejectionCategory.QUALITY],
            rejected_duplicates=categories[RejectionCategory.DUPLICATE],
            rejected_manual=categories[RejectionCategory.MANUAL],
            heroes=sum(1 for p in keepers if p.star_rating == 5),
            scenes={scene: scenes.get(scene, 0) for scene in SceneType},
        )

    # ------------------------------------------------------ manual curation

    def set_selected(self, photo_id: str, selected: bool) -> None:
        """Restore a photo to the keepers or reject it manually."""
        photo = self._state.photos[photo_id]
        photo.is_selected = selected
        photo.selection_locked = True
        if selected:
            photo.rejection_category = RejectionCategory.NONE
            photo.rejection_reason = None
        else:
            photo.rejection_category = RejectionCategory.MANUAL
            photo.rejection_reason = MANUAL_REJECTION_REASON
        self._sync_album()
        self.save()

    def set_scene(self, photo_id: str, scene: SceneType) -> None:
        """Override a photo's scene type."""
        photo = self._state.photos[photo_id]
        photo.scene = scene
        photo.scene_locked = True
        self.save()

    def set_rating(self, photo_id: str, stars: int) -> None:
        """Override a photo's star rating (clamped to 1-5)."""
        photo = self._state.photos[photo_id]
        photo.star_rating = min(max(int(stars), 1), 5)
        photo.rating_locked = True
        self.save()

    def reset_overrides(self, photo_id: str) -> None:
        """Drop all manual decisions for a photo and re-curate."""
        photo = self._state.photos[photo_id]
        photo.scene_locked = photo.rating_locked = photo.selection_locked = False
        self.curate()

    # ---------------------------------------------------------- organization

    def preview_organization(
        self, strategy: OrganizationStrategy, mode: FileActionMode
    ) -> list[FileTransaction]:
        """Dry run: the exact operations a strategy would perform."""
        batch = self._organizer.organize(
            list(self._state.photos.values()),
            self._state.clusters,
            strategy,
            mode,
            dry_run=True,
        )
        return batch.transactions

    def execute_organization(
        self, strategy: OrganizationStrategy, mode: FileActionMode
    ) -> OperationBatch:
        """Execute a strategy and track moved files in the state."""
        batch = self._organizer.organize(
            list(self._state.photos.values()), self._state.clusters, strategy, mode
        )
        for transaction in batch.transactions:
            if transaction.executed and transaction.mode is FileActionMode.MOVE:
                self._repath(transaction.photo_id, Path(transaction.destination))
        self.save()
        return batch

    def can_undo_organization(self) -> bool:
        """``True`` when there is an organization batch to undo."""
        return self._organizer.can_undo()

    def undo_last_organization(self) -> OperationBatch | None:
        """Undo the most recent organization batch and restore paths."""
        batch = self._organizer.undo_last_operation()
        if batch is None:
            return None
        for transaction in batch.transactions:
            reverted = not transaction.executed and transaction.note == "Reverted"
            if reverted and transaction.mode is FileActionMode.MOVE:
                self._repath(transaction.photo_id, Path(transaction.source))
        self.save()
        return batch

    def _repath(self, photo_id: str, absolute: Path) -> None:
        """Point a photo at its new location on disk."""
        photo = self._state.photos.get(photo_id)
        if photo is None:
            return
        try:
            photo.rel_path = absolute.relative_to(self._root).as_posix()
        except ValueError:
            return

    # -------------------------------------------------------------- metadata

    def write_metadata(self, progress: ProgressCallback | None = None) -> MetadataReport:
        """Write ratings and keywords into every analyzed photo.

        The file fingerprint in the state is refreshed after each write so
        the tagged file is not treated as modified by the next scan.

        Args:
            progress: Optional ``(done, total, rel_path)`` reporter.

        Returns:
            Counts of written, skipped (non-JPEG) and failed files.
        """
        report = MetadataReport()
        titles = {c.cluster_id: c.title for c in self._state.clusters}
        photos = self._state.chronological()
        for done, photo in enumerate(photos, start=1):
            path = photo.absolute_path(self._root)
            result = self._metadata_writer.write(path, photo, titles.get(photo.cluster_id or ""))
            match result.status:
                case MetadataStatus.WRITTEN:
                    report.written += 1
                    stat = path.stat()
                    photo.size_bytes, photo.mtime_ns = stat.st_size, stat.st_mtime_ns
                case MetadataStatus.SKIPPED:
                    report.skipped += 1
                case MetadataStatus.FAILED:
                    report.failed += 1
                    report.failures[photo.rel_path] = result.message
            if progress is not None:
                progress(done, len(photos), photo.rel_path)
        self.save()
        return report

    # ----------------------------------------------------------------- album

    def keepers(self, cluster_id: str | None = None) -> list[PhotoItem]:
        """Selected photos in chronological order, optionally for one event."""
        return [
            photo
            for photo in self._state.chronological()
            if photo.is_selected and (cluster_id is None or photo.cluster_id == cluster_id)
        ]

    def generate_album(
        self, max_photos_per_page: int = 4, chapter_breaks: bool = True
    ) -> list[AlbumPage]:
        """Build the whole album in one step, replacing existing pages.

        Args:
            max_photos_per_page: Upper bound per page (1-4).
            chapter_breaks: Start every event on a new page with a heading.
                When ``False`` all keepers flow as one continuous sequence.

        Returns:
            The new pages.
        """
        pages: list[AlbumPage] = []
        if chapter_breaks:
            for cluster in self._state.clusters:
                pages.extend(
                    self._plan(self.keepers(cluster.cluster_id), cluster, max_photos_per_page)
                )
        else:
            pages = self._plan(self.keepers(), None, max_photos_per_page)
        self._state.pages = pages
        self.save()
        return pages

    def generate_chapter(self, cluster_id: str, max_photos_per_page: int = 4) -> list[AlbumPage]:
        """Build (or rebuild) the pages of one event, keeping the others.

        The chapter's pages are inserted at the chronologically correct
        position among the chapters that already exist.

        Args:
            cluster_id: Event to lay out.
            max_photos_per_page: Upper bound per page (1-4).

        Returns:
            The chapter's new pages.

        Raises:
            KeyError: If the event does not exist.
        """
        cluster = self._state.cluster_by_id(cluster_id)
        if cluster is None:
            raise KeyError(f"Unknown event: {cluster_id}")
        new_pages = self._plan(self.keepers(cluster_id), cluster, max_photos_per_page)

        order = {c.cluster_id: i for i, c in enumerate(self._state.clusters)}
        position = order[cluster_id]
        kept = [p for p in self._state.pages if p.cluster_id != cluster_id]
        # A photo placed manually on another chapter's page moves back home.
        placed = {pid for page in new_pages for pid in page.photo_ids}
        for page in kept:
            page.photo_ids = [pid for pid in page.photo_ids if pid not in placed]
        insert_at = len(kept)
        for index, page in enumerate(kept):
            if order.get(page.cluster_id or "", -1) > position:
                insert_at = index
                break
        self._state.pages = [*kept[:insert_at], *new_pages, *kept[insert_at:]]
        self._sync_album()
        self.save()
        return new_pages

    def chapters_built(self) -> set[str]:
        """Ids of the events that currently have album pages."""
        return {page.cluster_id for page in self._state.pages if page.cluster_id}

    def clear_album(self) -> None:
        """Remove all album pages."""
        self._state.pages = []
        self.save()

    def _plan(
        self,
        photos: Sequence[PhotoItem],
        cluster: EventCluster | None,
        max_photos_per_page: int,
    ) -> list[AlbumPage]:
        """Delegate to the layout planner."""
        return self._layout_planner.plan_pages(
            photos,
            self._state.config,
            max_photos_per_page=max_photos_per_page,
            cluster=cluster,
        )

    def _sync_album(self) -> None:
        """Drop non-keepers from pages, remove empty pages, fix templates.

        A chapter heading on a page that gets removed is handed to the next
        page of the same chapter so the heading is not lost.
        """
        photos = self._state.photos
        synced: list[AlbumPage] = []
        orphan_heading: tuple[str | None, str] | None = None
        for page in self._state.pages:
            valid = [pid for pid in page.photo_ids if pid in photos and photos[pid].is_selected]
            changed = valid != page.photo_ids
            page.photo_ids = valid
            if not valid:
                if page.chapter_title:
                    orphan_heading = (page.cluster_id, page.chapter_title)
                continue
            if orphan_heading and orphan_heading[0] == page.cluster_id:
                page.chapter_title = page.chapter_title or orphan_heading[1]
            orphan_heading = None
            if changed:
                self._layout_planner.recalculate_page(page, photos)
            synced.append(page)
        self._state.pages = synced

    # ---------------------------------------------------------- page editing

    def page_index(self, page_id: str) -> int:
        """Position of a page in the album.

        Raises:
            KeyError: If the page does not exist.
        """
        for index, page in enumerate(self._state.pages):
            if page.page_id == page_id:
                return index
        raise KeyError(f"Unknown page: {page_id}")

    def _page(self, page_id: str) -> AlbumPage:
        """Look a page up by id."""
        return self._state.pages[self.page_index(page_id)]

    def unused_keepers(self) -> list[PhotoItem]:
        """Keepers that are not placed on any album page."""
        placed = {pid for page in self._state.pages for pid in page.photo_ids}
        return [photo for photo in self.keepers() if photo.photo_id not in placed]

    def set_page_template(self, page_id: str, template: LayoutTemplate) -> None:
        """Switch a page's layout template.

        Raises:
            ValueError: If the template cannot hold the page's photos.
        """
        page = self._page(page_id)
        if template.capacity < len(page.photo_ids):
            raise ValueError(
                f"{template.label} holds {template.capacity} photo(s); "
                f"the page has {len(page.photo_ids)}"
            )
        page.template = template
        page.template_locked = True
        self.save()

    def reset_page_template(self, page_id: str) -> None:
        """Return a page to automatic template selection."""
        page = self._page(page_id)
        page.template_locked = False
        self._layout_planner.recalculate_page(page, self._state.photos)
        self.save()

    def reorder_photo(self, page_id: str, photo_id: str, offset: int) -> None:
        """Swap a photo with its neighbour ``offset`` slots away (-1 or +1)."""
        page = self._page(page_id)
        index = page.photo_ids.index(photo_id)
        target = index + offset
        if 0 <= target < len(page.photo_ids):
            page.photo_ids[index], page.photo_ids[target] = (
                page.photo_ids[target],
                page.photo_ids[index],
            )
            self.save()

    def add_photo_to_page(self, page_id: str, photo_id: str) -> None:
        """Place a keeper on a page (removing it from any other page).

        Raises:
            ValueError: If the page is full or the photo is not a keeper.
        """
        page = self._page(page_id)
        photo = self._state.photos[photo_id]
        if not photo.is_selected:
            raise ValueError("Only keepers can be placed in the album")
        if photo_id in page.photo_ids:
            return
        if len(page.photo_ids) >= MAX_PHOTOS_PER_PAGE:
            raise ValueError(f"A page holds at most {MAX_PHOTOS_PER_PAGE} photos")
        for other in self._state.pages:
            if photo_id in other.photo_ids:
                other.photo_ids.remove(photo_id)
                self._layout_planner.recalculate_page(other, self._state.photos)
        page.photo_ids.append(photo_id)
        self._layout_planner.recalculate_page(page, self._state.photos)
        self._sync_album()
        self.save()

    def move_photo(self, photo_id: str, target_page_id: str) -> None:
        """Move a photo from its current page to another page."""
        self.add_photo_to_page(target_page_id, photo_id)

    def remove_photo_from_page(self, page_id: str, photo_id: str) -> None:
        """Take a photo off a page; it returns to the unused keeper pool."""
        page = self._page(page_id)
        if photo_id in page.photo_ids:
            page.photo_ids.remove(photo_id)
            self._layout_planner.recalculate_page(page, self._state.photos)
            self._sync_album()
            self.save()

    def add_page(self, photo_id: str, after_page_id: str | None = None) -> AlbumPage:
        """Create a new page holding one unused keeper.

        Args:
            photo_id: Keeper to place on the new page.
            after_page_id: Insert after this page; append when ``None``.

        Returns:
            The new page.
        """
        photo = self._state.photos[photo_id]
        if not photo.is_selected:
            raise ValueError("Only keepers can be placed in the album")
        for other in self._state.pages:
            if photo_id in other.photo_ids:
                other.photo_ids.remove(photo_id)
                self._layout_planner.recalculate_page(other, self._state.photos)
        anchor = self._page(after_page_id) if after_page_id else None
        page = AlbumPage(
            page_id=new_id(),
            template=LayoutTemplate.HERO_SINGLE,
            photo_ids=[photo_id],
            cluster_id=anchor.cluster_id if anchor else photo.cluster_id,
        )
        index = self.page_index(after_page_id) + 1 if after_page_id else len(self._state.pages)
        self._state.pages.insert(index, page)
        self._sync_album()
        self.save()
        return page

    def delete_page(self, page_id: str) -> None:
        """Delete a page; its photos return to the unused keeper pool."""
        page = self._page(page_id)
        page.photo_ids = []
        self._sync_album()
        self.save()

    def move_page(self, page_id: str, offset: int) -> None:
        """Move a page earlier (-1) or later (+1) in the album."""
        index = self.page_index(page_id)
        target = index + offset
        pages = self._state.pages
        if 0 <= target < len(pages):
            pages[index], pages[target] = pages[target], pages[index]
            self.save()

    # ---------------------------------------------------------------- export

    def render_page_preview(
        self,
        page_id: str,
        options: ExportOptions,
        thumbnail_uri: Callable[[PhotoItem], str],
    ) -> str:
        """Render one page as a small HTML document for the studio preview.

        Args:
            page_id: Page to preview.
            options: Page format and crop mode.
            thumbnail_uri: Returns an image ``src`` for a photo (the UI
                passes its thumbnail cache).

        Returns:
            A complete HTML document sized to fit its viewport.
        """
        return self._exporter.render_preview(
            self._page(page_id), self._state.photos, options, thumbnail_uri
        )

    def export_album(
        self, options: ExportOptions | None = None, output_path: Path | None = None
    ) -> Path:
        """Export the album as a print-ready HTML file.

        Args:
            options: Title, format and image handling.
            output_path: Destination; defaults to
                ``<folder>/PhotoSeder_Album.html``.

        Returns:
            Path of the written file.

        Raises:
            ValueError: If the album has no pages.
        """
        if not self._state.pages:
            raise ValueError("The album has no pages yet; generate it first")
        return self._exporter.export(
            self._state.pages,
            self._state.photos,
            self._root,
            options or ExportOptions(),
            output_path or self._root / DEFAULT_ALBUM_FILENAME,
        )

    # ------------------------------------------------------------- autopilot

    def run_autopilot(
        self,
        *,
        max_photos_per_page: int = 4,
        chapter_breaks: bool = True,
        options: ExportOptions | None = None,
        output_path: Path | None = None,
        progress: ProgressCallback | None = None,
    ) -> Path:
        """Scan, curate, build the album and export it in one call.

        Args:
            max_photos_per_page: Upper bound per page (1-4).
            chapter_breaks: Start every event on a new page.
            options: Export options.
            output_path: Destination HTML file.
            progress: Optional analysis progress reporter.

        Returns:
            Path of the exported album.
        """
        self.scan_and_analyze(progress)
        self.generate_album(max_photos_per_page, chapter_breaks)
        return self.export_album(options, output_path)
