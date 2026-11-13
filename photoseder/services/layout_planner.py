"""Aspect-ratio and quality-aware album page layout engine.

Algorithm (deterministic greedy pass over a chronological sequence):
    1. A *hero* photo — scene ``LANDSCAPE`` or ``PORTRAIT`` with
       ``overall_score >= hero_threshold`` — always gets its own
       ``HERO_SINGLE`` page.
    2. The photos between heroes form a *run*. A run is cut into pages of up
       to ``max_photos_per_page`` photos:

       * four photos become a ``QUAD_GRID`` only when all four are ``GROUP``
         or ``GENERAL`` candids; otherwise the page takes three;
       * three photos become a ``TRIO_FEATURE`` with the highest-scoring
         photo moved to the dominant slot;
       * two photos become ``DUO_HORIZONTAL`` (stacked) when both are
         horizontal, else ``DUO_VERTICAL`` (side by side);
       * one photo becomes a ``HERO_SINGLE``.

    3. *Orphan avoidance*: if taking the planned number would leave exactly
       one photo at the end of the run, the page takes one fewer (so five
       photos become 3 + 2 rather than 4 + 1).

    Chronological order is preserved across pages, which keeps the album
    reading as a story; only the dominant slot of a trio is reordered.

Interactive editing:
    :meth:`SmartLayoutPlanner.recalculate_page` re-picks the template for
    whatever photos a page holds after the user adds, removes or moves one.
    A template chosen manually is kept as long as it can hold the photos.
"""

from __future__ import annotations

from collections.abc import Sequence

from photoseder.domain.enums import LayoutTemplate, Orientation, SceneType
from photoseder.domain.models import (
    AlbumPage,
    AnalysisConfig,
    EventCluster,
    PhotoItem,
    new_id,
)

MAX_PHOTOS_PER_PAGE = 4
_HERO_SCENES = (SceneType.LANDSCAPE, SceneType.PORTRAIT)
_CANDID_SCENES = (SceneType.GROUP, SceneType.GENERAL)


def compatible_templates(photo_count: int) -> list[LayoutTemplate]:
    """Templates able to display ``photo_count`` photos (capacity >= count)."""
    return [t for t in LayoutTemplate if t.capacity >= max(photo_count, 1)]


class SmartLayoutPlanner:
    """Groups photos into visually balanced album pages."""

    def plan_pages(
        self,
        photos: Sequence[PhotoItem],
        config: AnalysisConfig,
        *,
        max_photos_per_page: int,
        cluster: EventCluster | None = None,
    ) -> list[AlbumPage]:
        """Lay out a chronological photo sequence as album pages.

        Args:
            photos: Keepers to place, in the order they should appear.
            config: Supplies ``hero_threshold``.
            max_photos_per_page: Upper bound per page, clamped to 1-4.
            cluster: Chapter the photos belong to; its title is attached to
                the first page as the chapter heading.

        Returns:
            Pages in print order (empty when ``photos`` is empty).
        """
        limit = min(max(max_photos_per_page, 1), MAX_PHOTOS_PER_PAGE)
        pages: list[AlbumPage] = []
        run: list[PhotoItem] = []

        for photo in photos:
            if self.is_hero(photo, config):
                pages.extend(self._layout_run(run, limit))
                run = []
                pages.append(self._page([photo], LayoutTemplate.HERO_SINGLE))
            else:
                run.append(photo)
        pages.extend(self._layout_run(run, limit))

        for page in pages:
            page.cluster_id = cluster.cluster_id if cluster else None
        if pages and cluster is not None:
            pages[0].chapter_title = cluster.title
        return pages

    @staticmethod
    def is_hero(photo: PhotoItem, config: AnalysisConfig) -> bool:
        """Return ``True`` when a photo deserves a full page of its own."""
        return photo.scene in _HERO_SCENES and photo.metrics.overall_score >= config.hero_threshold

    def _layout_run(self, run: list[PhotoItem], limit: int) -> list[AlbumPage]:
        """Cut a run of non-hero photos into pages."""
        pages: list[AlbumPage] = []
        index = 0
        while index < len(run):
            remaining = len(run) - index
            size = min(limit, remaining)
            if size == 4 and not all(p.scene in _CANDID_SCENES for p in run[index : index + 4]):
                size = 3
            if size >= 3 and remaining - size == 1:
                size -= 1
            group = run[index : index + size]
            index += size

            template = self.choose_template(group)
            if template is LayoutTemplate.TRIO_FEATURE:
                dominant = max(group, key=lambda p: p.metrics.overall_score)
                group = [dominant, *[p for p in group if p is not dominant]]
            pages.append(self._page(group, template))
        return pages

    def choose_template(self, photos: Sequence[PhotoItem]) -> LayoutTemplate:
        """Pick the best template for the photos currently on one page.

        Args:
            photos: Photos on the page (at most four are considered).

        Returns:
            The template matching the photo count and orientations.
        """
        count = len(photos)
        if count <= 1:
            return LayoutTemplate.HERO_SINGLE
        if count == 2:
            both_horizontal = all(p.orientation is Orientation.HORIZONTAL for p in photos)
            return LayoutTemplate.DUO_HORIZONTAL if both_horizontal else LayoutTemplate.DUO_VERTICAL
        if count == 3:
            return LayoutTemplate.TRIO_FEATURE
        return LayoutTemplate.QUAD_GRID

    def recalculate_page(self, page: AlbumPage, photos_by_id: dict[str, PhotoItem]) -> None:
        """Re-pick a page's template after its photos changed.

        A user-chosen template is kept while its capacity still fits the
        photos; otherwise the lock is released and the best automatic
        template is applied.

        Args:
            page: Page to update in place.
            photos_by_id: Lookup for the page's photos.
        """
        photos = [photos_by_id[pid] for pid in page.photo_ids if pid in photos_by_id]
        if page.template_locked and page.template.capacity >= len(photos):
            return
        page.template_locked = False
        page.template = self.choose_template(photos)

    @staticmethod
    def _page(photos: Sequence[PhotoItem], template: LayoutTemplate) -> AlbumPage:
        """Create a page holding ``photos`` in the given order."""
        return AlbumPage(
            page_id=new_id(),
            template=template,
            photo_ids=[photo.photo_id for photo in photos],
        )
