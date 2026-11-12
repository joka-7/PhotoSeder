"""End-to-end tests: pipeline, organizer round trips and incremental caching."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from photoseder.domain.enums import (
    FileActionMode,
    LayoutTemplate,
    OrganizationStrategy,
    RejectionCategory,
    SceneType,
)
from photoseder.orchestrator.pipeline import Pipeline
from photoseder.services.analyzer import OpenCvImageAnalyzer
from photoseder.services.interfaces import ExportOptions


def snapshot(root: Path) -> dict[str, str]:
    """Map every photo under ``root`` to a content hash."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.jpg"))
    }


class CountingAnalyzer(OpenCvImageAnalyzer):
    """Analyzer that counts how many files it actually decodes."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def analyze(self, root: Path, rel_path: str):  # noqa: ANN201
        self.calls += 1
        return super().analyze(root, rel_path)


@pytest.fixture
def pipeline(library: Path) -> Pipeline:
    pipe = Pipeline(library)
    pipe.scan_and_analyze()
    return pipe


def by_name(pipe: Pipeline) -> dict:
    return {photo.name: photo for photo in pipe.state.photos.values()}


def test_curation_decisions(pipeline: Pipeline) -> None:
    photos = by_name(pipeline)
    stats = pipeline.stats()

    assert stats.total == 14
    assert not photos["b_blurry.jpg"].is_selected
    assert not photos["c_dark.jpg"].is_selected
    assert not photos["d_bright.jpg"].is_selected
    assert photos["e_burst1.jpg"].is_selected
    assert photos["e_burst2.jpg"].rejection_category is RejectionCategory.DUPLICATE
    assert (stats.rejected_quality, stats.rejected_duplicates) == (3, 1)
    assert stats.keepers == 10
    assert [c.title for c in pipeline.state.clusters] == [
        "2026-10-15_Morning",
        "2026-10-15_Evening",
        "2026-10-16_Morning",
    ]


def test_cache_only_analyzes_new_and_modified(library: Path, make_photo) -> None:
    Pipeline(library).scan_and_analyze()

    analyzer = CountingAnalyzer()
    pipe = Pipeline(library, analyzer=analyzer)
    report = pipe.scan_and_analyze()
    assert analyzer.calls == 0 and report.cached == 14 and report.analyzed == 0

    make_photo("תמונות/new.jpg", seed=99)
    make_photo("תמונות/a_sharp.jpg", seed=98)
    (library / "day2_0.jpg").unlink()
    old_id = by_name(pipe)["a_sharp.jpg"].photo_id
    report = pipe.scan_and_analyze()

    assert analyzer.calls == 2 and report.removed == 1
    assert len(pipe.state.photos) == 14
    assert by_name(pipe)["a_sharp.jpg"].photo_id == old_id


def test_manual_overrides_survive_rescoring(pipeline: Pipeline) -> None:
    blurry = by_name(pipeline)["b_blurry.jpg"]
    pipeline.set_selected(blurry.photo_id, True)
    pipeline.set_scene(blurry.photo_id, SceneType.PORTRAIT)
    pipeline.set_rating(blurry.photo_id, 4)
    pipeline.curate()

    reloaded = by_name(Pipeline(pipeline.root))["b_blurry.jpg"]
    assert reloaded.is_selected and reloaded.scene is SceneType.PORTRAIT
    assert reloaded.star_rating == 4

    pipeline.reset_overrides(blurry.photo_id)
    assert not by_name(pipeline)["b_blurry.jpg"].is_selected


@pytest.mark.parametrize("strategy", list(OrganizationStrategy))
@pytest.mark.parametrize("mode", list(FileActionMode))
def test_organize_then_undo_restores_tree(
    pipeline: Pipeline, strategy: OrganizationStrategy, mode: FileActionMode
) -> None:
    root = pipeline.root
    before = snapshot(root)
    paths_before = {p.photo_id: p.rel_path for p in pipeline.state.photos.values()}

    preview = pipeline.preview_organization(strategy, mode)
    assert snapshot(root) == before, "dry run must not touch the disk"
    assert preview and not pipeline.can_undo_organization()

    batch = pipeline.execute_organization(strategy, mode)
    assert batch.executed_count == len(preview)
    assert snapshot(root) != before
    for photo in pipeline.state.photos.values():
        assert photo.absolute_path(root).is_file()

    # Copies and links must not be re-ingested as new photos.
    assert pipeline.scan_and_analyze().analyzed == 0
    assert len(pipeline.state.photos) == 14

    undone = pipeline.undo_last_organization()
    assert undone is not None and undone.executed_count == 0
    assert snapshot(root) == before
    assert {p.photo_id: p.rel_path for p in pipeline.state.photos.values()} == paths_before
    assert [p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")] == []
    assert not pipeline.can_undo_organization()


def test_cull_folders_and_restore(pipeline: Pipeline) -> None:
    root = pipeline.root
    pipeline.execute_organization(OrganizationStrategy.CULL_REJECTED, FileActionMode.MOVE)

    assert sorted(p.name for p in (root / "_Rejected_Quality").iterdir()) == [
        "b_blurry.jpg",
        "c_dark.jpg",
        "d_bright.jpg",
    ]
    assert [p.name for p in (root / "_Rejected_Duplicates").iterdir()] == ["e_burst2.jpg"]
    assert len(list(root.glob("*.jpg"))) == 10

    # Restoring a culled photo and culling again brings it back to the root.
    blurry = by_name(pipeline)["b_blurry.jpg"]
    pipeline.set_selected(blurry.photo_id, True)
    pipeline.execute_organization(OrganizationStrategy.CULL_REJECTED, FileActionMode.MOVE)
    assert (root / "b_blurry.jpg").is_file()
    assert blurry.rel_path == "b_blurry.jpg"


def test_hardlinks_share_storage_and_names_never_collide(pipeline: Pipeline) -> None:
    root = pipeline.root
    (root / "Best_Of").mkdir()
    (root / "Best_Of" / "a_sharp.jpg").write_bytes(b"user file")

    batch = pipeline.execute_organization(
        OrganizationStrategy.BEST_OF_HEROES, FileActionMode.HARDLINK
    )
    assert (root / "Best_Of" / "a_sharp.jpg").read_bytes() == b"user file"
    linked = root / "Best_Of" / "a_sharp_1.jpg"
    assert os.path.samefile(linked, root / "a_sharp.jpg")
    assert all(t.mode is FileActionMode.HARDLINK for t in batch.transactions)

    pipeline.undo_last_organization()
    assert [p.name for p in (root / "Best_Of").iterdir()] == ["a_sharp.jpg"]


def test_metadata_write_keeps_cache_valid(library: Path) -> None:
    pipe = Pipeline(library)
    pipe.scan_and_analyze()
    report = pipe.write_metadata()
    assert (report.written, report.skipped, report.failed) == (14, 0, 0)

    analyzer = CountingAnalyzer()
    assert Pipeline(library, analyzer=analyzer).scan_and_analyze().cached == 14
    assert analyzer.calls == 0


def test_album_generation_editing_and_export(pipeline: Pipeline) -> None:
    pages = pipeline.generate_album(max_photos_per_page=4, chapter_breaks=True)
    keeper_ids = {p.photo_id for p in pipeline.keepers()}

    assert {pid for page in pages for pid in page.photo_ids} == keeper_ids
    assert sum(len(page.photo_ids) for page in pages) == len(keeper_ids)
    assert [p.chapter_title for p in pages if p.chapter_title] == [
        "2026-10-15_Morning",
        "2026-10-15_Evening",
        "2026-10-16_Morning",
    ]
    assert all(len(p.photo_ids) <= p.template.capacity for p in pages)
    assert not pipeline.unused_keepers()

    # Rejecting a photo removes it from the album and returns nothing unused.
    victim = pages[-1].photo_ids[0]
    pipeline.set_selected(victim, False)
    assert victim not in {pid for page in pipeline.state.pages for pid in page.photo_ids}
    pipeline.set_selected(victim, True)
    assert [p.photo_id for p in pipeline.unused_keepers()] == [victim]

    # Page editing.
    target = next(p for p in pipeline.state.pages if len(p.photo_ids) < 4)
    pipeline.add_photo_to_page(target.page_id, victim)
    assert victim in target.photo_ids and not pipeline.unused_keepers()
    first = target.photo_ids[0]
    pipeline.reorder_photo(target.page_id, first, +1)
    assert target.photo_ids[1] == first
    with pytest.raises(ValueError):
        pipeline.set_page_template(target.page_id, LayoutTemplate.HERO_SINGLE)
    pipeline.set_page_template(target.page_id, LayoutTemplate.QUAD_GRID)
    assert target.template is LayoutTemplate.QUAD_GRID

    count = len(pipeline.state.pages)
    new_page = pipeline.add_page(victim, after_page_id=pipeline.state.pages[0].page_id)
    assert pipeline.state.pages[1] is new_page and len(pipeline.state.pages) == count + 1
    pipeline.delete_page(new_page.page_id)
    assert len(pipeline.state.pages) == count

    # Chapter wizard rebuilds one event in place.
    pipeline.generate_chapter("2026-10-15_Evening", max_photos_per_page=2)
    rank = {c.cluster_id: i for i, c in enumerate(pipeline.state.clusters)}
    order = [rank[p.cluster_id] for p in pipeline.state.pages]
    assert order == sorted(order)
    evening = [p for p in pipeline.state.pages if p.cluster_id == "2026-10-15_Evening"]
    assert all(len(p.photo_ids) <= 2 for p in evening)

    output = pipeline.export_album(ExportOptions(title="טיול סתיו"))
    html = output.read_text(encoding="utf-8")
    assert html.count('<section class="page') == len(pipeline.state.pages) + 1
    assert "טיול סתיו" in html and "https://" not in html

    # Pages persist across sessions.
    assert len(Pipeline(pipeline.root).state.pages) == len(pipeline.state.pages)


def test_autopilot(library: Path, tmp_path: Path) -> None:
    output = Pipeline(library).run_autopilot(output_path=tmp_path / "out" / "album.html")
    assert output.is_file() and output.stat().st_size > 100_000
