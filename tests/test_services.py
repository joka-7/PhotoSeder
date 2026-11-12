"""Unit tests for the analysis, deduplication, tagging and layout services."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import piexif
from PIL import Image

from photoseder.domain.enums import (
    LayoutTemplate,
    Orientation,
    RejectionCategory,
    SceneType,
)
from photoseder.domain.models import (
    AnalysisConfig,
    FaceBox,
    PhotoItem,
    ProjectState,
    QualityMetrics,
)
from photoseder.services.analyzer import OpenCvImageAnalyzer, star_rating_for
from photoseder.services.deduplicator import (
    BURST_REJECTION_REASON,
    BurstDeduplicator,
    compute_dhash,
    hamming_distance,
)
from photoseder.services.exporter import HtmlAlbumExporter
from photoseder.services.interfaces import ExportOptions, MetadataStatus
from photoseder.services.layout_planner import SmartLayoutPlanner
from photoseder.services.metadata_writer import ExifMetadataWriter, decode_xp_keywords
from photoseder.services.state_manager import JsonStateStore


class FakeFaces:
    """Face detector stub returning preset boxes."""

    def __init__(self, faces: list[FaceBox]) -> None:
        self.faces = faces

    def detect(self, gray: np.ndarray) -> list[FaceBox]:
        return list(self.faces)


def photo(
    photo_id: str,
    *,
    score: float = 0.6,
    scene: SceneType = SceneType.GENERAL,
    orientation: Orientation = Orientation.HORIZONTAL,
    taken: datetime | None = None,
    dhash: int = 0,
) -> PhotoItem:
    """Build an in-memory photo with a preset score."""
    return PhotoItem(
        photo_id=photo_id,
        rel_path=f"{photo_id}.jpg",
        size_bytes=1,
        mtime_ns=1,
        captured_at=taken or datetime(2026, 1, 1, 10, 0, 0),
        width=1600 if orientation is not Orientation.VERTICAL else 1067,
        height=1067 if orientation is not Orientation.VERTICAL else 1600,
        orientation=orientation,
        metrics=QualityMetrics(overall_score=score),
        dhash=dhash,
        scene=scene,
        star_rating=star_rating_for(score),
    )


# ------------------------------------------------------------------ analyzer


def test_sharp_photo_is_kept_and_blurry_rejected(tmp_path: Path, make_photo) -> None:
    make_photo("sharp.jpg", seed=1)
    make_photo("blurry.jpg", seed=1, blur=9)
    analyzer = OpenCvImageAnalyzer()

    sharp = analyzer.analyze(tmp_path, "sharp.jpg")
    blurry = analyzer.analyze(tmp_path, "blurry.jpg")

    assert sharp.is_selected and sharp.star_rating >= 4
    assert sharp.metrics.sharpness > blurry.metrics.sharpness
    assert not blurry.is_selected
    assert blurry.star_rating == 1
    assert blurry.rejection_category is RejectionCategory.QUALITY
    assert blurry.rejection_reason.startswith("Blurry")


def test_exposure_defects_are_named(tmp_path: Path, make_photo) -> None:
    make_photo("dark.jpg", seed=3, gain=0.06)
    make_photo("bright.jpg", seed=4, gain=0.1, offset=245)
    analyzer = OpenCvImageAnalyzer()

    dark = analyzer.analyze(tmp_path, "dark.jpg")
    bright = analyzer.analyze(tmp_path, "bright.jpg")

    assert not dark.is_selected and not bright.is_selected
    assert dark.metrics.exposure < 0.4 and bright.metrics.exposure < 0.4
    assert dark.rejection_reason.startswith("Underexposed")
    assert bright.rejection_reason.startswith("Overexposed")


def test_exif_date_and_orientation_are_respected(tmp_path: Path, make_photo) -> None:
    taken = datetime(2024, 5, 6, 7, 8, 9)
    make_photo("rotated.jpg", seed=1, taken=taken, orientation=6)
    item = OpenCvImageAnalyzer().analyze(tmp_path, "rotated.jpg")

    assert item.captured_at == taken
    assert (item.width, item.height) == (1067, 1600)
    assert item.orientation is Orientation.VERTICAL


def test_capture_time_falls_back_to_mtime(tmp_path: Path) -> None:
    Image.fromarray(np.full((300, 400, 3), 128, np.uint8)).save(tmp_path / "plain.png")
    item = OpenCvImageAnalyzer().analyze(tmp_path, "plain.png")
    mtime = datetime.fromtimestamp((tmp_path / "plain.png").stat().st_mtime)
    assert abs(item.captured_at - mtime) < timedelta(seconds=1)


def test_scene_rules(tmp_path: Path, make_photo) -> None:
    make_photo("h.jpg", seed=1)
    make_photo("v.jpg", seed=2, size=(1067, 1600))
    big = FaceBox(0.3, 0.2, 0.3, 0.3, eyes_detected=2)
    small = FaceBox(0.1, 0.1, 0.05, 0.05, eyes_detected=2)

    def scene(name: str, faces: list[FaceBox]) -> SceneType:
        return OpenCvImageAnalyzer(face_detector=FakeFaces(faces)).analyze(tmp_path, name).scene

    assert scene("h.jpg", []) is SceneType.LANDSCAPE
    assert scene("v.jpg", []) is SceneType.GENERAL
    assert scene("h.jpg", [big]) is SceneType.PORTRAIT
    assert scene("h.jpg", [small]) is SceneType.GROUP
    assert scene("h.jpg", [small, small, small]) is SceneType.GROUP


def test_closed_eyes_lower_the_score(tmp_path: Path, make_photo) -> None:
    make_photo("p.jpg", seed=1)
    open_eyes = OpenCvImageAnalyzer(
        face_detector=FakeFaces([FaceBox(0.3, 0.2, 0.3, 0.3, 2)])
    ).analyze(tmp_path, "p.jpg")
    closed = OpenCvImageAnalyzer(face_detector=FakeFaces([FaceBox(0.3, 0.2, 0.3, 0.3, 0)])).analyze(
        tmp_path, "p.jpg"
    )

    assert open_eyes.metrics.eyes_open == 1.0 and closed.metrics.eyes_open == 0.0
    assert abs(open_eyes.metrics.overall_score - closed.metrics.overall_score - 0.10) < 1e-6


def test_rescore_respects_locks_and_thresholds(tmp_path: Path, make_photo) -> None:
    make_photo("s.jpg", seed=1)
    analyzer = OpenCvImageAnalyzer()
    item = analyzer.analyze(tmp_path, "s.jpg")
    assert item.is_selected

    strict = AnalysisConfig(blur_threshold=item.metrics.laplacian_variance * 2)
    analyzer.rescore(item, strict)
    assert not item.is_selected

    item.is_selected, item.selection_locked = True, True
    item.star_rating, item.rating_locked = 5, True
    analyzer.rescore(item, strict)
    assert item.is_selected and item.star_rating == 5


def test_star_mapping() -> None:
    assert [star_rating_for(s) for s in (0.9, 0.85, 0.84, 0.7, 0.55, 0.4, 0.39)] == [
        5,
        5,
        4,
        4,
        3,
        2,
        1,
    ]


# -------------------------------------------------------------- deduplicator


def test_dhash_is_stable_for_near_duplicates(tmp_path: Path, make_photo) -> None:
    import cv2

    def gray(path: Path) -> np.ndarray:
        return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)

    a = compute_dhash(gray(make_photo("a.jpg", seed=5)))
    b = compute_dhash(gray(make_photo("b.jpg", seed=5, blur=0.6)))
    c = compute_dhash(gray(make_photo("c.jpg", seed=6)))

    assert hamming_distance(a, b) <= 6
    assert hamming_distance(a, c) > 6


def test_burst_keeps_best_and_respects_time_window() -> None:
    t0 = datetime(2026, 1, 1, 10, 0, 0)
    best = photo("best", score=0.9, taken=t0 + timedelta(seconds=2), dhash=0b1)
    worse = photo("worse", score=0.7, taken=t0, dhash=0b0)
    late = photo("late", score=0.5, taken=t0 + timedelta(seconds=60), dhash=0b0)
    different = photo("different", score=0.5, taken=t0 + timedelta(seconds=4), dhash=2**40 - 1)

    rejected = BurstDeduplicator().mark_bursts([best, worse, late, different], AnalysisConfig())

    assert rejected == 1
    assert best.is_selected and late.is_selected and different.is_selected
    assert not worse.is_selected
    assert worse.rejection_reason == BURST_REJECTION_REASON
    assert worse.rejection_category is RejectionCategory.DUPLICATE
    assert best.burst_id == worse.burst_id and late.burst_id is None


def test_event_clustering_by_gap_and_date() -> None:
    photos = [
        photo("a", taken=datetime(2026, 10, 15, 9, 0)),
        photo("b", taken=datetime(2026, 10, 15, 9, 30)),
        photo("c", taken=datetime(2026, 10, 15, 13, 0)),
        photo("d", taken=datetime(2026, 10, 15, 23, 50)),
        photo("e", taken=datetime(2026, 10, 16, 0, 10)),
    ]
    clusters = BurstDeduplicator().cluster_events(photos, AnalysisConfig())

    assert [c.title for c in clusters] == [
        "2026-10-15_Morning",
        "2026-10-15_Afternoon",
        "2026-10-15_Night",
        "2026-10-16_Night",
    ]
    assert clusters[0].photo_ids == ["a", "b"]
    assert photos[4].cluster_id == "2026-10-16_Night"


# ------------------------------------------------------------ metadata writer


def _pixel_hash(path: Path) -> str:
    with Image.open(path) as image:
        return hashlib.sha256(image.tobytes()).hexdigest()


def test_exif_write_is_lossless_and_readable(tmp_path: Path, make_photo) -> None:
    path = make_photo("שלום.jpg", seed=1, taken=datetime(2026, 10, 15, 9, 0))
    before_pixels, before_mtime = _pixel_hash(path), path.stat().st_mtime_ns
    item = photo("x", score=0.9, scene=SceneType.LANDSCAPE)
    writer = ExifMetadataWriter()

    assert writer.write(path, item, "2026-10-15_Morning").status is MetadataStatus.WRITTEN
    assert writer.write(path, item, "2026-10-15_Morning").status is MetadataStatus.WRITTEN

    exif = piexif.load(str(path))
    assert exif["0th"][piexif.ImageIFD.Rating] == 5
    assert exif["0th"][piexif.ImageIFD.RatingPercent] == 99
    assert decode_xp_keywords(exif["0th"][piexif.ImageIFD.XPKeywords]) == [
        "Landscape",
        "Hero_5Star",
        "2026-10-15_Morning",
    ]
    description = exif["0th"][piexif.ImageIFD.ImageDescription].decode()
    assert description.count("[PhotoSeder:") == 1
    assert exif["Exif"][piexif.ExifIFD.DateTimeOriginal] == b"2026:10:15 09:00:00"
    assert _pixel_hash(path) == before_pixels
    assert path.stat().st_mtime_ns == before_mtime


def test_exif_write_skips_non_jpeg(tmp_path: Path) -> None:
    path = tmp_path / "x.png"
    Image.new("RGB", (10, 10)).save(path)
    result = ExifMetadataWriter().write(path, photo("x"), None)
    assert result.status is MetadataStatus.SKIPPED


# -------------------------------------------------------------- layout planner


def test_layout_rules() -> None:
    planner, config = SmartLayoutPlanner(), AnalysisConfig()
    v, h = Orientation.VERTICAL, Orientation.HORIZONTAL
    hero = photo("hero", score=0.9, scene=SceneType.LANDSCAPE)

    assert planner.choose_template([hero]) is LayoutTemplate.HERO_SINGLE
    assert (
        planner.choose_template([photo("a", orientation=v), photo("b", orientation=v)])
        is LayoutTemplate.DUO_VERTICAL
    )
    assert (
        planner.choose_template([photo("a", orientation=h), photo("b", orientation=h)])
        is LayoutTemplate.DUO_HORIZONTAL
    )

    run = [photo(f"p{i}", score=0.5 + i / 100) for i in range(5)]
    pages = planner.plan_pages([hero, *run], config, max_photos_per_page=4)
    assert [p.template for p in pages] == [
        LayoutTemplate.HERO_SINGLE,
        LayoutTemplate.TRIO_FEATURE,
        LayoutTemplate.DUO_HORIZONTAL,
    ]
    assert pages[1].photo_ids[0] == "p2"  # highest score takes the dominant slot

    quad = planner.plan_pages(run[:4], config, max_photos_per_page=4)
    assert [p.template for p in quad] == [LayoutTemplate.QUAD_GRID]

    capped = planner.plan_pages(run, config, max_photos_per_page=2)
    assert [len(p.photo_ids) for p in capped] == [2, 2, 1]
    assert sorted(pid for p in capped for pid in p.photo_ids) == sorted(p.photo_id for p in run)


def test_recalculate_page_keeps_valid_manual_template() -> None:
    planner = SmartLayoutPlanner()
    photos = {f"p{i}": photo(f"p{i}") for i in range(4)}
    page = planner.plan_pages(list(photos.values()), AnalysisConfig(), max_photos_per_page=4)[0]

    page.photo_ids.pop()
    planner.recalculate_page(page, photos)
    assert page.template is LayoutTemplate.TRIO_FEATURE

    page.template, page.template_locked = LayoutTemplate.QUAD_GRID, True
    page.photo_ids.pop()
    planner.recalculate_page(page, photos)
    assert page.template is LayoutTemplate.QUAD_GRID

    page.template = LayoutTemplate.HERO_SINGLE
    planner.recalculate_page(page, photos)
    assert page.template is LayoutTemplate.DUO_HORIZONTAL and not page.template_locked


# ------------------------------------------------------- state store, exporter


def test_state_round_trip_and_diff(tmp_path: Path, make_photo) -> None:
    make_photo("a.jpg", seed=1)
    make_photo("sub/b.jpg", seed=2)
    store, analyzer = JsonStateStore(), OpenCvImageAnalyzer()
    state = ProjectState(root=tmp_path)
    for rel in store.discover(tmp_path, set()):
        item = analyzer.analyze(tmp_path, rel)
        state.photos[item.photo_id] = item
    store.save(state)

    loaded = store.load(tmp_path)
    assert loaded is not None
    assert loaded.to_dict()["photos"] == state.to_dict()["photos"]

    (tmp_path / "sub" / "b.jpg").rename(tmp_path / "b.jpg")
    make_photo("c.jpg", seed=3)
    make_photo("a.jpg", seed=9)
    diff = store.diff(loaded, store.discover(tmp_path, set()))
    assert diff.new == ["c.jpg"] and diff.modified == ["a.jpg"]
    assert list(diff.moved.values()) == ["b.jpg"] and not diff.removed


def test_corrupt_state_is_set_aside(tmp_path: Path) -> None:
    store = JsonStateStore()
    store.state_path(tmp_path).write_text("{not json", encoding="utf-8")
    assert store.load(tmp_path) is None
    assert (tmp_path / ".photoseder_state.json.corrupt").exists()


def test_export_embedded_and_linked(tmp_path: Path, make_photo) -> None:
    make_photo("a b.jpg", seed=1)
    item = OpenCvImageAnalyzer().analyze(tmp_path, "a b.jpg")
    photos = {item.photo_id: item}
    pages = SmartLayoutPlanner().plan_pages([item], AnalysisConfig(), max_photos_per_page=4)
    exporter = HtmlAlbumExporter()

    embedded = exporter.render(pages, photos, tmp_path, ExportOptions(title="<Trip> טיול"))
    assert "data:image/jpeg;base64," in embedded
    assert "&lt;Trip&gt; טיול" in embedded and "<Trip>" not in embedded
    assert "http://" not in embedded and "https://" not in embedded
    assert "size: 297mm 210mm" in embedded

    linked = exporter.render(
        pages, photos, tmp_path, ExportOptions(embed_images=False), output_dir=tmp_path / "out"
    )
    assert 'src="../a%20b.jpg"' in linked
