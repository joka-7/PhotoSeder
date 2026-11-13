"""Perceptual hashing, burst deduplication and chronological event clustering.

Algorithms:
    **dHash (difference hash).** The grayscale image is shrunk to 9x8 pixels
    with area averaging, which discards detail and keeps the coarse luminance
    structure. For each of the 8 rows, every pixel is compared with its right
    neighbour: bit ``i`` is 1 when ``P[y, x + 1] > P[y, x]``. The 8 x 8 = 64
    comparisons form a 64-bit integer that encodes the *horizontal gradient
    direction* of the image. Small changes in exposure, compression or framing
    flip only a few bits, so near-identical frames have a small Hamming
    distance ``popcount(a XOR b)``.

    **Burst culling.** Photos are sorted by capture time and chained: a photo
    joins the current burst when it was taken within ``burst_window_seconds``
    of the previous photo *and* their hashes differ by at most
    ``burst_hamming_distance`` bits. Chaining against the previous member
    (rather than the first) follows slow drifts across a long burst. The
    keeper of each burst maximizes ``(passes quality, overall_score,
    eyes_open)``; the rest are rejected as duplicates.

    **Event clustering.** Walking the same chronological order, a new cluster
    starts when the gap to the previous photo is at least ``event_gap_hours``
    or the calendar date changes. Clusters are titled
    ``YYYY-MM-DD_<Morning|Afternoon|Evening|Night>`` from their first photo,
    with a numeric suffix when two clusters would share a title.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

import cv2
import numpy as np

from photoseder.domain.enums import RejectionCategory
from photoseder.domain.models import AnalysisConfig, EventCluster, PhotoItem

BURST_REJECTION_REASON = "Burst duplicate (kept sharper/better version)"

_HASH_WIDTH = 9
_HASH_HEIGHT = 8


def compute_dhash(gray: np.ndarray) -> int:
    """Compute the 64-bit horizontal difference hash of a grayscale image.

    Args:
        gray: Single-channel ``uint8`` image of any size.

    Returns:
        A 64-bit integer; bit order is row-major, most significant bit first.
    """
    small = cv2.resize(gray, (_HASH_WIDTH, _HASH_HEIGHT), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """Number of differing bits between two hashes."""
    return (hash_a ^ hash_b).bit_count()


def day_part(moment: datetime) -> str:
    """Name the part of the day a timestamp falls in."""
    hour = moment.hour
    if 5 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 22:
        return "Evening"
    return "Night"


def _chronological(photos: Sequence[PhotoItem]) -> list[PhotoItem]:
    """Sort photos by capture time, breaking ties by path for determinism."""
    return sorted(photos, key=lambda p: (p.captured_at, p.rel_path))


class BurstDeduplicator:
    """dHash-based burst culling and time-gap event clustering."""

    def group_bursts(
        self, photos: Sequence[PhotoItem], config: AnalysisConfig
    ) -> list[list[PhotoItem]]:
        """Group consecutive near-identical photos.

        Args:
            photos: Photos in any order.
            config: Supplies the burst window and Hamming threshold.

        Returns:
            Chronological groups; a photo with no near-duplicate neighbour
            forms a group of one.
        """
        window = timedelta(seconds=config.burst_window_seconds)
        groups: list[list[PhotoItem]] = []
        for photo in _chronological(photos):
            if groups:
                previous = groups[-1][-1]
                close_in_time = photo.captured_at - previous.captured_at <= window
                similar = (
                    hamming_distance(photo.dhash, previous.dhash) <= config.burst_hamming_distance
                )
                if close_in_time and similar:
                    groups[-1].append(photo)
                    continue
            groups.append([photo])
        return groups

    def mark_bursts(self, photos: Sequence[PhotoItem], config: AnalysisConfig) -> int:
        """Keep the best photo of every burst and reject the others.

        The method is idempotent provided automatic quality decisions were
        refreshed first (see ``Analyzer.rescore``): it clears every previous
        ``burst_id`` and only rejects photos that are currently selected and
        not locked by the user.

        Args:
            photos: All photos of the project.
            config: Supplies the burst window and Hamming threshold.

        Returns:
            Number of photos rejected as burst duplicates.
        """
        for photo in photos:
            photo.burst_id = None

        rejected = 0
        for group in self.group_bursts(photos, config):
            if len(group) < 2:
                continue
            burst_id = f"burst_{group[0].photo_id}"
            keeper = max(
                group,
                key=lambda p: (
                    p.is_selected,
                    p.metrics.overall_score,
                    p.metrics.eyes_open,
                    # Earlier frame wins a perfect tie (deterministic).
                    -p.captured_at.timestamp(),
                ),
            )
            for photo in group:
                photo.burst_id = burst_id
                if photo is keeper or photo.selection_locked or not photo.is_selected:
                    continue
                photo.is_selected = False
                photo.rejection_category = RejectionCategory.DUPLICATE
                photo.rejection_reason = BURST_REJECTION_REASON
                rejected += 1
        return rejected

    def cluster_events(
        self, photos: Sequence[PhotoItem], config: AnalysisConfig
    ) -> list[EventCluster]:
        """Partition photos into chronological chapters.

        Every photo (keeper or not) is assigned a ``cluster_id`` so a photo
        restored later already belongs to a chapter.

        Args:
            photos: All photos of the project.
            config: Supplies ``event_gap_hours``.

        Returns:
            Clusters in chronological order.
        """
        gap = timedelta(hours=config.event_gap_hours)
        clusters: list[EventCluster] = []
        used_titles: dict[str, int] = {}
        previous: PhotoItem | None = None

        for photo in _chronological(photos):
            starts_new = (
                previous is None
                or photo.captured_at - previous.captured_at >= gap
                or photo.captured_at.date() != previous.captured_at.date()
            )
            if starts_new:
                base = f"{photo.captured_at:%Y-%m-%d}_{day_part(photo.captured_at)}"
                used_titles[base] = used_titles.get(base, 0) + 1
                count = used_titles[base]
                title = base if count == 1 else f"{base}_{count}"
                clusters.append(
                    EventCluster(
                        cluster_id=title,
                        title=title,
                        start=photo.captured_at,
                        end=photo.captured_at,
                    )
                )
            cluster = clusters[-1]
            cluster.end = photo.captured_at
            cluster.photo_ids.append(photo.photo_id)
            photo.cluster_id = cluster.cluster_id
            previous = photo
        return clusters
