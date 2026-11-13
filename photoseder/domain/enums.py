"""Enumerations shared by every PhotoSeder layer.

All enums derive from :class:`enum.StrEnum`, so members serialize to JSON as
plain strings and round-trip through ``EnumClass(value)`` without any custom
encoder. The domain layer has no dependency on services, UI or third-party
libraries.
"""

from __future__ import annotations

from enum import StrEnum


class SceneType(StrEnum):
    """Semantic category assigned to a photo by the analyzer (or the user)."""

    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"
    GROUP = "group"
    GENERAL = "general"

    @property
    def folder_name(self) -> str:
        """Folder used by the ``BY_SCENE`` organization strategy."""
        return {
            SceneType.LANDSCAPE: "Landscapes",
            SceneType.PORTRAIT: "Portraits",
            SceneType.GROUP: "Groups",
            SceneType.GENERAL: "General",
        }[self]

    @property
    def label(self) -> str:
        """Human-readable label, also used as an EXIF keyword."""
        return self.value.capitalize()


class Orientation(StrEnum):
    """Display orientation after the EXIF orientation tag has been applied."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    SQUARE = "square"


class LayoutTemplate(StrEnum):
    """Album page templates rendered as CSS Grid layouts by the exporter."""

    HERO_SINGLE = "hero_single"
    DUO_VERTICAL = "duo_vertical"
    DUO_HORIZONTAL = "duo_horizontal"
    TRIO_FEATURE = "trio_feature"
    QUAD_GRID = "quad_grid"

    @property
    def capacity(self) -> int:
        """Maximum number of photos the template can display."""
        return {
            LayoutTemplate.HERO_SINGLE: 1,
            LayoutTemplate.DUO_VERTICAL: 2,
            LayoutTemplate.DUO_HORIZONTAL: 2,
            LayoutTemplate.TRIO_FEATURE: 3,
            LayoutTemplate.QUAD_GRID: 4,
        }[self]

    @property
    def label(self) -> str:
        """Human-readable label for UI dropdowns."""
        return self.value.replace("_", " ").title()


class OrganizationStrategy(StrEnum):
    """Physical folder organization modes supported by the organizer."""

    CULL_REJECTED = "cull_rejected"
    BY_EVENT = "by_event"
    BY_SCENE = "by_scene"
    BEST_OF_HEROES = "best_of_heroes"


class FileActionMode(StrEnum):
    """How a file reaches its destination during an organization batch."""

    MOVE = "move"
    COPY = "copy"
    HARDLINK = "hardlink"


class RejectionCategory(StrEnum):
    """Why a photo is not a keeper; drives dashboard counts and cull folders."""

    NONE = "none"
    QUALITY = "quality"
    DUPLICATE = "duplicate"
    MANUAL = "manual"


class PageSize(StrEnum):
    """Print formats supported by the HTML album exporter."""

    A4_LANDSCAPE = "a4_landscape"
    SQUARE = "square"

    @property
    def dimensions_mm(self) -> tuple[int, int]:
        """Page ``(width, height)`` in millimetres."""
        return {
            PageSize.A4_LANDSCAPE: (297, 210),
            PageSize.SQUARE: (210, 210),
        }[self]

    @property
    def label(self) -> str:
        """Human-readable label for UI dropdowns."""
        return {
            PageSize.A4_LANDSCAPE: "A4 landscape (297 x 210 mm)",
            PageSize.SQUARE: "Square (210 x 210 mm)",
        }[self]
