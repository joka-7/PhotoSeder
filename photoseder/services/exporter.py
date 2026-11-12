"""Print-ready HTML5 / CSS Grid album renderer (Jinja2, no remote assets).

Design:
    Each :class:`AlbumPage` becomes one fixed-size ``<section class="page">``
    whose photo area is a CSS Grid chosen by the page's
    :class:`LayoutTemplate`:

    ============== =========================== ==========================
    Template       Grid (rows / columns)       Notes
    ============== =========================== ==========================
    hero_single    ``1fr / 1fr``               one full-bleed cell
    duo_vertical   ``1fr / 1fr 1fr``           side by side
    duo_horizontal ``1fr 1fr / 1fr``           stacked
    trio_feature   ``1fr 1fr / 2fr 1fr``       first cell spans both rows
    quad_grid      ``1fr 1fr / 1fr 1fr``       2 x 2
    ============== =========================== ==========================

    All lengths derive from one CSS variable, ``--u`` ("one page
    millimetre"). In print ``--u`` is exactly ``1mm`` and ``@page`` sets the
    sheet to the album format with zero margins, so one section fills one
    sheet. On screen ``--u`` shrinks to fit the viewport, which makes the same
    document usable as a browser preview and as the embedded page preview in
    the Streamlit studio.

Privacy:
    The document references no fonts, scripts or stylesheets from the
    network. With ``embed_images`` every photo is inlined as a base64 JPEG,
    so the single file is fully self-contained; otherwise photos are linked
    by relative path.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from jinja2 import Environment, select_autoescape

from photoseder.domain.models import AlbumPage, PhotoItem
from photoseder.services.interfaces import ExportOptions
from photoseder.services.thumbnails import encode_jpeg, jpeg_data_uri, load_upright_rgb

type ImageSource = Callable[[PhotoItem], str]
"""Maps a photo to the value of its ``<img src>`` attribute."""

_EMBED_JPEG_QUALITY = 90

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<style>
  @page { size: {{ page_w }}mm {{ page_h }}mm; margin: 0; }
  :root {
    --u: min(1mm, calc({{ fit_w }}vw / {{ page_w }}), calc({{ fit_h }}vh / {{ page_h }}));
    --paper: #ffffff;
    --ink: #1f2328;
    --muted: #6b7280;
    --desk: {{ desk_color }};
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--desk); color: var(--ink);
    font-family: "Segoe UI", "Helvetica Neue", Arial, "Noto Sans Hebrew", sans-serif;
    -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .page { position: relative; display: flex; flex-direction: column; overflow: hidden;
    width: calc({{ page_w }} * var(--u)); height: calc({{ page_h }} * var(--u));
    padding: calc(10 * var(--u)); margin: calc({{ page_gap }} * var(--u)) auto;
    background: var(--paper);
    box-shadow: 0 calc(1 * var(--u)) calc(6 * var(--u)) rgba(0, 0, 0, .35);
    break-after: page; page-break-after: always; break-inside: avoid; }
  .chapter { margin: 0 0 calc(4 * var(--u)); text-align: center; font-weight: 600;
    font-size: calc(4.6 * var(--u)); letter-spacing: .06em; color: var(--muted); }
  .grid { flex: 1; min-height: 0; display: grid; gap: calc(4 * var(--u)); }
  .hero_single { grid-template: 1fr / 1fr; }
  .duo_vertical { grid-template: 1fr / 1fr 1fr; }
  .duo_horizontal { grid-template: 1fr 1fr / 1fr; }
  .trio_feature { grid-template: 1fr 1fr / 2fr 1fr; }
  .trio_feature .cell:first-child { grid-row: 1 / span 2; }
  .quad_grid { grid-template: 1fr 1fr / 1fr 1fr; }
  .cell { position: relative; min-width: 0; min-height: 0; overflow: hidden; }
  .cell img { position: absolute; inset: 0; width: 100%; height: 100%; display: block;
    object-fit: {{ 'cover' if crop_to_fill else 'contain' }}; object-position: center 35%; }
  .cell.empty { border: calc(.4 * var(--u)) dashed #c9ced6; }
  .folio { position: absolute; left: 0; right: 0; bottom: calc(3.2 * var(--u));
    text-align: center; font-size: calc(2.6 * var(--u)); color: var(--muted); }
  .cover { align-items: center; justify-content: center; text-align: center; }
  .cover h1 { margin: 0; font-size: calc(15 * var(--u)); font-weight: 300; letter-spacing: .02em; }
  .cover .rule { width: calc(40 * var(--u)); height: calc(.5 * var(--u));
    margin: calc(8 * var(--u)) auto; background: var(--ink); }
  .cover p { margin: 0; font-size: calc(4.4 * var(--u)); color: var(--muted); }
  @media print {
    :root { --u: 1mm; }
    html, body { background: #fff; }
    .page { margin: 0; box-shadow: none; }
    .page:last-child { break-after: auto; page-break-after: auto; }
  }
</style>
</head>
<body>
{% if cover %}
<section class="page cover">
  <h1 dir="auto">{{ title }}</h1>
  <div class="rule"></div>
  <p dir="auto">{{ cover.subtitle }}</p>
</section>
{% endif %}
{% for page in pages %}
<section class="page">
  {% if page.chapter_title %}<h2 class="chapter" dir="auto">{{ page.chapter_title }}</h2>{% endif %}
  <div class="grid {{ page.template }}">
    {% for cell in page.cells %}
    {% if cell.src %}<div class="cell"><img src="{{ cell.src }}" alt="{{ cell.alt }}"></div>
    {% else %}<div class="cell empty"></div>{% endif %}
    {% endfor %}
  </div>
  {% if page.number %}<div class="folio">{{ page.number }}</div>{% endif %}
</section>
{% endfor %}
</body>
</html>
"""


class HtmlAlbumExporter:
    """Renders album pages to a single print-ready HTML document."""

    def __init__(self) -> None:
        """Compile the embedded Jinja2 template (autoescaping enabled)."""
        environment = Environment(
            autoescape=select_autoescape(default=True, default_for_string=True),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._template = environment.from_string(_TEMPLATE)

    def render(
        self,
        pages: Sequence[AlbumPage],
        photos_by_id: dict[str, PhotoItem],
        root: Path,
        options: ExportOptions,
        *,
        output_dir: Path | None = None,
    ) -> str:
        """Return the album as an HTML document string.

        Args:
            pages: Pages in print order.
            photos_by_id: Lookup for the photos referenced by the pages.
            root: Project folder the photos live in.
            options: Title, format and image handling.
            output_dir: Folder the HTML will be saved in; linked image paths
                are made relative to it (defaults to ``root``).

        Returns:
            A complete HTML5 document.
        """
        if options.embed_images:
            source = self._embedded_source(root, options.max_edge_px)
        else:
            source = self._linked_source(root, output_dir or root)
        return self._render(pages, photos_by_id, options, source, standalone=True)

    def render_preview(
        self,
        page: AlbumPage,
        photos_by_id: dict[str, PhotoItem],
        options: ExportOptions,
        image_source: ImageSource,
    ) -> str:
        """Render one page for on-screen preview (no cover, no page gap).

        Args:
            page: Page to preview.
            photos_by_id: Lookup for the page's photos.
            options: Supplies the page format and crop mode.
            image_source: Returns a lightweight ``src`` (e.g. a thumbnail).

        Returns:
            A complete HTML5 document sized to fit its viewport.
        """
        return self._render([page], photos_by_id, options, image_source, standalone=False)

    def export(
        self,
        pages: Sequence[AlbumPage],
        photos_by_id: dict[str, PhotoItem],
        root: Path,
        options: ExportOptions,
        output_path: Path,
    ) -> Path:
        """Write the album to ``output_path``.

        Args:
            pages: Pages in print order.
            photos_by_id: Lookup for the photos referenced by the pages.
            root: Project folder the photos live in.
            options: Title, format and image handling.
            output_path: Destination ``.html`` file (parents are created).

        Returns:
            ``output_path``.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        document = self.render(pages, photos_by_id, root, options, output_dir=output_path.parent)
        output_path.write_text(document, encoding="utf-8")
        return output_path

    def _render(
        self,
        pages: Sequence[AlbumPage],
        photos_by_id: dict[str, PhotoItem],
        options: ExportOptions,
        image_source: ImageSource,
        *,
        standalone: bool,
    ) -> str:
        """Build the template context and render it."""
        page_w, page_h = options.page_size.dimensions_mm
        used: list[PhotoItem] = []
        rendered_pages: list[dict[str, object]] = []

        for number, page in enumerate(pages, start=1):
            cells: list[dict[str, str]] = []
            for photo_id in page.photo_ids[: page.template.capacity]:
                photo = photos_by_id.get(photo_id)
                if photo is None:
                    continue
                used.append(photo)
                cells.append({"src": image_source(photo), "alt": photo.name})
            while len(cells) < page.template.capacity:
                cells.append({"src": "", "alt": ""})
            rendered_pages.append(
                {
                    "template": page.template.value,
                    "chapter_title": page.chapter_title,
                    "cells": cells,
                    "number": number if standalone else None,
                }
            )

        cover = None
        if standalone and options.include_cover:
            cover = {"subtitle": self._cover_subtitle(used)}
        return self._template.render(
            title=options.title,
            page_w=page_w,
            page_h=page_h,
            fit_w=94 if standalone else 100,
            fit_h=10_000 if standalone else 100,
            page_gap=8 if standalone else 0,
            desk_color="#3b3f45" if standalone else "transparent",
            crop_to_fill=options.crop_to_fill,
            cover=cover,
            pages=rendered_pages,
        )

    @staticmethod
    def _cover_subtitle(photos: Sequence[PhotoItem]) -> str:
        """Describe the album on the cover: date range and photo count."""
        if not photos:
            return "Empty album"
        first: datetime = min(p.captured_at for p in photos)
        last: datetime = max(p.captured_at for p in photos)
        if first.date() == last.date():
            dates = f"{first:%d %B %Y}"
        else:
            dates = f"{first:%d %B %Y} – {last:%d %B %Y}"
        noun = "photo" if len(photos) == 1 else "photos"
        return f"{dates} · {len(photos)} {noun}"

    @staticmethod
    def _embedded_source(root: Path, max_edge_px: int) -> ImageSource:
        """Image source that inlines a resized base64 JPEG."""

        def source(photo: PhotoItem) -> str:
            try:
                image = load_upright_rgb(photo.absolute_path(root), max_edge_px)
            except (OSError, ValueError):
                return ""
            return jpeg_data_uri(encode_jpeg(image, _EMBED_JPEG_QUALITY))

        return source

    @staticmethod
    def _linked_source(root: Path, output_dir: Path) -> ImageSource:
        """Image source that links the original file by relative URL."""

        def source(photo: PhotoItem) -> str:
            absolute = photo.absolute_path(root)
            try:
                relative = os.path.relpath(absolute, output_dir)
            except ValueError:
                # Different drive on Windows: fall back to a file URI.
                return absolute.as_uri()
            return quote(Path(relative).as_posix())

        return source
