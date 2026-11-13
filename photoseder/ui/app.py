"""PhotoSeder — Streamlit application.

Run with::

    streamlit run photoseder/ui/app.py

Architecture:
    The UI is a thin layer over :class:`photoseder.orchestrator.pipeline.Pipeline`.
    One pipeline per opened folder lives in ``st.session_state``; every widget
    mutates state through an ``on_click`` / ``on_change`` callback that calls a
    pipeline method, after which Streamlit reruns the script and redraws from
    the (already saved) project state. The UI holds no business logic and
    never touches image files except through the thumbnail cache.

    Everything runs on the local machine. The app makes no network requests;
    Streamlit's usage telemetry is disabled in ``.streamlit/config.toml``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # Allows running without `pip install -e .`
    sys.path.insert(0, str(_REPO_ROOT))

from photoseder.domain.enums import (  # noqa: E402
    FileActionMode,
    LayoutTemplate,
    OrganizationStrategy,
    PageSize,
    RejectionCategory,
    SceneType,
)
from photoseder.domain.models import AnalysisConfig, PhotoItem  # noqa: E402
from photoseder.orchestrator.pipeline import (  # noqa: E402
    DEFAULT_ALBUM_FILENAME,
    Pipeline,
)
from photoseder.services.interfaces import ExportOptions  # noqa: E402
from photoseder.services.layout_planner import (  # noqa: E402
    MAX_PHOTOS_PER_PAGE,
    compatible_templates,
)
from photoseder.services.thumbnails import ThumbnailCache  # noqa: E402

GALLERY_COLUMNS = 4
GALLERY_PAGE_SIZE = 24

FILTER_ALL = "All photos"
FILTER_KEEPERS = "Keepers"
FILTER_HEROES = "5-star heroes"
FILTER_QUALITY = "Rejected: blurry / exposure"
FILTER_DUPLICATES = "Rejected: burst duplicates"
FILTER_MANUAL = "Rejected: manually"
GALLERY_FILTERS = (
    FILTER_ALL,
    FILTER_KEEPERS,
    FILTER_HEROES,
    FILTER_QUALITY,
    FILTER_DUPLICATES,
    FILTER_MANUAL,
)

SORT_TIME = "Capture time"
SORT_BEST = "Score: best first"
SORT_WORST = "Score: worst first"

STRATEGY_HELP: dict[OrganizationStrategy, tuple[str, str]] = {
    OrganizationStrategy.CULL_REJECTED: (
        "Cull folder",
        "Move rejected photos into `_Rejected_Quality/` and "
        "`_Rejected_Duplicates/`, leaving only keepers in place.",
    ),
    OrganizationStrategy.BY_EVENT: (
        "Organize by event / date",
        "Sort keepers into one subfolder per event, e.g. `2026-10-15_Morning/`.",
    ),
    OrganizationStrategy.BY_SCENE: (
        "Organize by scene",
        "Sort keepers into `Landscapes/`, `Portraits/`, `Groups/` and `General/`.",
    ),
    OrganizationStrategy.BEST_OF_HEROES: (
        "Curate Best-Of",
        "Collect 4-star and 5-star keepers in `Best_Of/`.",
    ),
}

MODE_HELP: dict[FileActionMode, str] = {
    FileActionMode.MOVE: "Move — files leave their current folder",
    FileActionMode.COPY: "Copy — originals stay, uses extra disk space",
    FileActionMode.HARDLINK: "Hard link — originals stay, no extra disk space",
}


# --------------------------------------------------------------------- state


def pipeline() -> Pipeline | None:
    """Return the pipeline of the currently opened folder, if any."""
    return st.session_state.get("pipeline")


def thumbnails() -> ThumbnailCache:
    """Return the thumbnail cache of the currently opened folder."""
    return st.session_state["thumbnails"]


def flash(message: str, kind: str = "success") -> None:
    """Queue a message to show at the top of the next rerun."""
    st.session_state.setdefault("flash", []).append((kind, message))


def show_flashes() -> None:
    """Display and clear queued messages.

    The messages are drawn inside a container that is created on every run,
    whether or not there is anything to show. That keeps the position of the
    elements below it (notably the tab bar) stable, so Streamlit does not
    reset the active tab when a message appears or disappears.
    """
    area = st.container()
    for kind, message in st.session_state.pop("flash", []):
        getattr(area, kind)(message)


def stars_text(count: int) -> str:
    """Render a star rating such as ``★★★★☆``."""
    return "\u2605" * count + "\u2606" * (5 - count)


def relative(path: str, root: Path) -> str:
    """Show an absolute path relative to the project folder when possible."""
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return path


# ------------------------------------------------------------------- sidebar


def open_folder(folder_text: str) -> None:
    """Open a folder, load its cache and analyze new or changed photos."""
    folder_text = folder_text.strip().strip('"')
    if not folder_text:
        st.sidebar.error("Enter the path of a folder that contains photos.")
        return
    try:
        pipe = Pipeline(Path(folder_text))
    except (NotADirectoryError, OSError) as exc:
        st.sidebar.error(f"Cannot open folder: {exc}")
        return

    st.session_state["pipeline"] = pipe
    st.session_state["thumbnails"] = ThumbnailCache(pipe.root)
    for key in ("preview", "page_number", "gallery_page"):
        st.session_state.pop(key, None)
    run_scan(pipe)


def run_scan(pipe: Pipeline) -> None:
    """Scan the folder with a progress bar and report the outcome."""
    bar = st.sidebar.progress(0.0, text="Looking for photos...")

    def report(done: int, total: int, name: str) -> None:
        bar.progress(done / total, text=f"Analyzing {done}/{total}: {name}")

    result = pipe.scan_and_analyze(report)
    bar.empty()
    parts = [f"{result.analyzed} analyzed", f"{result.cached} loaded from cache"]
    if result.moved:
        parts.append(f"{result.moved} re-linked")
    if result.removed:
        parts.append(f"{result.removed} no longer on disk")
    flash("Scan complete: " + ", ".join(parts) + ".")
    if result.errors:
        flash(f"{len(result.errors)} file(s) could not be read (see tab 1).", "warning")


def render_sidebar() -> None:
    """Folder selection and analysis thresholds."""
    st.sidebar.title("PhotoSeder")
    st.sidebar.caption("Local-first photo curation. Nothing leaves this computer.")

    pipe = pipeline()
    folder_text = st.sidebar.text_input(
        "Photo folder",
        value=str(pipe.root) if pipe else "",
        placeholder=r"e.g. C:\Photos\Trip or /home/me/Pictures/Trip",
        help="Paste the full path. Subfolders are included; Unicode names work.",
    )
    label = "Rescan folder" if pipe and str(pipe.root) == folder_text.strip() else "Open & scan"
    if st.sidebar.button(label, type="primary", width="stretch"):
        open_folder(folder_text)
        st.rerun()

    pipe = pipeline()
    if pipe is None:
        return

    config = pipe.state.config
    with st.sidebar.form("thresholds"):
        st.subheader("Quality thresholds")
        blur = st.slider(
            "Blur sensitivity",
            20,
            400,
            int(config.blur_threshold),
            step=10,
            help="Laplacian-variance threshold. Higher = stricter: more photos count as blurry.",
        )
        reject = st.slider(
            "Reject below overall score",
            0.20,
            0.70,
            float(config.reject_threshold),
            step=0.01,
        )
        exposure_gate = st.slider(
            "Reject below exposure score",
            0.10,
            0.70,
            float(config.exposure_gate),
            step=0.01,
            help="Rejects badly over/under-exposed photos even when sharp.",
        )
        hero = st.slider(
            "Hero page from score",
            0.60,
            0.95,
            float(config.hero_threshold),
            step=0.01,
            help="Landscapes and portraits at or above this get a full page.",
        )
        st.subheader("Bursts and events")
        window = st.slider("Burst window (seconds)", 1, 60, int(config.burst_window_seconds))
        hamming = st.slider(
            "Burst similarity (max hash distance)",
            0,
            16,
            int(config.burst_hamming_distance),
            help="Bits (of 64) by which two shots may differ and still count as the same burst.",
        )
        gap = st.slider(
            "New event after a gap of (hours)",
            0.5,
            12.0,
            float(config.event_gap_hours),
            step=0.5,
        )
        if st.form_submit_button("Apply thresholds", width="stretch"):
            pipe.update_config(
                AnalysisConfig(
                    blur_threshold=float(blur),
                    reject_threshold=reject,
                    exposure_gate=exposure_gate,
                    resolution_target_mp=config.resolution_target_mp,
                    portrait_face_ratio=config.portrait_face_ratio,
                    landscape_min_sharpness=config.landscape_min_sharpness,
                    burst_window_seconds=float(window),
                    burst_hamming_distance=hamming,
                    event_gap_hours=gap,
                    hero_threshold=hero,
                )
            )
            flash("Thresholds applied. Manual decisions were kept.")
            st.rerun()
    st.sidebar.caption(
        "Changing thresholds re-scores instantly from cached measurements; no photo is read again."
    )


# --------------------------------------------------------------- tab 1: cull


def on_toggle_selected(photo_id: str, selected: bool) -> None:
    """Callback: restore or reject a photo."""
    pipeline().set_selected(photo_id, selected)


def on_scene_change(photo_id: str, key: str) -> None:
    """Callback: override the scene type."""
    pipeline().set_scene(photo_id, SceneType(st.session_state[key]))


def on_rating_change(photo_id: str, key: str) -> None:
    """Callback: override the star rating."""
    pipeline().set_rating(photo_id, int(st.session_state[key]))


def on_reset_overrides(photo_id: str) -> None:
    """Callback: drop manual decisions for a photo."""
    pipeline().reset_overrides(photo_id)


def filter_photos(photos: list[PhotoItem], status: str, scenes: list[str]) -> list[PhotoItem]:
    """Apply the gallery's status and scene filters."""
    predicates = {
        FILTER_ALL: lambda p: True,
        FILTER_KEEPERS: lambda p: p.is_selected,
        FILTER_HEROES: lambda p: p.is_selected and p.star_rating == 5,
        FILTER_QUALITY: lambda p: p.rejection_category is RejectionCategory.QUALITY,
        FILTER_DUPLICATES: lambda p: p.rejection_category is RejectionCategory.DUPLICATE,
        FILTER_MANUAL: lambda p: p.rejection_category is RejectionCategory.MANUAL,
    }
    matches = predicates[status]
    return [
        photo for photo in photos if matches(photo) and (not scenes or photo.scene.value in scenes)
    ]


def render_photo_card(photo: PhotoItem) -> None:
    """One gallery card: thumbnail, facts and the curation controls."""
    with st.container(border=True):
        thumb = thumbnails().path_for(photo)
        if thumb is not None:
            st.image(str(thumb), width="stretch")
        else:
            st.warning("Preview unavailable")

        metrics = photo.metrics
        st.markdown(f"**{photo.name}**")
        st.caption(
            f"{stars_text(photo.star_rating)} · score {metrics.overall_score:.2f} · "
            f"{photo.captured_at:%Y-%m-%d %H:%M}"
        )
        if photo.is_selected:
            st.caption(
                ":green[Keeper]" + (" (restored manually)" if photo.selection_locked else "")
            )
        else:
            st.caption(f":red[{photo.rejection_reason or 'Rejected'}]")

        st.button(
            "Reject" if photo.is_selected else "Restore to keepers",
            key=f"toggle_{photo.photo_id}",
            on_click=on_toggle_selected,
            args=(photo.photo_id, not photo.is_selected),
            type="secondary" if photo.is_selected else "primary",
            width="stretch",
        )

        scene_key, rating_key = f"scene_{photo.photo_id}", f"rating_{photo.photo_id}"
        st.session_state[scene_key] = photo.scene.value
        st.session_state[rating_key] = photo.star_rating
        left, right = st.columns(2)
        left.selectbox(
            "Scene",
            [scene.value for scene in SceneType],
            key=scene_key,
            format_func=str.capitalize,
            on_change=on_scene_change,
            args=(photo.photo_id, scene_key),
        )
        right.selectbox(
            "Stars",
            [5, 4, 3, 2, 1],
            key=rating_key,
            on_change=on_rating_change,
            args=(photo.photo_id, rating_key),
        )

        with st.popover("Details", width="stretch"):
            st.markdown(
                f"`{photo.rel_path}`  \n"
                f"{photo.width} x {photo.height} px ({photo.megapixels:.1f} MP), "
                f"{photo.orientation.value}"
            )
            st.markdown(
                f"- Sharpness **{metrics.sharpness:.2f}** "
                f"(Laplacian variance {metrics.laplacian_variance:.0f})\n"
                f"- Exposure / contrast **{metrics.exposure:.2f}** "
                f"(dark clip {metrics.clipped_low:.1%}, bright clip {metrics.clipped_high:.1%})\n"
                f"- Resolution **{metrics.resolution:.2f}**\n"
                f"- Faces **{metrics.face_count}** "
                f"({metrics.face_area_ratio:.1%} of frame), eyes open **{metrics.eyes_open:.2f}**\n"
                f"- Event `{photo.cluster_id}`"
                + (f"\n- Part of burst `{photo.burst_id}`" if photo.burst_id else "")
            )
            locked = photo.scene_locked or photo.rating_locked or photo.selection_locked
            st.button(
                "Reset to automatic",
                key=f"reset_{photo.photo_id}",
                on_click=on_reset_overrides,
                args=(photo.photo_id,),
                disabled=not locked,
            )


def render_cull_tab(pipe: Pipeline) -> None:
    """Tab 1: dashboard metrics and the interactive curation gallery."""
    stats = pipe.stats()
    cols = st.columns(5)
    cols[0].metric("Total photos", stats.total)
    cols[1].metric("Keepers", stats.keepers)
    cols[2].metric("Rejected (blurry / exposure)", stats.rejected_quality)
    cols[3].metric("Rejected (duplicates)", stats.rejected_duplicates)
    cols[4].metric("5-star heroes", stats.heroes)

    scene_cols = st.columns(5)
    for column, scene in zip(scene_cols, SceneType, strict=False):
        column.metric(f"{scene.folder_name}", stats.scenes.get(scene, 0))
    scene_cols[4].metric("Rejected (manually)", stats.rejected_manual)
    st.caption("Scene counts cover keepers only.")

    if pipe.state.unreadable:
        with st.expander(f"{len(pipe.state.unreadable)} file(s) could not be read"):
            for rel_path, reason in sorted(pipe.state.unreadable.items()):
                st.markdown(f"- `{rel_path}` — {reason}")

    st.divider()
    filter_cols = st.columns([2, 2, 2, 1])
    status = filter_cols[0].selectbox("Show", GALLERY_FILTERS, key="gallery_status")
    scenes = filter_cols[1].multiselect(
        "Scene",
        [scene.value for scene in SceneType],
        format_func=str.capitalize,
        key="gallery_scenes",
        placeholder="All scenes",
    )
    order = filter_cols[2].selectbox(
        "Sort by", (SORT_TIME, SORT_BEST, SORT_WORST), key="gallery_sort"
    )

    photos = filter_photos(pipe.state.chronological(), status, scenes)
    if order != SORT_TIME:
        photos.sort(key=lambda p: p.metrics.overall_score, reverse=(order == SORT_BEST))
    if not photos:
        st.info("No photos match this filter.")
        return

    page_count = (len(photos) - 1) // GALLERY_PAGE_SIZE + 1
    current = min(st.session_state.get("gallery_page", 1), page_count)
    st.session_state["gallery_page"] = current
    filter_cols[3].number_input(
        f"Page (of {page_count})", 1, page_count, key="gallery_page", step=1
    )
    start = (current - 1) * GALLERY_PAGE_SIZE
    visible = photos[start : start + GALLERY_PAGE_SIZE]
    st.caption(f"Showing {start + 1}-{start + len(visible)} of {len(photos)} photos.")

    for row_start in range(0, len(visible), GALLERY_COLUMNS):
        columns = st.columns(GALLERY_COLUMNS)
        for column, photo in zip(
            columns, visible[row_start : row_start + GALLERY_COLUMNS], strict=False
        ):
            with column:
                render_photo_card(photo)


# ----------------------------------------------------------- tab 2: organize


def render_organize_tab(pipe: Pipeline) -> None:
    """Tab 2: folder organization with dry run, undo and EXIF tagging."""
    st.subheader("Organize folders")
    left, right = st.columns(2)
    strategy = left.selectbox(
        "Strategy",
        list(OrganizationStrategy),
        format_func=lambda s: STRATEGY_HELP[s][0],
        key="organize_strategy",
    )
    left.caption(STRATEGY_HELP[strategy][1])
    mode = right.selectbox(
        "File action",
        list(FileActionMode),
        format_func=lambda m: MODE_HELP[m],
        key="organize_mode",
    )
    if strategy is OrganizationStrategy.CULL_REJECTED and mode is not FileActionMode.MOVE:
        right.caption(
            "Culling normally uses Move; with Copy or Hard link the rejected "
            "photos also stay in the main folder."
        )
    else:
        right.caption("Existing files are never overwritten; clashes get a `_1` suffix.")

    signature = (strategy, mode)
    buttons = st.columns(3)
    if buttons[0].button("Preview (Dry Run)", width="stretch"):
        st.session_state["preview"] = (signature, pipe.preview_organization(strategy, mode))
    if buttons[1].button("Execute Organization", type="primary", width="stretch"):
        batch = pipe.execute_organization(strategy, mode)
        st.session_state.pop("preview", None)
        failed = len(batch.transactions) - batch.executed_count
        if not batch.transactions:
            flash("Nothing to do: every photo is already in place.", "info")
        else:
            flash(
                f"{STRATEGY_HELP[strategy][0]}: {batch.executed_count} file(s) processed"
                + (f", {failed} skipped (see the journal notes)." if failed else ".")
            )
        st.rerun()
    if buttons[2].button(
        "Undo Last Organization",
        width="stretch",
        disabled=not pipe.can_undo_organization(),
        help="Reverts the most recent batch using the transaction journal.",
    ):
        batch = pipe.undo_last_organization()
        st.session_state.pop("preview", None)
        if batch is not None:
            kept = batch.executed_count
            flash(
                f"Undone: {STRATEGY_HELP[batch.strategy][0]} "
                f"({len(batch.transactions) - kept} file(s) restored"
                + (f", {kept} could not be reverted)." if kept else ").")
            )
        st.rerun()

    stored = st.session_state.get("preview")
    if stored and stored[0] == signature:
        transactions = stored[1]
        if not transactions:
            st.info("Dry run: nothing to do — every photo is already in place.")
        else:
            st.markdown(
                f"**Dry run: {len(transactions)} planned operation(s).** "
                "Nothing has changed on disk yet."
            )
            st.dataframe(
                [
                    {
                        "Action": t.mode.value,
                        "Source": relative(t.source, pipe.root),
                        "Destination": relative(t.destination, pipe.root),
                        "Note": t.note,
                    }
                    for t in transactions
                ],
                width="stretch",
                hide_index=True,
            )

    st.divider()
    st.subheader("Write ratings & tags to EXIF")
    st.caption(
        "Writes the star rating (EXIF Rating / RatingPercent) and the keywords "
        "scene, quality tier and event (XPKeywords, ImageDescription) into each "
        "JPEG. Only the metadata block is replaced; image pixels are not "
        "re-encoded. Other formats are skipped."
    )
    if st.button("Write Ratings & Tags to EXIF", type="primary"):
        bar = st.progress(0.0, text="Writing metadata...")
        report = pipe.write_metadata(
            lambda done, total, name: bar.progress(
                done / total, text=f"Tagging {done}/{total}: {name}"
            )
        )
        bar.empty()
        st.session_state["metadata_report"] = report
    report = st.session_state.get("metadata_report")
    if report is not None:
        cols = st.columns(3)
        cols[0].metric("Written", report.written)
        cols[1].metric("Skipped (not JPEG)", report.skipped)
        cols[2].metric("Failed", report.failed)
        if report.failures:
            with st.expander("Failures"):
                for rel_path, reason in report.failures.items():
                    st.markdown(f"- `{rel_path}` — {reason}")


# -------------------------------------------------------------- tab 3: album


def render_album_tab(pipe: Pipeline) -> None:
    """Tab 3: one-click album generation and the per-chapter wizard."""
    keepers = pipe.keepers()
    if not keepers:
        st.info("There are no keepers yet. Scan a folder and curate it in tab 1.")
        return

    left, right = st.columns(2)
    max_per_page = left.slider(
        "Maximum photos per page", 1, MAX_PHOTOS_PER_PAGE, 4, key="album_max_per_page"
    )
    chapter_breaks = right.checkbox(
        "Start every event on a new page (chapter breaks)",
        value=True,
        key="album_chapter_breaks",
    )

    st.subheader("Auto-Pilot")
    st.caption(f"Lays out all {len(keepers)} keepers in one step. Replaces the current album.")
    if st.button("Generate entire album", type="primary"):
        pages = pipe.generate_album(max_per_page, chapter_breaks)
        st.session_state["page_number"] = 1
        flash(f"Album generated: {len(pages)} page(s). Fine-tune it in tab 4.")
        st.rerun()

    st.divider()
    st.subheader("Step-by-step wizard")
    st.caption("Build or rebuild one chapter at a time; other chapters are left untouched.")
    built = pipe.chapters_built()
    clusters = [c for c in pipe.state.clusters if pipe.keepers(c.cluster_id)]
    pages_per_cluster: dict[str, int] = {}
    for page in pipe.state.pages:
        if page.cluster_id:
            pages_per_cluster[page.cluster_id] = pages_per_cluster.get(page.cluster_id, 0) + 1

    st.dataframe(
        [
            {
                "Chapter": cluster.title,
                "From": f"{cluster.start:%Y-%m-%d %H:%M}",
                "To": f"{cluster.end:%H:%M}",
                "Keepers": len(pipe.keepers(cluster.cluster_id)),
                "Pages": pages_per_cluster.get(cluster.cluster_id, 0),
                "Status": "Built" if cluster.cluster_id in built else "Not built",
            }
            for cluster in clusters
        ],
        width="stretch",
        hide_index=True,
    )

    pending = [c for c in clusters if c.cluster_id not in built]
    cols = st.columns([2, 1, 1, 1])
    chosen = cols[0].selectbox(
        "Chapter",
        [c.cluster_id for c in clusters],
        format_func=lambda cid: next(c.title for c in clusters if c.cluster_id == cid),
        key="wizard_chapter",
        label_visibility="collapsed",
    )
    if cols[1].button("Build this chapter", width="stretch"):
        pages = pipe.generate_chapter(chosen, max_per_page)
        flash(f"Chapter {chosen}: {len(pages)} page(s).")
        st.rerun()
    if cols[2].button("Build next unbuilt", width="stretch", disabled=not pending):
        pages = pipe.generate_chapter(pending[0].cluster_id, max_per_page)
        flash(f"Chapter {pending[0].title}: {len(pages)} page(s).")
        st.rerun()
    if cols[3].button("Clear album", width="stretch", disabled=not pipe.state.pages):
        pipe.clear_album()
        flash("Album cleared.", "info")
        st.rerun()

    total_pages = len(pipe.state.pages)
    if total_pages:
        unused = len(pipe.unused_keepers())
        st.success(
            f"The album has {total_pages} page(s)"
            + (
                f"; {unused} keeper(s) are not placed yet."
                if unused
                else "; every keeper is placed."
            )
        )


# ------------------------------------------------------------- tab 4: studio


def export_options() -> ExportOptions:
    """Read the export settings from the studio widgets."""
    return ExportOptions(
        title=st.session_state.get("export_title") or "PhotoSeder Album",
        page_size=PageSize(st.session_state.get("export_page_size", PageSize.A4_LANDSCAPE)),
        embed_images=st.session_state.get("export_embed", True),
        crop_to_fill=st.session_state.get("export_crop", True),
        include_cover=st.session_state.get("export_cover", True),
    )


def on_template_change(page_id: str, key: str) -> None:
    """Callback: switch the layout template of a page."""
    try:
        pipeline().set_page_template(page_id, LayoutTemplate(st.session_state[key]))
    except ValueError as exc:
        flash(str(exc), "error")


def on_page_action(action: str, page_id: str, photo_id: str = "", offset: int = 0) -> None:
    """Callback: dispatch a studio editing action to the pipeline."""
    pipe = pipeline()
    try:
        match action:
            case "reorder":
                pipe.reorder_photo(page_id, photo_id, offset)
            case "remove":
                pipe.remove_photo_from_page(page_id, photo_id)
            case "add":
                pipe.add_photo_to_page(page_id, photo_id)
            case "move":
                pipe.move_photo(photo_id, page_id)
            case "new_page":
                pipe.add_page(photo_id, after_page_id=page_id)
                st.session_state["page_number"] = pipe.page_index(page_id) + 2
            case "delete_page":
                pipe.delete_page(page_id)
            case "move_page":
                pipe.move_page(page_id, offset)
                st.session_state["page_number"] = pipe.page_index(page_id) + 1
            case "auto_template":
                pipe.reset_page_template(page_id)
    except (ValueError, KeyError) as exc:
        flash(str(exc), "error")


def on_step_page(offset: int) -> None:
    """Callback: go to the previous or next page."""
    st.session_state["page_number"] = st.session_state.get("page_number", 1) + offset


def render_studio_tab(pipe: Pipeline) -> None:
    """Tab 4: page-by-page editor and print export."""
    pages = pipe.state.pages
    if not pages:
        st.info("The album is empty. Generate it in tab 3 first.")
        return

    page_count = len(pages)
    number = min(max(st.session_state.get("page_number", 1), 1), page_count)
    st.session_state["page_number"] = number

    nav = st.columns([1, 2, 1, 4])
    nav[0].button(
        "Previous", on_click=on_step_page, args=(-1,), disabled=number <= 1, width="stretch"
    )
    nav[1].number_input(
        "Page", 1, page_count, key="page_number", step=1, label_visibility="collapsed"
    )
    nav[2].button(
        "Next", on_click=on_step_page, args=(1,), disabled=number >= page_count, width="stretch"
    )
    page = pages[number - 1]
    nav[3].markdown(
        f"**Page {number} of {page_count}** · {page.template.label}"
        + (f" · chapter `{page.cluster_id}`" if page.cluster_id else "")
    )

    preview_col, edit_col = st.columns([3, 2])
    with preview_col:
        html = pipe.render_page_preview(page.page_id, export_options(), thumbnails().data_uri)
        components.html(html, height=470, scrolling=False)

    with edit_col:
        options = compatible_templates(len(page.photo_ids))
        template_key = f"template_{page.page_id}"
        st.session_state[template_key] = page.template.value
        st.selectbox(
            "Layout template",
            [template.value for template in options],
            key=template_key,
            format_func=lambda value: LayoutTemplate(value).label,
            on_change=on_template_change,
            args=(page.page_id, template_key),
            help="Templates that can hold this page's photos.",
        )
        st.button(
            "Use automatic layout",
            on_click=on_page_action,
            args=("auto_template", page.page_id),
            disabled=not page.template_locked,
            width="stretch",
        )
        cols = st.columns(3)
        cols[0].button(
            "Page earlier",
            on_click=on_page_action,
            args=("move_page", page.page_id),
            kwargs={"offset": -1},
            disabled=number <= 1,
            width="stretch",
        )
        cols[1].button(
            "Page later",
            on_click=on_page_action,
            args=("move_page", page.page_id),
            kwargs={"offset": 1},
            disabled=number >= page_count,
            width="stretch",
        )
        cols[2].button(
            "Delete page",
            on_click=on_page_action,
            args=("delete_page", page.page_id),
            width="stretch",
            help="Photos return to the unused keeper pool.",
        )

    st.markdown("##### Photos on this page")
    other_pages = {
        f"Page {index}": other.page_id
        for index, other in enumerate(pages, start=1)
        if other.page_id != page.page_id and len(other.photo_ids) < MAX_PHOTOS_PER_PAGE
    }
    for slot, photo_id in enumerate(page.photo_ids):
        photo = pipe.state.photos[photo_id]
        row = st.columns([1, 3, 1, 1, 2, 1, 1])
        thumb = thumbnails().path_for(photo)
        if thumb is not None:
            row[0].image(str(thumb), width="stretch")
        row[1].markdown(
            f"**Slot {slot + 1}** · {photo.name}  \n"
            f"{stars_text(photo.star_rating)} · {photo.scene.label} · {photo.orientation.value}"
        )
        row[2].button(
            "Earlier",
            key=f"up_{photo_id}",
            on_click=on_page_action,
            args=("reorder", page.page_id, photo_id, -1),
            disabled=slot == 0,
            width="stretch",
        )
        row[3].button(
            "Later",
            key=f"down_{photo_id}",
            on_click=on_page_action,
            args=("reorder", page.page_id, photo_id, 1),
            disabled=slot == len(page.photo_ids) - 1,
            width="stretch",
        )
        target_label = row[4].selectbox(
            "Move to",
            list(other_pages) or ["No page with free space"],
            key=f"target_{photo_id}",
            label_visibility="collapsed",
        )
        row[5].button(
            "Move",
            key=f"move_{photo_id}",
            on_click=on_page_action,
            args=("move", other_pages.get(target_label, ""), photo_id),
            disabled=not other_pages,
            width="stretch",
        )
        row[6].button(
            "Remove",
            key=f"remove_{photo_id}",
            on_click=on_page_action,
            args=("remove", page.page_id, photo_id),
            width="stretch",
        )

    unused = pipe.unused_keepers()
    st.markdown(f"##### Unused keepers ({len(unused)})")
    if unused:
        by_id = {photo.photo_id: photo for photo in unused}
        pool = st.columns([1, 3, 2, 2])
        chosen = pool[1].selectbox(
            "Keeper",
            list(by_id),
            format_func=lambda pid: " · ".join(
                (by_id[pid].name, stars_text(by_id[pid].star_rating), by_id[pid].scene.label)
            ),
            key="pool_choice",
            label_visibility="collapsed",
        )
        thumb = thumbnails().path_for(by_id[chosen])
        if thumb is not None:
            pool[0].image(str(thumb), width="stretch")
        pool[2].button(
            "Add to this page",
            on_click=on_page_action,
            args=("add", page.page_id, chosen),
            disabled=len(page.photo_ids) >= MAX_PHOTOS_PER_PAGE,
            width="stretch",
        )
        pool[3].button(
            "New page after this one",
            on_click=on_page_action,
            args=("new_page", page.page_id, chosen),
            width="stretch",
        )
    else:
        st.caption("Every keeper is placed in the album.")

    st.divider()
    render_export_section(pipe)


def render_export_section(pipe: Pipeline) -> None:
    """Export settings and the print-ready HTML export button."""
    st.subheader("Print export")
    cols = st.columns([2, 2, 2])
    cols[0].text_input("Album title", value="PhotoSeder Album", key="export_title")
    cols[1].selectbox(
        "Page format",
        [size.value for size in PageSize],
        format_func=lambda value: PageSize(value).label,
        key="export_page_size",
    )
    cols[2].text_input("File name", value=DEFAULT_ALBUM_FILENAME, key="export_filename")
    toggles = st.columns(3)
    toggles[0].checkbox(
        "Embed photos (single self-contained file)",
        value=True,
        key="export_embed",
        help="Off: the HTML links to the original files and breaks if they move.",
    )
    toggles[1].checkbox(
        "Crop photos to fill their cell",
        value=True,
        key="export_crop",
        help="Off: whole photos are shown with white space around them.",
    )
    toggles[2].checkbox("Add a cover page", value=True, key="export_cover")

    if st.button("Export print-ready album (HTML)", type="primary"):
        name = Path(st.session_state["export_filename"].strip() or DEFAULT_ALBUM_FILENAME).name
        if not name.lower().endswith((".html", ".htm")):
            name += ".html"
        with st.spinner("Rendering album..."):
            try:
                output = pipe.export_album(export_options(), pipe.root / name)
            except (ValueError, OSError) as exc:
                st.error(f"Export failed: {exc}")
                return
        st.session_state["export_path"] = str(output)

    exported = st.session_state.get("export_path")
    if exported and Path(exported).is_file():
        size_mb = Path(exported).stat().st_size / 1_048_576
        st.success(f"Album saved to `{exported}` ({size_mb:.1f} MB).")
        st.caption(
            "Open the file in your browser and print it (Ctrl/Cmd + P) → "
            '"Save as PDF". Set margins to *None* and enable *Background '
            "graphics*; the page size is set by the document."
        )
        st.download_button(
            "Download a copy",
            data=Path(exported).read_bytes(),
            file_name=Path(exported).name,
            mime="text/html",
        )


# ---------------------------------------------------------------------- main


def render_welcome() -> None:
    """Landing view shown before a folder is opened."""
    st.title("PhotoSeder")
    st.markdown(
        "Smart photo curation, folder organization, EXIF tagging and album "
        "building — **100% on this computer**. No photo, path or piece of "
        "metadata is ever sent anywhere.\n\n"
        "**To begin**, paste the path of a photo folder in the sidebar and "
        "press **Open & scan**.\n\n"
        "1. **Scan, Cull & Curate** — blurry, badly exposed and burst-duplicate "
        "photos are set aside; you can restore any of them.\n"
        "2. **Organize & Tag** — sort files into folders with a dry run and "
        "one-click undo; write ratings and keywords into EXIF.\n"
        "3. **Album Builder** — lay out the keepers automatically.\n"
        "4. **Page Studio & Export** — adjust pages and export a print-ready album."
    )


def main() -> None:
    """Application entry point."""
    st.set_page_config(page_title="PhotoSeder", layout="wide")
    render_sidebar()

    pipe = pipeline()
    if pipe is None:
        render_welcome()
        return

    st.title("PhotoSeder")
    st.caption(f"Folder: `{pipe.root}`")
    show_flashes()

    if not pipe.state.photos:
        st.warning("No photos were found in this folder.")
        if pipe.state.unreadable:
            st.write(pipe.state.unreadable)
        return

    cull, organize, album, studio = st.tabs(
        [
            "1 · Scan, Cull & Curate · בקרת איכות וסינון",
            "2 · Folder Organizer & EXIF Tagger · סידור תיקיות ותיוג",
            "3 · Album Builder · בניית אלבום אוטומטית / בשלבים",
            "4 · Page Studio & Print Export · עורך עמודים וייצוא",
        ]
    )
    with cull:
        render_cull_tab(pipe)
    with organize:
        render_organize_tab(pipe)
    with album:
        render_album_tab(pipe)
    with studio:
        render_studio_tab(pipe)


main()
