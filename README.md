# PhotoSeder

A **100% local-first** photo curation, folder organizer, EXIF tagger and photo album engine with a
Streamlit UI. Every analysis step runs on your machine with OpenCV, Pillow and piexif. No image,
path or piece of metadata is sent anywhere; Streamlit telemetry is disabled in
`.streamlit/config.toml` and the exported album references no remote assets.

## Run it

Requires Python 3.12 or newer.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

On Windows use `.venv\Scripts\activate` instead.

```bash
pip install -r requirements.txt
```

```bash
streamlit run photoseder/ui/app.py
```

Run the last command from the repository root so `.streamlit/config.toml` is picked up. The app
opens at `http://localhost:8501`. Paste a folder path in the sidebar and press **Open & scan**.

Optional:

```bash
pip install -e .
```

makes `photoseder` importable from anywhere, and

```bash
pip install pillow-heif
```

adds HEIC/HEIF (iPhone) input.

Run the tests with:

```bash
pytest
```

## What it does

| Tab | Function |
|---|---|
| 1 · Scan, Cull & Curate | Dashboard, filterable gallery, restore/reject, override scene and stars |
| 2 · Folder Organizer & EXIF Tagger | Four strategies × move/copy/hard link, dry run, execute, undo; write ratings and keywords to EXIF |
| 3 · Album Builder | One-click album or chapter-by-chapter wizard |
| 4 · Page Studio & Print Export | Per-page preview, template switch, reorder/move/remove/add photos, export print-ready HTML |

To get a PDF, open the exported HTML in a browser, print (Ctrl/Cmd + P), choose *Save as PDF*, set
margins to *None* and enable *Background graphics*. The page size (A4 landscape or square) is set
by the document.

## Architecture

```text
photoseder/
├── domain/          enums.py, models.py        pure data, no third-party imports
├── services/
│   ├── interfaces.py        Protocols for every service
│   ├── analyzer.py          sharpness, exposure, faces/eyes, scene, score, stars
│   ├── deduplicator.py      dHash, burst culling, event clustering
│   ├── organizer.py         planned/dry-run file operations + undo journal
│   ├── metadata_writer.py   lossless EXIF rating and keyword injection
│   ├── layout_planner.py    page template selection
│   ├── state_manager.py     JSON cache + incremental change detection
│   ├── thumbnails.py        on-disk thumbnail cache for the UI
│   └── exporter.py          HTML5 / CSS Grid album renderer
├── orchestrator/pipeline.py  the single entry point used by the UI
└── ui/app.py                 Streamlit application
```

The pipeline receives its services by constructor injection and depends only on the protocols, so
any service can be swapped (for example a different `FaceDetector`) without touching the UI.

### Scoring

| Component | Weight | Measurement |
|---|---|---|
| Sharpness | 55% | 90th-percentile tile variance of the Laplacian on a 1024 px copy, `min(1, var / (3 × blur threshold))` |
| Exposure / contrast | 25% | Histogram clipping (≤5, ≥250), RMS contrast, mean luminance |
| Resolution | 10% | `min(1, megapixels / 8)` |
| Eyes open | 10% | Mean of `min(eyes found, 2) / 2` over detected faces |

Stars: ≥0.85 → 5, ≥0.70 → 4, ≥0.55 → 3, ≥0.40 → 2, otherwise 1 (rejected).

### Deliberate deviations from the original specification

- **Quality gates.** With the specified weights alone, a completely blurred but well-exposed photo
  still scores about 0.43 and would be kept. A photo below the blur threshold, or below the
  exposure gate, is therefore capped just under the reject threshold and rejected with a specific reason.
- **No faces → weights renormalized.** Without this, a faceless photo could never exceed 0.90.
- **Tiled sharpness** instead of one whole-frame variance, so a sharp subject on a blurred
  background is not flagged as blurry.
- **Hero pages** use the configurable hero threshold (default 0.82) rather than "5 stars", since
  the two numbers in the specification conflict.
- **One event naming scheme**, `YYYY-MM-DD_Morning|Afternoon|Evening|Night` (with `_2`, `_3` on
  collision), for both chapter titles and event folders.
- **OpenCV is pinned to 4.x.** OpenCV 5 removed the Haar cascade detector the specification requires.

## Files PhotoSeder creates in your photo folder

| File | Purpose |
|---|---|
| `.photoseder_state.json` | Cached analysis, your manual decisions and the album |
| `.photoseder_journal.json` | Undo journal; exists only while there is something to undo |
| `.photoseder_cache/` | Thumbnails; safe to delete |
| `PhotoSeder_Album.html` | Exported album (name is configurable) |

## Known limitations

- **Blink detection is approximate.** The Haar eye cascade reports whether eyes were *found*, which
  correlates with open eyes but is unreliable with glasses, profiles and small faces. Haar face
  detection also produces occasional false positives.
- **Ratings are written to EXIF only** (`Rating`, `RatingPercent`, `XPKeywords`,
  `ImageDescription`), which Windows Explorer and many viewers read. Lightroom, Bridge and digiKam
  read XMP ratings, which are not written.
- **EXIF tagging works for JPEG only**; other formats are skipped and counted.
- **Tag before hard-linking.** Tagging replaces the file atomically, which gives it a new identity
  on disk; hard links created earlier keep pointing at the untagged version.
- **The undo journal stores absolute paths.** Undo an organization before renaming or moving the
  project folder itself.
- **Events split at midnight**, as specified, so a party that runs past 00:00 becomes two chapters.
- The default thresholds were calibrated on a small set of sample images; expect to adjust
  *Blur sensitivity* for your camera.
