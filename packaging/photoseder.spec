# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the PhotoSeder desktop app (one-folder build).

Build from the repository root:

    pyinstaller packaging/photoseder.spec --noconfirm

Result: dist/PhotoSeder/PhotoSeder (PhotoSeder.exe on Windows).
"""

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

ROOT = Path(SPECPATH).parent
ICON = ROOT / "packaging" / "photoseder.ico"

datas = [(str(ROOT / "photoseder" / "ui" / "app.py"), "photoseder/ui")]
binaries = []
hiddenimports = collect_submodules("photoseder")

# Streamlit ships its web front end as data files and reads its own package
# metadata at start-up; collect everything it (and its dependencies) needs.
for package in ("streamlit",):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden
datas += copy_metadata("streamlit", recursive=True)

# The Haar cascade files used for face and eye detection (cv2/data/*.xml).
datas += collect_data_files(
    "cv2",
    includes=["data/haarcascade_frontalface_default.xml", "data/haarcascade_eye.xml"],
)

# The window (pywebview) is optional at build time: without it the app falls back
# to the default browser.
hiddenimports += collect_submodules("webview")

analysis = Analysis(
    [str(ROOT / "photoseder" / "__main__.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["pytest", "ruff"],
    noarchive=False,
)
archive = PYZ(analysis.pure)

executable = EXE(
    archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="PhotoSeder",
    console=False,
    icon=str(ICON) if ICON.exists() else None,
)

COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    name="PhotoSeder",
)
