"""Local JSON state cache and incremental change detection.

The whole :class:`ProjectState` is stored in ``<folder>/.photoseder_state.json``.
Reopening a folder loads this file instead of re-analyzing every photo; only
files that are new or whose ``(size, mtime_ns)`` fingerprint changed are sent
to the analyzer.

Change detection (:meth:`JsonStateStore.diff`):
    * same relative path and fingerprint -> cached, nothing to do;
    * same path, different fingerprint -> ``modified`` (re-analyze, keep id);
    * path not in the cache -> ``new``, unless a cached photo that vanished
      has the same file name and fingerprint, in which case the file was
      moved or renamed outside PhotoSeder and its analysis is reused;
    * cached photo with no file left -> ``removed``.

Writes are atomic (temporary file + ``os.replace``), so a crash while saving
never corrupts the previous state. An unreadable state file is set aside as
``.photoseder_state.json.corrupt`` and the folder is treated as new.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from photoseder.domain.models import IMAGE_EXTENSIONS, Fingerprint, ProjectState
from photoseder.services.interfaces import ScanDiff

STATE_FILENAME = ".photoseder_state.json"


class JsonStateStore:
    """Persists project state as JSON next to the photos."""

    def state_path(self, root: Path) -> Path:
        """Location of the state file for a folder."""
        return root / STATE_FILENAME

    def load(self, root: Path) -> ProjectState | None:
        """Load the cached state for a folder.

        Args:
            root: Project folder.

        Returns:
            The cached state, or ``None`` when there is none or it is invalid.
        """
        path = self.state_path(root)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ProjectState.from_dict(root, data)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            try:
                os.replace(path, path.with_name(path.name + ".corrupt"))
            except OSError:
                # Could not set the file aside; the next save overwrites it.
                return None
            return None

    def save(self, state: ProjectState) -> None:
        """Persist the state atomically.

        Args:
            state: State to write; ``updated_at`` is refreshed.
        """
        state.updated_at = datetime.now()
        path = self.state_path(state.root)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def discover(self, root: Path, excluded: set[str]) -> dict[str, Fingerprint]:
        """Find image files under ``root`` recursively.

        Hidden files and folders (names starting with ``.``, which includes
        PhotoSeder's own cache) are skipped, as is every path in ``excluded``.

        Args:
            root: Project folder.
            excluded: Absolute paths to ignore (copies made by the organizer).

        Returns:
            ``rel_path -> (size_bytes, mtime_ns)`` with POSIX separators.
        """
        excluded_keys = {os.path.normcase(path) for path in excluded}
        found: dict[str, Fingerprint] = {}
        for directory, subdirs, files in os.walk(root):
            subdirs[:] = sorted(d for d in subdirs if not d.startswith("."))
            for name in sorted(files):
                if name.startswith("."):
                    continue
                if os.path.splitext(name)[1].lower() not in IMAGE_EXTENSIONS:
                    continue
                full = os.path.join(directory, name)
                if os.path.normcase(full) in excluded_keys:
                    continue
                try:
                    stat = os.stat(full)
                except OSError:
                    continue
                rel_path = Path(full).relative_to(root).as_posix()
                found[rel_path] = (stat.st_size, stat.st_mtime_ns)
        return found

    def diff(self, state: ProjectState, discovered: dict[str, Fingerprint]) -> ScanDiff:
        """Compare cached photos with the files found on disk.

        Args:
            state: Cached state.
            discovered: Output of :meth:`discover`.

        Returns:
            What must be analyzed, re-analyzed, re-pathed or dropped.
        """
        result = ScanDiff()
        cached = state.photos_by_path()

        missing = {
            (photo.name, photo.fingerprint): photo.photo_id
            for path, photo in cached.items()
            if path not in discovered
        }
        for rel_path, fingerprint in discovered.items():
            photo = cached.get(rel_path)
            if photo is not None:
                if photo.fingerprint == fingerprint:
                    result.unchanged += 1
                else:
                    result.modified.append(rel_path)
                continue
            name = rel_path.rsplit("/", 1)[-1]
            moved_id = missing.pop((name, fingerprint), None)
            if moved_id is not None:
                result.moved[moved_id] = rel_path
            else:
                result.new.append(rel_path)
        result.removed = list(missing.values())
        return result
