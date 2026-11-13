"""Safe physical folder organization with a reversible undo journal.

Architecture:
    :class:`FolderOrganizer` separates *planning* from *execution*.

    * :meth:`FolderOrganizer.plan` is pure: it maps photos to destination
      paths for a strategy and resolves name collisions, but never touches the
      disk. Its output is exactly what the UI shows as the dry run.
    * :meth:`FolderOrganizer.organize` executes a plan transaction by
      transaction and records the batch in ``.photoseder_journal.json``.
    * :meth:`FolderOrganizer.undo_last_operation` replays the most recent
      batch in reverse.

Safety rules:
    * Nothing is ever overwritten. A destination that exists on disk or is
      already claimed by the same plan gets a ``_1``, ``_2``, ... suffix.
    * The journal is written atomically (temporary file + ``os.replace``)
      *before* the first file is touched and again as the batch progresses,
      so a crash in the middle of a batch leaves a journal that can undo
      whatever was already done.
    * Undo of a ``MOVE`` moves the file back only if the original location is
      free. Undo of a ``COPY`` / ``HARDLINK`` deletes the destination only if
      it is still the very file the batch created (same inode and size), so a
      copy the user edited afterwards is kept.
    * Hard links cannot span volumes; such a transaction is planned (and
      shown in the dry run) as a ``COPY``, and an unexpected ``os.link``
      failure at execution time falls back to copying as well.
    * Directories created by a batch are removed on undo when empty.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from photoseder.domain.enums import (
    FileActionMode,
    OrganizationStrategy,
    RejectionCategory,
)
from photoseder.domain.models import (
    EventCluster,
    FileTransaction,
    OperationBatch,
    PhotoItem,
    new_id,
)

JOURNAL_FILENAME = ".photoseder_journal.json"
REJECTED_QUALITY_DIR = "_Rejected_Quality"
REJECTED_DUPLICATES_DIR = "_Rejected_Duplicates"
BEST_OF_DIR = "Best_Of"
BEST_OF_MIN_STARS = 4

_JOURNAL_VERSION = 1
_JOURNAL_FLUSH_EVERY = 25


class FolderOrganizer:
    """Plans, executes and undoes file organization batches for one folder."""

    def __init__(self, root: Path) -> None:
        """Bind the organizer to a project folder.

        Args:
            root: Project folder; all destinations are created beneath it.
        """
        self._root = root
        self._journal_path = root / JOURNAL_FILENAME

    # ------------------------------------------------------------------ plan

    def plan(
        self,
        photos: Sequence[PhotoItem],
        clusters: Sequence[EventCluster],
        strategy: OrganizationStrategy,
        mode: FileActionMode,
    ) -> list[FileTransaction]:
        """Compute the file operations for a strategy without touching disk.

        Strategy rules:
            * ``CULL_REJECTED``: rejected photos go to ``_Rejected_Quality/``
              (quality or manual rejections) or ``_Rejected_Duplicates/``
              (burst duplicates). A keeper that still sits in one of those
              folders (because the user restored it) returns to the root.
            * ``BY_EVENT``: keepers go to ``<cluster title>/``.
            * ``BY_SCENE``: keepers go to ``Landscapes/``, ``Portraits/``,
              ``Groups/`` or ``General/``.
            * ``BEST_OF_HEROES``: keepers rated 4-5 stars go to ``Best_Of/``.

        Photos already located in their target folder are skipped, which
        makes every strategy safe to run repeatedly.

        Args:
            photos: Photos to organize.
            clusters: Event clusters (used by ``BY_EVENT``).
            strategy: Organization strategy.
            mode: Requested file action.

        Returns:
            Transactions in deterministic (chronological) order.
        """
        titles = {cluster.cluster_id: cluster.title for cluster in clusters}
        claimed: set[str] = set()
        transactions: list[FileTransaction] = []

        ordered = sorted(photos, key=lambda p: (p.captured_at, p.rel_path))
        for photo in ordered:
            folder = self._target_folder(photo, strategy, titles)
            if folder is None:
                continue
            source = photo.absolute_path(self._root)
            target_dir = self._root / folder if folder else self._root
            if source.parent == target_dir:
                continue

            destination = self._free_destination(target_dir / source.name, claimed)
            claimed.add(os.path.normcase(str(destination)))
            action, note = mode, ""
            if mode is FileActionMode.HARDLINK and not self._same_volume(source, target_dir):
                action = FileActionMode.COPY
                note = "Different volume: hard link not possible, copying instead"
            transactions.append(
                FileTransaction(
                    photo_id=photo.photo_id,
                    source=str(source),
                    destination=str(destination),
                    mode=action,
                    note=note,
                )
            )
        return transactions

    def _target_folder(
        self,
        photo: PhotoItem,
        strategy: OrganizationStrategy,
        titles: dict[str, str],
    ) -> str | None:
        """Return the destination folder name, ``""`` for the root or ``None``.

        ``None`` means the strategy does not apply to this photo.
        """
        match strategy:
            case OrganizationStrategy.CULL_REJECTED:
                if photo.is_selected:
                    top = photo.rel_path.split("/", 1)[0]
                    in_rejected = "/" in photo.rel_path and top in (
                        REJECTED_QUALITY_DIR,
                        REJECTED_DUPLICATES_DIR,
                    )
                    return "" if in_rejected else None
                if photo.rejection_category is RejectionCategory.DUPLICATE:
                    return REJECTED_DUPLICATES_DIR
                return REJECTED_QUALITY_DIR
            case OrganizationStrategy.BY_EVENT:
                if not photo.is_selected or photo.cluster_id is None:
                    return None
                return _safe_folder_name(titles.get(photo.cluster_id, photo.cluster_id))
            case OrganizationStrategy.BY_SCENE:
                return photo.scene.folder_name if photo.is_selected else None
            case OrganizationStrategy.BEST_OF_HEROES:
                if photo.is_selected and photo.star_rating >= BEST_OF_MIN_STARS:
                    return BEST_OF_DIR
                return None
        return None

    @staticmethod
    def _free_destination(candidate: Path, claimed: set[str]) -> Path:
        """Return ``candidate`` or the first ``stem_N`` variant that is free."""
        destination = candidate
        counter = 0
        while os.path.lexists(destination) or os.path.normcase(str(destination)) in claimed:
            counter += 1
            destination = candidate.with_name(f"{candidate.stem}_{counter}{candidate.suffix}")
        return destination

    @staticmethod
    def _same_volume(source: Path, target_dir: Path) -> bool:
        """Check whether a hard link from ``source`` into ``target_dir`` works."""
        existing = target_dir
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        try:
            return source.stat().st_dev == existing.stat().st_dev
        except OSError:
            return False

    # --------------------------------------------------------------- execute

    def organize(
        self,
        photos: Sequence[PhotoItem],
        clusters: Sequence[EventCluster],
        strategy: OrganizationStrategy,
        mode: FileActionMode,
        *,
        dry_run: bool = False,
    ) -> OperationBatch:
        """Plan and (unless ``dry_run``) execute and journal a batch.

        Args:
            photos: Photos to organize.
            clusters: Event clusters (used by ``BY_EVENT``).
            strategy: Organization strategy.
            mode: Requested file action.
            dry_run: When ``True`` only the plan is returned; nothing is
                written to disk or to the journal.

        Returns:
            The batch; each transaction's ``executed`` flag and ``note``
            report what actually happened.
        """
        batch = OperationBatch(
            batch_id=new_id(),
            created_at=datetime.now(),
            strategy=strategy,
            mode=mode,
            transactions=self.plan(photos, clusters, strategy, mode),
            dry_run=dry_run,
        )
        if dry_run or not batch.transactions:
            return batch

        journal = self._read_journal()
        journal.append(batch)
        self._write_journal(journal)
        try:
            for index, transaction in enumerate(batch.transactions, start=1):
                self._execute(transaction, batch)
                if index % _JOURNAL_FLUSH_EVERY == 0:
                    self._write_journal(journal)
        finally:
            if batch.executed_count == 0 and not batch.created_dirs:
                journal.remove(batch)
            self._write_journal(journal)
        return batch

    def _execute(self, transaction: FileTransaction, batch: OperationBatch) -> None:
        """Perform one transaction, recording the outcome on it."""
        source = Path(transaction.source)
        destination = Path(transaction.destination)
        try:
            if not source.is_file():
                raise FileNotFoundError("Source file is missing")
            if os.path.lexists(destination):
                raise FileExistsError("Destination already exists")
            self._make_dirs(destination.parent, batch)

            match transaction.mode:
                case FileActionMode.MOVE:
                    shutil.move(str(source), str(destination))
                case FileActionMode.COPY:
                    shutil.copy2(source, destination)
                case FileActionMode.HARDLINK:
                    try:
                        os.link(source, destination)
                    except OSError as exc:
                        shutil.copy2(source, destination)
                        transaction.mode = FileActionMode.COPY
                        transaction.note = f"Hard link failed ({exc}); copied instead"
            stat = destination.stat()
            transaction.dest_inode = stat.st_ino
            transaction.dest_size = stat.st_size
            transaction.executed = True
        except OSError as exc:
            transaction.executed = False
            transaction.note = f"Skipped: {exc}"

    def _make_dirs(self, directory: Path, batch: OperationBatch) -> None:
        """Create ``directory`` and record each newly created level."""
        missing: list[Path] = []
        current = directory
        while not current.exists():
            missing.append(current)
            current = current.parent
        for path in reversed(missing):
            path.mkdir()
            batch.created_dirs.append(str(path))

    # ------------------------------------------------------------------ undo

    def can_undo(self) -> bool:
        """Return ``True`` when the journal holds a batch to undo."""
        return bool(self._read_journal())

    def last_batch(self) -> OperationBatch | None:
        """Return the most recent journaled batch without undoing it."""
        journal = self._read_journal()
        return journal[-1] if journal else None

    def undo_last_operation(self) -> OperationBatch | None:
        """Revert the most recent journaled batch.

        Transactions are reverted in reverse order. After the call, each
        transaction's ``executed`` flag is ``False`` if it was reverted and
        stays ``True`` (with an explanatory ``note``) if it could not be.

        Returns:
            The reverted batch, or ``None`` when the journal is empty.
        """
        journal = self._read_journal()
        if not journal:
            return None
        batch = journal.pop()

        for transaction in reversed(batch.transactions):
            if transaction.executed:
                self._revert(transaction)
        for directory in sorted(batch.created_dirs, key=len, reverse=True):
            try:
                os.rmdir(directory)
            except OSError:
                # Not empty (or already gone): leave it in place.
                continue
        self._write_journal(journal)
        return batch

    @staticmethod
    def _revert(transaction: FileTransaction) -> None:
        """Undo one executed transaction if it is safe to do so."""
        source = Path(transaction.source)
        destination = Path(transaction.destination)
        try:
            if not os.path.lexists(destination):
                transaction.note = "Undo skipped: destination no longer exists"
                return
            if transaction.mode is FileActionMode.MOVE:
                if os.path.lexists(source):
                    transaction.note = "Undo skipped: original location is occupied"
                    return
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destination), str(source))
            else:
                stat = destination.stat()
                unchanged = (
                    stat.st_ino == transaction.dest_inode and stat.st_size == transaction.dest_size
                )
                if not unchanged:
                    transaction.note = "Undo skipped: copy was modified, kept"
                    return
                destination.unlink()
            transaction.executed = False
            transaction.note = "Reverted"
        except OSError as exc:
            transaction.note = f"Undo failed: {exc}"

    # --------------------------------------------------------------- journal

    def linked_copies(self) -> set[str]:
        """Absolute paths of copies/hardlinks created by journaled batches.

        The scanner excludes these so a ``Best_Of`` or by-scene *copy* is not
        re-ingested as a new photo (and immediately flagged as a duplicate).
        """
        return {
            transaction.destination
            for batch in self._read_journal()
            for transaction in batch.transactions
            if transaction.executed and transaction.mode is not FileActionMode.MOVE
        }

    def _read_journal(self) -> list[OperationBatch]:
        """Load all journaled batches; a missing or corrupt file yields none."""
        try:
            raw = json.loads(self._journal_path.read_text(encoding="utf-8"))
            return [OperationBatch.from_dict(b) for b in raw.get("batches", [])]
        except FileNotFoundError:
            return []
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return []

    def _write_journal(self, batches: list[OperationBatch]) -> None:
        """Persist the journal atomically, or delete it when empty."""
        if not batches:
            self._journal_path.unlink(missing_ok=True)
            return
        payload = {
            "version": _JOURNAL_VERSION,
            "batches": [batch.to_dict() for batch in batches],
        }
        temporary = self._journal_path.with_name(self._journal_path.name + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(temporary, self._journal_path)


def _safe_folder_name(name: str) -> str:
    """Replace characters that are invalid in folder names on any OS."""
    cleaned = "".join("_" if ch in '<>:"/\\|?*' or ord(ch) < 32 else ch for ch in name)
    return cleaned.strip(" .") or "Untitled"
