"""Desktop launcher: runs PhotoSeder in its own window instead of a browser tab.

One program, four modes (selected by command-line flags):

========================  ==================================================
Mode                      What it does
========================  ==================================================
*(no flag)*               Starts the Streamlit server in a child process on a
                          free ``127.0.0.1`` port, opens a window, and stops
                          the server when the window is closed.
``--server PORT``         Runs the Streamlit server (used by the mode above).
``--pick-folder OUTFILE`` Shows the native folder dialog and writes the chosen
                          path to ``OUTFILE`` (used by the sidebar's Browse
                          button).
``--smoke-test [REPORT]`` Exercises the engine and the server end to end and
                          exits 0 on success. Used by CI on the packaged build.
========================  ==================================================

Design notes:
    * **Two processes.** Streamlit installs signal handlers and wants the main
      thread; so does a GUI window. Running the server as a child process of
      the same executable avoids the conflict and makes shutdown clean.
    * **Window.** ``pywebview`` shows the page in the system web view (Edge
      WebView2 on Windows). A "Starting..." page is shown immediately and
      replaced by the app once the server answers its health check. When no
      window can be created, the default browser is used and a small tkinter
      window offers "Quit".
    * **Privacy.** The server listens on ``127.0.0.1`` only, usage statistics
      are disabled, and the health checks bypass any HTTP proxy.
    * **Diagnostics.** Windowed Windows builds have no console, so the server's
      output goes to ``server.log`` in the user's application-data folder; its
      path is shown if start-up fails.

This module imports only the standard library at import time so that the UI can
use :func:`pick_folder_dialog` and :func:`tkinter_available` cheaply.
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import IO

APP_TITLE = "PhotoSeder"
HOST = "127.0.0.1"
HEALTH_PATH = "/_stcore/health"
STARTUP_TIMEOUT_SECONDS = 120.0
PRIMARY_COLOR = "#2f6f5e"

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_SPLASH_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>PhotoSeder</title>
<style>body{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;
flex-direction:column;font-family:Segoe UI,Arial,sans-serif;background:#f6f7f6;color:#1f2328}
h1{font-weight:600;margin:0 0 .4em}p{margin:0;color:#6b7280}
@media (prefers-color-scheme:dark){body{background:#1b1f1d;color:#e6e8e6}}</style></head>
<body><h1>PhotoSeder</h1><p>Starting&hellip; the first start can take a few seconds.</p>
</body></html>"""

_ERROR_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>PhotoSeder</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;padding:3em;max-width:44em;margin:auto;
color:#1f2328;background:#f6f7f6}}code{{background:#e6e8e6;padding:.1em .3em}}</style></head>
<body><h1>PhotoSeder could not start</h1><p>The internal server did not respond.</p>
<p>Details are in the log file:<br><code>{log}</code></p></body></html>"""


# --------------------------------------------------------------------- helpers


def is_frozen() -> bool:
    """Return ``True`` when running from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def app_script_path() -> Path:
    """Location of the Streamlit script (also valid inside the bundle)."""
    return Path(__file__).resolve().parent / "ui" / "app.py"


def self_command(*arguments: str) -> list[str]:
    """Command line that starts this program again with ``arguments``."""
    if is_frozen():
        return [sys.executable, *arguments]
    return [sys.executable, "-m", "photoseder", *arguments]


def child_environment() -> dict[str, str]:
    """Environment for child processes.

    From source, the repository root is added to ``PYTHONPATH`` so that
    ``python -m photoseder`` works without an editable install. Localhost is
    always excluded from proxies.
    """
    env = os.environ.copy()
    if not is_frozen():
        root = str(Path(__file__).resolve().parents[1])
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = root + os.pathsep + existing if existing else root
    bypass = f"{HOST},localhost"
    existing_bypass = env.get("NO_PROXY") or env.get("no_proxy")
    env["NO_PROXY"] = f"{existing_bypass},{bypass}" if existing_bypass else bypass
    return env


def find_free_port() -> int:
    """Ask the operating system for an unused TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((HOST, 0))
        return int(probe.getsockname()[1])


def streamlit_arguments(port: int, script: Path | None = None) -> list[str]:
    """Arguments for ``streamlit`` that run the app as a local desktop server."""
    return [
        "run",
        str(script or app_script_path()),
        "--server.port",
        str(port),
        "--server.address",
        HOST,
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
        "--global.developmentMode",
        "false",
        "--client.toolbarMode",
        "minimal",
        "--theme.primaryColor",
        PRIMARY_COLOR,
    ]


def log_directory() -> Path:
    """Folder for the server log (created on demand)."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "state")
    directory = Path(base) / APP_TITLE
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def wait_until_healthy(
    url: str,
    timeout: float = STARTUP_TIMEOUT_SECONDS,
    process: subprocess.Popen[bytes] | None = None,
) -> bool:
    """Poll ``url`` until it answers ``ok``.

    Args:
        url: Health-check URL.
        timeout: Seconds to wait in total.
        process: Server process; waiting stops early if it exits.

    Returns:
        ``True`` once the server is healthy, ``False`` on timeout or if the
        server process died.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return False
        try:
            with opener.open(url, timeout=2) as response:
                if response.status == 200 and response.read().strip() == b"ok":
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(0.25)
    return False


def start_server(port: int, log: IO[bytes]) -> subprocess.Popen[bytes]:
    """Start the Streamlit server as a child process, logging to ``log``."""
    return subprocess.Popen(
        self_command("--server", str(port)),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=child_environment(),
        creationflags=_NO_WINDOW,
    )


def stop_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate a child process, escalating to kill after five seconds."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def show_error(message: str) -> None:
    """Show an error where the user can see it (windowed builds have no console)."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, APP_TITLE, 0x10)
            return
        except (AttributeError, OSError):
            pass
    print(message, file=sys.stderr)


# --------------------------------------------------------------- folder dialog


def tkinter_available() -> bool:
    """Return ``True`` when the native folder dialog can be shown."""
    return importlib.util.find_spec("tkinter") is not None


def pick_folder(initial: str = "") -> str | None:
    """Show the native folder dialog in this process.

    Args:
        initial: Folder the dialog starts in, if it exists.

    Returns:
        The chosen folder, or ``None`` if the dialog was cancelled.
    """
    import tkinter
    from tkinter import filedialog

    root = tkinter.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except tkinter.TclError:
        pass  # Not every window manager supports it; the dialog still works.
    start = initial if initial and Path(initial).is_dir() else None
    try:
        chosen = filedialog.askdirectory(
            parent=root,
            title="Choose your photo folder",
            initialdir=start,
            mustexist=True,
        )
    finally:
        root.destroy()
    return str(Path(chosen)) if chosen else None


def pick_folder_main(out_file: Path, initial: str) -> int:
    """``--pick-folder`` mode: write the chosen folder (or nothing) to a file."""
    try:
        chosen = pick_folder(initial)
    except ImportError:
        return 2
    out_file.write_text(chosen or "", encoding="utf-8")
    return 0


def pick_folder_dialog(initial: str = "") -> str | None:
    """Open the folder dialog from the Streamlit server process.

    The dialog runs in its own process because tkinter is unreliable off the
    main thread, and its answer travels through a temporary file because a
    windowed Windows build has no usable standard output.

    Args:
        initial: Folder the dialog starts in.

    Returns:
        The chosen folder, or ``None`` if cancelled or unavailable.
    """
    with tempfile.TemporaryDirectory() as directory:
        out_file = Path(directory) / "choice.txt"
        command = self_command("--pick-folder", str(out_file), "--initial", initial)
        try:
            subprocess.run(
                command,
                check=False,
                timeout=900,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=child_environment(),
                creationflags=_NO_WINDOW,
            )
            choice = out_file.read_text(encoding="utf-8").strip()
        except (OSError, subprocess.TimeoutExpired):
            return None
    return choice or None


# -------------------------------------------------------------------- server


def run_server(port: int) -> int:
    """``--server`` mode: run Streamlit until it is terminated."""
    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", *streamlit_arguments(port)]
    streamlit_cli.main()
    return 0


# -------------------------------------------------------------------- windows


def _run_in_window(server: subprocess.Popen[bytes], url: str, log_file: Path) -> bool:
    """Show the app in a native window; return ``False`` if none can be made."""
    try:
        import webview

        try:
            webview.settings["ALLOW_DOWNLOADS"] = True
        except (AttributeError, KeyError, TypeError):
            pass  # Older pywebview: downloads simply stay disabled.
        window = webview.create_window(
            APP_TITLE, html=_SPLASH_HTML, width=1400, height=900, min_size=(1000, 700)
        )

        def load_app() -> None:
            if wait_until_healthy(url + HEALTH_PATH, process=server):
                window.load_url(url)
            else:
                window.load_html(_ERROR_HTML.format(log=log_file))

        webview.start(load_app)
    except Exception:  # noqa: BLE001 - any GUI failure falls back to the browser
        return False
    return True


def _wait_in_browser_mode(server: subprocess.Popen[bytes]) -> None:
    """Keep the server alive while the app is open in the default browser."""
    try:
        import tkinter

        root = tkinter.Tk()
        root.title(APP_TITLE)
        tkinter.Label(
            root,
            text="PhotoSeder is running in your browser.\nClose this window to quit.",
            padx=24,
            pady=18,
        ).pack()
        tkinter.Button(root, text="Quit", command=root.destroy).pack(pady=(0, 18))

        def poll() -> None:
            if server.poll() is not None:
                root.destroy()
            else:
                root.after(500, poll)

        poll()
        root.mainloop()
    except Exception:  # noqa: BLE001 - no tkinter or no display: just wait
        server.wait()


def run_desktop() -> int:
    """Default mode: start the server and show the app."""
    log_file = log_directory() / "server.log"
    port = find_free_port()
    url = f"http://{HOST}:{port}"
    with log_file.open("ab") as log:
        log.write(f"\n--- {datetime.now():%Y-%m-%d %H:%M:%S} starting on port {port}\n".encode())
        log.flush()
        server = start_server(port, log)
        try:
            if _run_in_window(server, url, log_file):
                return 0
            if not wait_until_healthy(url + HEALTH_PATH, process=server):
                show_error(f"PhotoSeder could not start.\n\nDetails: {log_file}")
                return 1
            webbrowser.open(url)
            _wait_in_browser_mode(server)
            return 0
        finally:
            stop_process(server)


# ------------------------------------------------------------------ self test


def _check_engine() -> str:
    """Run the whole engine on one generated photo."""
    import numpy as np
    from PIL import Image

    from photoseder.orchestrator.pipeline import Pipeline

    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        noise = np.random.default_rng(1).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        Image.fromarray(noise).save(folder / "test.jpg", quality=92)

        pipe = Pipeline(folder)
        scan = pipe.scan_and_analyze()
        if scan.errors or scan.analyzed != 1:
            raise RuntimeError(f"analysis failed: {scan.errors or scan}")
        pipe.set_selected(next(iter(pipe.state.photos)), True)
        pipe.generate_album()
        album = pipe.export_album(output_path=folder / "album.html")
        if album.stat().st_size < 1000:
            raise RuntimeError("album export is empty")
        tags = pipe.write_metadata()
        if tags.written != 1:
            raise RuntimeError(f"EXIF writing failed: {tags}")
    return "analyzed, album exported, EXIF written"


def _check_server() -> str:
    """Start the real server child process and request its page."""
    port = find_free_port()
    url = f"http://{HOST}:{port}"
    log_file = log_directory() / "smoke-server.log"
    with log_file.open("wb") as log:
        server = start_server(port, log)
        try:
            if not wait_until_healthy(url + HEALTH_PATH, process=server):
                tail = log_file.read_text(encoding="utf-8", errors="replace")[-1500:]
                raise RuntimeError(f"server did not become healthy; log tail:\n{tail}")
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(url + "/", timeout=10) as response:
                if response.status != 200:
                    raise RuntimeError(f"index page returned {response.status}")
        finally:
            stop_process(server)
    return f"server answered on port {port}"


def run_self_test(report: Callable[[str], None]) -> bool:
    """Run the smoke test, reporting one line per check.

    Args:
        report: Receives each result line.

    Returns:
        ``True`` if every check passed.
    """
    passed = True
    for name, check in (("engine", _check_engine), ("server", _check_server)):
        try:
            report(f"PASS {name}: {check()}")
        except Exception as exc:  # noqa: BLE001 - report every failure, never crash
            passed = False
            report(f"FAIL {name}: {type(exc).__name__}: {exc}")
    report(f"INFO folder dialog (tkinter): {'available' if tkinter_available() else 'missing'}")
    return passed


def smoke_test_main(report_file: str) -> int:
    """``--smoke-test`` mode: print or write the report; exit 0 on success."""
    lines: list[str] = []

    def report(line: str) -> None:
        lines.append(line)
        if not report_file:
            print(line, flush=True)

    passed = run_self_test(report)
    if report_file:
        Path(report_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if passed else 1


# ---------------------------------------------------------------------- entry


def _ensure_streams() -> None:
    """Give windowed builds (no console) usable ``stdout`` / ``stderr``."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="photoseder", description="PhotoSeder desktop app")
    parser.add_argument("--server", type=int, metavar="PORT", help=argparse.SUPPRESS)
    parser.add_argument("--pick-folder", metavar="OUTFILE", help=argparse.SUPPRESS)
    parser.add_argument("--initial", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--smoke-test",
        nargs="?",
        const="",
        default=None,
        metavar="REPORT",
        help="check the engine and the server, then exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Program entry point; returns the process exit code."""
    multiprocessing.freeze_support()
    _ensure_streams()
    args = _parser().parse_args(argv)
    if args.server is not None:
        return run_server(args.server)
    if args.pick_folder:
        return pick_folder_main(Path(args.pick_folder), args.initial)
    if args.smoke_test is not None:
        return smoke_test_main(args.smoke_test)
    return run_desktop()
