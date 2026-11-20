"""Tests for the desktop launcher and the Browse button."""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from photoseder import desktop

APP = str(desktop.app_script_path())


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def health_server():
    server = HTTPServer((desktop.HOST, 0), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://{desktop.HOST}:{server.server_address[1]}"
    server.shutdown()


def test_find_free_port_is_bindable() -> None:
    port = desktop.find_free_port()
    with socket.socket() as probe:
        probe.bind((desktop.HOST, port))


def test_streamlit_arguments_are_local_and_private() -> None:
    args = desktop.streamlit_arguments(8123)
    assert args[0] == "run" and args[1].endswith("app.py")

    def value(flag: str) -> str:
        return args[args.index(flag) + 1]

    assert value("--server.port") == "8123"
    assert value("--server.address") == "127.0.0.1"
    assert value("--server.headless") == "true"
    assert value("--browser.gatherUsageStats") == "false"
    assert value("--global.developmentMode") == "false"


def test_app_script_exists() -> None:
    assert desktop.app_script_path().is_file()


def test_self_command_from_source_and_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    assert desktop.self_command("--x")[-3:] == ["-m", "photoseder", "--x"]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert desktop.self_command("--x") == [sys.executable, "--x"]


def test_child_environment_bypasses_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_PROXY", "example.org")
    env = desktop.child_environment()
    assert "127.0.0.1" in env["NO_PROXY"] and "example.org" in env["NO_PROXY"]
    assert str(Path(desktop.__file__).resolve().parents[1]) in env["PYTHONPATH"]


def test_wait_until_healthy(health_server: str) -> None:
    assert desktop.wait_until_healthy(health_server, timeout=5)


def test_wait_until_healthy_times_out_and_stops_if_process_died() -> None:
    unused = f"http://{desktop.HOST}:{desktop.find_free_port()}"
    assert not desktop.wait_until_healthy(unused, timeout=0.6)

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    assert not desktop.wait_until_healthy(unused, timeout=30, process=dead)


def test_stop_process_terminates_child() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    desktop.stop_process(child)
    assert child.poll() is not None


def test_pick_folder_main_writes_choice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(desktop, "pick_folder", lambda initial="": str(tmp_path))
    out = tmp_path / "out.txt"
    assert desktop.pick_folder_main(out, "") == 0
    assert out.read_text(encoding="utf-8") == str(tmp_path)


def test_pick_folder_main_without_tkinter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def missing(initial: str = "") -> str:
        raise ImportError("no tkinter")

    monkeypatch.setattr(desktop, "pick_folder", missing)
    assert desktop.pick_folder_main(tmp_path / "out.txt", "") == 2


def test_pick_folder_uses_tkinter_dialog(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeRoot:
        def withdraw(self) -> None: ...
        def attributes(self, *args: object) -> None: ...
        def destroy(self) -> None: ...

    fake_tk = types.ModuleType("tkinter")
    fake_tk.Tk = FakeRoot
    fake_tk.TclError = RuntimeError
    fake_dialog = types.ModuleType("tkinter.filedialog")
    fake_dialog.askdirectory = lambda **kwargs: str(tmp_path)
    fake_tk.filedialog = fake_dialog
    monkeypatch.setitem(sys.modules, "tkinter", fake_tk)
    monkeypatch.setitem(sys.modules, "tkinter.filedialog", fake_dialog)

    assert desktop.pick_folder(str(tmp_path)) == str(tmp_path)
    fake_dialog.askdirectory = lambda **kwargs: ""
    assert desktop.pick_folder() is None


def test_pick_folder_dialog_reads_child_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(command: list[str], **kwargs: object) -> None:
        assert "--pick-folder" in command
        Path(command[command.index("--pick-folder") + 1]).write_text(
            str(tmp_path), encoding="utf-8"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert desktop.pick_folder_dialog("") == str(tmp_path)

    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: None)
    assert desktop.pick_folder_dialog("") is None  # child wrote nothing


def test_smoke_test_reports_to_file(tmp_path: Path) -> None:
    report = tmp_path / "report.txt"
    assert desktop.smoke_test_main(str(report)) == 0
    text = report.read_text(encoding="utf-8")
    assert "PASS engine" in text and "PASS server" in text


# ------------------------------------------------------------ Browse button


@pytest.fixture
def photo_folder(tmp_path: Path, make_photo) -> Path:
    make_photo("one.jpg", seed=1)
    return tmp_path


def test_browse_button_scans_the_chosen_folder(
    monkeypatch: pytest.MonkeyPatch, photo_folder: Path
) -> None:
    monkeypatch.setattr(desktop, "tkinter_available", lambda: True)
    monkeypatch.setattr(desktop, "pick_folder_dialog", lambda initial="": str(photo_folder))

    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception
    browse = next(b for b in at.sidebar.button if b.label == "Browse...")
    browse.click().run()

    assert not at.exception
    assert at.session_state["folder_text"] == str(photo_folder)
    assert len(at.tabs) == 4  # the scan ran and the four workflow tabs appeared


def test_cancelled_browse_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(desktop, "tkinter_available", lambda: True)
    monkeypatch.setattr(desktop, "pick_folder_dialog", lambda initial="": None)

    at = AppTest.from_file(APP, default_timeout=90).run()
    next(b for b in at.sidebar.button if b.label == "Browse...").click().run()

    assert not at.exception
    assert at.session_state["folder_text"] == ""
    assert len(at.tabs) == 0


def test_browse_button_hidden_without_tkinter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(desktop, "tkinter_available", lambda: False)
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert [b.label for b in at.sidebar.button] == ["Open & scan"]


def test_bad_folder_keeps_its_error_visible() -> None:
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.sidebar.text_input[0].set_value("/definitely/not/a/folder")
    next(b for b in at.sidebar.button if b.label == "Open & scan").click().run()

    assert [e.value for e in at.sidebar.error] == [
        "Cannot open folder: Not a folder: /definitely/not/a/folder"
    ]
