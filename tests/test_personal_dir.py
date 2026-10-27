"""exe 版的个人数据目录解析 + 同步脚本的纯逻辑测试。

背景：exe 与源码版各自读自己的 ``config.json``，同一只手在两边是两套阈值。
``personal/`` 目录就是让 exe 与源码版共用同一份个人数据的开关——
它的判定一旦错了，用户看到的是"exe 版没毛病但就是不跟手"，很难归因到配置上。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from handmouse import config as config_module
from scripts import sync_personal


def _fake_frozen(monkeypatch: pytest.MonkeyPatch, exe_path: Path) -> None:
    """把当前进程伪装成 PyInstaller 单文件 exe。"""

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_path))


def test_source_mode_ignores_personal_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """源码模式必须完全不受影响：即使旁边有 personal/，仍用项目根目录。

    🔴 这条是有代价的教训——源码版是作者天天在用的那条路径，
    为了修 exe 而让源码版改读别的目录，等于拿主路径冒风险。
    """

    monkeypatch.delattr(sys, "frozen", raising=False)
    (tmp_path / config_module.PERSONAL_DIR_NAME).mkdir()

    assert config_module._base_dir() == Path(config_module.__file__).resolve().parent.parent


def test_frozen_without_personal_uses_exe_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """发布包形态：没有 personal/ → 就在 exe 旁边干净起步。"""

    exe = tmp_path / "HandSolo.exe"
    exe.write_bytes(b"stub")
    _fake_frozen(monkeypatch, exe)

    assert config_module._base_dir() == tmp_path


def test_frozen_with_personal_uses_personal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """个人版形态：有 personal/ → 配置、手势库、日志全部落在里面。"""

    exe = tmp_path / "HandSolo.exe"
    exe.write_bytes(b"stub")
    personal = tmp_path / config_module.PERSONAL_DIR_NAME
    personal.mkdir()
    _fake_frozen(monkeypatch, exe)

    assert config_module._base_dir() == personal


def test_personal_dir_name_is_plain(tmp_path: Path) -> None:
    """目录名不带点、不带空格：用户要能自己在资源管理器里看见它。"""

    assert config_module.PERSONAL_DIR_NAME == "personal"
    assert config_module.PERSONAL_DIR_NAME.isidentifier()


def test_sync_copies_config_and_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """同步脚本要把配置和样本库一起搬过去，少一样 exe 就退回默认手感。"""

    root = tmp_path / "proj"
    (root / "dist").mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps({"drag_hold_ms": 1000, "onboarding_done": True}),
        encoding="utf-8",
    )
    data = root / "gesture_data"
    data.mkdir()
    (data / "samples.jsonl").write_text('{"label": "move"}\n{"label": "move"}\n', encoding="utf-8")
    (data / "onboarding.json").write_text('{"speed": {"p50": 0.1}}', encoding="utf-8")

    assert sync_personal.sync(root, quiet=True) == 0

    target = root / "dist" / "personal"
    assert json.loads((target / "config.json").read_text(encoding="utf-8"))["drag_hold_ms"] == 1000
    assert (target / "gesture_data" / "samples.jsonl").is_file()
    assert (target / "gesture_data" / "onboarding.json").is_file()


def test_sync_leaves_release_config_untouched(tmp_path: Path) -> None:
    """🔴 发布基线不能被动：dist/config.json 必须保持出厂默认，只有 personal/ 有个人数据。

    混进来的后果是"陌生人下载即用作者的阈值"——他手大你手小，第一天就劝退。
    """

    root = tmp_path / "proj"
    (root / "dist").mkdir(parents=True)
    release = root / "dist" / "config.json"
    release.write_text('{"drag_hold_ms": 5000}', encoding="utf-8")
    (root / "config.json").write_text('{"drag_hold_ms": 1000}', encoding="utf-8")
    (root / "gesture_data").mkdir()
    (root / "gesture_data" / "samples.jsonl").write_text("{}\n", encoding="utf-8")

    assert sync_personal.sync(root, quiet=True) == 0

    assert json.loads(release.read_text(encoding="utf-8"))["drag_hold_ms"] == 5000


def test_sync_fails_loudly_without_dist(tmp_path: Path) -> None:
    """没打过包就同步 → 明确报错，而不是静默创建一个没人用的目录。"""

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    assert sync_personal.sync(tmp_path, quiet=True) == 1
    assert not (tmp_path / "dist").exists()


def test_sync_fails_loudly_without_source_config(tmp_path: Path) -> None:
    """源码版还没生成配置 → 报错，别让 exe 拿到一份空配置。"""

    (tmp_path / "dist").mkdir()

    assert sync_personal.sync(tmp_path, quiet=True) == 1
