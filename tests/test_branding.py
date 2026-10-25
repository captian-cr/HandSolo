"""产品名、版本、启动器形态、无控制台兜底。

2026-10-23 用户拍板：改名 ``HandSolo``、版本 ``0.9.0``、启动器改用 ``pythonw.exe``。
这一组测试守三件容易在后续改动中悄悄退化的事：

1. **产品名只有一个来源**（``handmouse.APP_NAME``）——改名那天就是因为展示名
   散在 5 个文件里才要一个个 grep。
2. **``__version__`` 与 ``pyproject.toml`` 不许漂移**——发 Release 时版本不
   一致最难查。
3. **``pythonw`` 没有 stdout，必须被接住**——否则任何 ``print`` 当场
   ``AttributeError``，而且用户看不到任何报错（没有控制台）。
"""

from __future__ import annotations

import io
import re
import sys
import time
from pathlib import Path

import pytest

import handmouse
from handmouse import APP_NAME, APP_NAME_CN, __main__ as entry

PROJECT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- 名字与版本


def test_product_name_is_single_source() -> None:
    assert APP_NAME == "HandSolo"
    assert APP_NAME_CN == "单手"
    # 旧名字不许再出现在展示位上（包名仍叫 handmouse，那是刻意保留的）。
    source = (PROJECT / "handmouse" / "app.py").read_text(encoding="utf-8")
    assert "HandMouse 视觉手势鼠标" not in source


def test_window_titles_follow_app_name() -> None:
    from handmouse import collect, debug_view

    assert debug_view.WINDOW_NAME.startswith(APP_NAME)
    assert collect._WINDOW.startswith(APP_NAME)
    # cv2 在 Windows 上按 GBK 解释标题，中文必然乱码——必须保持 ASCII。
    for title in (debug_view.WINDOW_NAME, collect._WINDOW):
        assert title.isascii(), title


def test_version_matches_pyproject() -> None:
    text = (PROJECT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None
    assert match.group(1) == handmouse.__version__


def test_version_is_090_or_later() -> None:
    """0.9.0 = 第一个可以发给别人试用的预发布版；别被误改回去。"""

    parts = tuple(int(p) for p in handmouse.__version__.split("."))
    assert parts >= (0, 9, 0)


# ---------------------------------------------------------------- 单实例


def test_single_instance_covers_old_name_too() -> None:
    """改名过渡期：旧名字也必须占，否则新旧两个版本会同时抢摄像头。"""

    assert f"{APP_NAME}.SingleInstance.v1" in entry._MUTEX_NAMES
    assert "HandMouse.SingleInstance.v1" in entry._MUTEX_NAMES


def test_acquire_one_reports_already_exists() -> None:
    # 同一进程内用同一个名字再占一次，Windows 会返回 ERROR_ALREADY_EXISTS。
    name = f"HandSolo.Test.{time.time_ns()}"
    first_ok, first_handle = entry._acquire_one(name)
    second_ok, second_handle = entry._acquire_one(name)
    assert first_ok is True and first_handle != 0
    assert second_ok is False
    assert second_handle == 0


# ---------------------------------------------------------------- 无控制台兜底


#: ⚠️ 必须**在测试函数体内**把 stdout/stderr 置 None，不能用 fixture：
#: pytest 的全局捕获会在 fixture 之后（runtest 开始时）重新给 ``sys.stdout``
#: 赋值，fixture 里的 monkeypatch 会被覆盖掉，表现为"加 -s 就过、不加就挂"。
def _pretend_pythonw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)


def test_ensure_output_redirects_when_no_console(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _pretend_pythonw(monkeypatch)
    path = entry._ensure_output(tmp_path)
    assert path is not None and path.exists()
    assert sys.stdout is not None and sys.stderr is not None
    # 关键：重定向之后 print 不再炸（旧代码就是在这里 AttributeError 的）。
    print("hello from pythonw")
    sys.stdout.flush()
    assert "hello from pythonw" in path.read_text(encoding="utf-8")


def test_ensure_output_is_noop_with_console(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    assert entry._ensure_output() is None
    assert sys.stdout is buffer


def test_ensure_output_survives_unwritable_log_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """连日志都开不出来时也不能抛——宁可吞掉输出，也要让程序起来。"""

    _pretend_pythonw(monkeypatch)

    def boom(*args: object, **kwargs: object):
        raise OSError("磁盘满了")

    monkeypatch.setattr(entry.Path, "open", boom, raising=False)
    assert entry._ensure_output(tmp_path) is None
    # 退化成内存黑洞，但 print 依然安全。
    print("still alive")


# ---------------------------------------------------------------- 启动器形态


@pytest.mark.parametrize(
    "filename",
    ["启动 HandSolo.bat", "HandSolo（后台无预览）.bat"],
)
def test_silent_launchers_use_pythonw(filename: str) -> None:
    path = PROJECT / filename
    if not path.exists():
        pytest.skip("启动器尚未生成（先跑 scripts/make_launchers.py）")
    text = io.open(path, encoding="gbk").read()
    assert "pythonw.exe" in text
    assert 'start ""' in text
    # `python.exe` 会让黑窗口回来；`start /b` 会让子进程跟着父 cmd 一起没。
    assert "python.exe" not in text
    assert "start /b" not in text


def test_diagnostic_launcher_keeps_console() -> None:
    path = PROJECT / "HandSolo（诊断·看报错）.bat"
    if not path.exists():
        pytest.skip("启动器尚未生成（先跑 scripts/make_launchers.py）")
    text = io.open(path, encoding="gbk").read()
    assert "python.exe" in text
    assert "pythonw.exe" not in text
