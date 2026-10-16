"""app.py 环境指纹辅助函数的单测（``_worktree_mark``）。"""

from __future__ import annotations

import subprocess

from handmouse import app


def _fake_run(stdout: str):
    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    return runner


def test_worktree_mark_covers_untracked_files(monkeypatch):
    """未跟踪的新代码文件也必须点亮 ``+dirty``。

    ``--untracked-files=no`` 会让"新代码写在未提交的新文件里"漏网：commit hash
    相同、dirty 却报 clean，正是"同 hash 两轮数据打架"的盲区。``logs/`` 已
    gitignore，默认 untracked 模式不会把它算进来，故无需再显式关掉未跟踪上报。
    """

    def runner(argv, **kwargs):
        # 模拟 git：只有带 --untracked-files=no 时才会吞掉 ``??`` 行。
        out = "" if "--untracked-files=no" in argv else "?? new_module.py\n"
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(app.subprocess, "run", runner)
    assert app._worktree_mark() == "+dirty"


def test_worktree_mark_dirty_on_tracked_modification(monkeypatch):
    monkeypatch.setattr(app.subprocess, "run", _fake_run(" M handmouse/app.py\n"))
    assert app._worktree_mark() == "+dirty"


def test_worktree_mark_clean_when_worktree_clean(monkeypatch):
    monkeypatch.setattr(app.subprocess, "run", _fake_run(""))
    assert app._worktree_mark() == ""


def test_worktree_mark_clean_when_git_unavailable(monkeypatch):
    def boom(argv, **kwargs):
        raise OSError("git 不存在")

    monkeypatch.setattr(app.subprocess, "run", boom)
    assert app._worktree_mark() == ""
