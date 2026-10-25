"""验证启动器：编码能进中文目录 + 六份 .bat 内容正确（不真的启动摄像头）。

三段检查：

1. **运行验证**：在 cmd.exe 里跑同一套 ``cd /d "%~dp0"`` + GBK 编码 + 中文路径，
   确认 ``--help`` 能正常输出（这一条保证"双击进得去"，不是靠眼看的）。
2. **静默验证**：主入口必须是 ``pythonw.exe`` + ``start ""``——只要出现
   ``python.exe``（不是 pythonw）就当失败，否则黑窗口又回来了。
3. **内容验证**：读六份 .bat（GBK 解码），确认参数对得上：
   后台版必须有 ``--no-preview``，主入口**不许**有；诊断版必须是 ``python.exe``
   （它就是要看报错）。

⚠️ 2026-10-23 起主入口不再有 ``if errorlevel`` + ``pause``：``start`` 是立刻返回的，
errorlevel 拿不到子进程结果。失败改由 ``handmouse/__main__.py`` 弹系统对话框
告知，所以这里也不再断言 pause。
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
VERIFY = PROJECT / "_verify_launcher.bat"

#: 用当前解释器，别写死路径（写死只对作者本机成立，别人 clone 后跑不起来）。
BODY = """@echo off
cd /d "%~dp0"
echo CWD=%CD%
"{python}" -m handmouse --help
echo EXITCODE=%errorlevel%
""".format(python=sys.executable)


def _read_gbk(path: Path) -> str:
    return io.open(path, encoding="gbk").read()


def main() -> int:
    failures: list[str] = []

    # ---- 1. 运行验证（编码 + 中文目录 + 参数）----
    io.open(VERIFY, "w", encoding="gbk", newline="\r\n").write(BODY)
    try:
        result = subprocess.run(
            ["cmd", "/c", str(VERIFY)],
            capture_output=True,
            text=True,
            encoding="gbk",
            errors="replace",
            cwd=str(PROJECT),
        )
    finally:
        VERIFY.unlink(missing_ok=True)

    out = (result.stdout or "") + (result.stderr or "")
    if "EXITCODE=0" not in out or "usage: handmouse" not in out:
        failures.append("cmd 里跑不通（编码/目录问题）")
    if "--no-preview" not in out:
        failures.append("--help 里没有 --no-preview（参数没生效）")
    if "HandSolo" not in out:
        failures.append("--help 里没有出现 HandSolo（改名没生效）")

    # ---- 2 & 3. 内容验证 ----
    desktop = Path(os.path.expandvars(r"%USERPROFILE%\Desktop"))
    startup = Path(
        os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup")
    )
    silent = ['cd /d "%~dp0"', "pythonw.exe", 'start ""', "-m handmouse"]
    #: 桌面/启动文件夹必须写死项目路径——它们的 ``%~dp0`` 是桌面和启动文件夹，
    #: 在那儿 ``-m handmouse`` 导不进来，还会在桌面造一个 logs/。
    remote = [f'cd /d "{PROJECT}"', "pythonw.exe", 'start ""', "-m handmouse"]
    checks = [
        (PROJECT / "启动 HandSolo.bat", silent, {"no": "--no-preview", "no_console": True}),
        (
            PROJECT / "HandSolo（后台无预览）.bat",
            silent,
            {"yes": "--no-preview", "no_console": True},
        ),
        (
            PROJECT / "HandSolo（诊断·看报错）.bat",
            ['cd /d "%~dp0"', "python.exe", "-m handmouse", "pause"],
            {"console": True},
        ),
        (
            # 首次设置向导：必须 python.exe（要看见录了多少帧），且必须带 --onboarding
            PROJECT / "HandSolo（首次设置）.bat",
            ['cd /d "%~dp0"', "python.exe", "-m handmouse", "--onboarding"],
            {"console": True, "yes": "--onboarding"},
        ),
        (desktop / "HandSolo.bat", remote, {"no": "--no-preview", "no_console": True}),
        (startup / "HandSolo（开机自启）.bat", remote, {"no_console": True}),
    ]
    for path, needles, rules in checks:
        if not path.exists():
            failures.append(f"缺少 {path}")
            continue
        text = _read_gbk(path)
        for needle in [n for n in needles if n]:
            if needle not in text:
                failures.append(f"{path.name} 里找不到 {needle!r}")
        if rules.get("no") and rules["no"] in text:
            failures.append(f"{path.name} 不该包含 {rules['no']!r}")
        if rules.get("yes") and rules["yes"] not in text:
            failures.append(f"{path.name} 缺少 {rules['yes']!r}")
        if rules.get("no_console") and "pythonw.exe" not in text:
            failures.append(f"{path.name} 没用 pythonw.exe，黑窗口会回来")
        if rules.get("console") and "pythonw.exe" in text:
            failures.append(f"{path.name} 是诊断入口，必须用 python.exe 才看得到报错")
        if path.parent != PROJECT and '"%~dp0"' in text:
            failures.append(f"{path.name} 在项目外，不能用 %~dp0（会在那里找不到包）")

    print(out.strip()[:400])
    print()
    if failures:
        for line in failures:
            print("✗", line)
        return 1
    print("✓ 启动器可用：编码正确、六份文件齐全、静默/诊断/首次设置入口分工正确")
    return 0


if __name__ == "__main__":
    sys.exit(main())
