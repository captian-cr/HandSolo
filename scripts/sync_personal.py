"""把源码版的个人数据同步给 exe 版（``dist/personal/``）。

## 为什么需要这一步

同一台机器上有两条启动路径：

* 源码版：``python -m handmouse``（桌面 ``HandSolo.bat`` 就是它）→ 读
  **项目根目录**的 ``config.json`` 和 ``gesture_data/``；
* exe 版：``dist/HandSolo.exe`` → 冻结后 ``PROJECT_ROOT`` 是 exe 所在目录。

两条路径各自读自己的数据，于是同一只手在两边是两套阈值：源码版是手标定过的
（不对称活动区、``pointer_response``、个人样本库），exe 版是出厂默认。
用户的原话是「exe 版本跟源码跑出来的效果不一样」——根因就在这里。

解决办法不是让 exe 去读源码目录（exe 可能被拷到别处），而是：

    exe 同目录有 ``personal/`` → 个人数据（config / gesture_data / logs）
                                  全部落在里面；
    没有 ``personal/``         → 干净起步（发布给陌生人时的形态）。

本脚本负责把源码版这份**真源**同步进 ``dist/personal/``。改完源码版的调参、
重跑过向导之后，跑一次本脚本，exe 就跟着一致了。

## 用法

双击 ````同步个人设置到exe版.bat````，或者：

    python scripts/sync_personal.py

## 不做什么

* **不动** ``dist/config.json``（那是发布基线，必须是出厂默认值）。
* **不动** 任何 ``.bat``（包括桌面上那份）。
* 不删目标目录里已有的东西，只覆盖同名文件。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

#: 需要同步的个人数据。日志不同步——那是运行产物，两边各记各的更清楚。
ITEMS = ("config.json", "gesture_data")


def _project_root() -> Path:
    """项目根目录。从脚本位置推导，别写死盘符。"""

    return Path(__file__).resolve().parent.parent


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n/1:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}GB"


def sync(project_root: Path | None = None, *, quiet: bool = False) -> int:
    """执行同步。返回 0 成功、1 失败。"""

    root = project_root or _project_root()
    dist = root / "dist"
    target = dist / "personal"

    def say(text: str = "") -> None:
        if not quiet:
            print(text)

    if not dist.is_dir():
        print(f"[同步] 找不到 {dist}")
        print("[同步] 先打包出 exe（pyinstaller handmouse.spec），再来同步个人数据。")
        return 1

    source_config = root / "config.json"
    if not source_config.is_file():
        print(f"[同步] 找不到源码版配置 {source_config}，先启动一次源码版让它生成。")
        return 1

    target.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    # ---- 1. 配置 ----
    shutil.copy2(source_config, target / "config.json")
    copied.append(f"config.json（{_human_size(source_config.stat().st_size)}）")

    # ---- 2. 个人手势库 ----
    source_data = root / "gesture_data"
    if source_data.is_dir():
        target_data = target / "gesture_data"
        target_data.mkdir(parents=True, exist_ok=True)
        frames = 0
        for item in sorted(source_data.iterdir()):
            if not item.is_file():
                continue
            shutil.copy2(item, target_data / item.name)
            copied.append(f"gesture_data/{item.name}（{_human_size(item.stat().st_size)}）")
            if item.suffix == ".jsonl" and item.stem.startswith("samples"):
                try:
                    frames = max(
                        frames,
                        sum(1 for line in item.open(encoding="utf-8") if line.strip()),
                    )
                except OSError:
                    pass
        if frames:
            copied.append(f"  └ 样本 {frames} 帧")
    else:
        say("[同步] 源码版还没有 gesture_data/，exe 会以「空样本库」运行。")

    say(f"[同步] 已写入 {target}")
    for line in copied:
        say(f"       {line}")
    say()
    say("[同步] exe 下次启动就会用这份数据，与源码版手感一致。")
    say("[同步] 发布用的 dist/config.json 没有被改动，仍是出厂默认值。")
    return 0


def main(argv: list[str] | None = None) -> int:
    return sync()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
