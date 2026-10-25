"""HandSolo —— 为一只手校准的手势鼠标。

引擎层移植自 `KIdiens/Vimoose <https://github.com/KIdiens/Vimoose>`_（MIT）。

🔴 **产品名只在这里定义一次**。2026-10-23 从 ``HandMouse`` 改名成 ``HandSolo``
时，展示名散落在 5 个文件里、要一个个 grep 才找全；所以现在窗口标题、命令行
说明、启动器说明一律引用 ``APP_NAME`` / ``APP_NAME_CN``，不许再硬写字符串。

⚠️ Python 包名仍叫 ``handmouse``（77 个 .py 里有 ``import handmouse``）。
产品名和包名不一致在开源界很常见（VS Code 的命令行就叫 ``code``），
改包名留到 1.0 一次性做完，避免为了好看引入一批 import 迁移风险。
"""

from __future__ import annotations

#: 对外英文产品名（GitHub 仓库名、Release 包名、窗口标题都用它）。
APP_NAME = "HandSolo"

#: 对外中文产品名（给人看的地方用）。
APP_NAME_CN = "单手"

#: 一句话定位，README 和 ``--help`` 共用。
APP_TAGLINE = "为一只手校准的鼠标"

__version__ = "0.11.0"

__all__ = ["APP_NAME", "APP_NAME_CN", "APP_TAGLINE", "__version__"]
