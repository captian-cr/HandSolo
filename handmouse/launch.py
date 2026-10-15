"""LAUNCH / 热键动作派发。

安全边界：目标只允许来自 ``config.launch_whitelist``。轨迹手势识别错一次
的代价不能是"随手打开任意 EXE"，所以白名单是硬约束而不是提示。

``dry_run=True``（``--safe-mode``）时只记录将要执行的动作、不产生任何系统
副作用，用来先跑通流程再接管输入。
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass

from handmouse.config import AppConfig
from handmouse.contracts import ActionCommand, ActionType
from handmouse.mouse import press_hotkey

LOGGER = logging.getLogger(__name__)


class LaunchRejected(Exception):
    """目标不在白名单内。"""


@dataclass(slots=True)
class DispatchResult:
    """一次动作派发的结果，供 HUD 显示与日志排查。"""

    ok: bool
    message: str


class LaunchDispatcher:
    """执行轨迹手势绑定到 White-List 的动作。"""

    def __init__(self, config: AppConfig, dry_run: bool = False) -> None:
        self.config = config
        self.dry_run = bool(dry_run)
        self.history: list[str] = []

    def update_config(self, config: AppConfig) -> None:
        self.config = config

    def resolve_launch_target(self, target: str) -> str:
        """别名或白名单内的完整命令 → 可直接执行的命令；否则拒绝。"""

        key = target.strip()
        whitelist = self.config.launch_whitelist
        if key.lower() in whitelist:
            return whitelist[key.lower()]
        if key in whitelist.values():
            return key
        raise LaunchRejected(f"目标 {target!r} 不在 launch_whitelist 内")

    def dispatch(self, command: ActionCommand) -> DispatchResult:
        if command.kind == ActionType.CANCEL:
            return DispatchResult(True, "已取消")

        if command.kind == ActionType.LAUNCH:
            if not self.config.launch_enabled:
                return DispatchResult(False, "LAUNCH 已关闭")
            try:
                resolved = self.resolve_launch_target(command.target)
            except LaunchRejected as error:
                LOGGER.warning("拒绝执行：%s", error)
                return DispatchResult(False, str(error))
            if self.dry_run:
                return self._record(f"[dry-run] 启动 {command.name} → {resolved}")
            return self._launch(command.name, resolved)

        if command.kind == ActionType.HOTKEY:
            if self.dry_run:
                return self._record(f"[dry-run] 热键 {command.name} → {command.target}")
            return self._hotkey(command.name, command.target)

        return DispatchResult(False, f"未知动作类型：{command.kind}")

    def _record(self, message: str) -> DispatchResult:
        self.history.append(message)
        LOGGER.info(message)
        return DispatchResult(True, message)

    def _launch(self, name: str, resolved: str) -> DispatchResult:
        try:
            # URI（ms-settings: 之类）必须走 ShellExecute 语义，普通可执行文件
            # 直接 Popen 即可；两者都不用 shell=True，避免命令拼接。
            if ":" in resolved and not os.path.exists(resolved):
                os.startfile(resolved)  # noqa: S606 - 白名单已限定目标
            else:
                subprocess.Popen([resolved], close_fds=True)
        except (OSError, ValueError) as error:
            LOGGER.exception("启动失败：%s", resolved)
            return DispatchResult(False, f"启动 {name} 失败：{error}")
        return self._record(f"已启动 {name}")

    def _hotkey(self, name: str, combo: str) -> DispatchResult:
        try:
            press_hotkey(combo)
        except (OSError, ValueError) as error:
            LOGGER.exception("热键注入失败：%s", combo)
            return DispatchResult(False, f"热键 {combo} 失败：{error}")
        return self._record(f"已发送 {name} ({combo})")
