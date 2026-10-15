# HandMouse · 视觉手势鼠标

用摄像头控制 Windows 光标：MediaPipe 手部关键点 → 手势状态机 → `user32.SendInput` 注入。
支持「食指光标 / 捏合点击 / 握拳冻结 / 张开手掌滚动」。

- 平台：Windows（非 Windows 自动降级为 safe-mode，只跑状态机不注入）
- Python：>= 3.10
- 版本：0.7.0
- 引擎层移植自 [`KIdiens/Vimoose`](https://github.com/KIdiens/Vimoose)（MIT，见 `handmouse/LICENSE.vimoose`）

## 当前阶段

MVP 已跑通：光标接管、点击、拖拽、冻结、滚动、翻页、控件磁吸（可选）、紧急停止、
多屏坐标抽象（接口就位，未在真机多屏验证）。

无 GUI / 无托盘：配置走 `config.json` + CLI flags，交互走 OpenCV 调试窗口与全局热键。
`ui` extra（PySide6）只是占位，MVP 未实现。

## 安装

依赖分层刻意做成 extras，`dependencies` 留空，避免“装包即拖进一堆重依赖”。

```bash
# 可用安装（摄像头 + 关键点 + 光标注入，够跑 safe-mode 与正式模式）
pip install -e ".[core]"

# 控件磁吸 / 找按钮（可选，与 comtypes 二选一）
pip install -e ".[uia]"

# 调试窗口的 CPU/内存读数（可选，缺失时自动跳过采样）
pip install -e ".[metrics]"

# 全部
pip install -e ".[all]"
```

> `mediapipe` 的 wheel 元数据硬依赖 `opencv-contrib-python`，pip 会自动装上它。
> **不要**再单独 pin `opencv-python`，否则两个发行版同时提供 `cv2` 会互相覆盖。
> 桌面预览用的就是 mediapipe 带来的那一份 OpenCV。

手部模型 `handmouse/models/hand_landmarker.task` 随包分发，运行时不需要联网下载。

## 运行

```bash
python -m handmouse                      # 启动，默认【不接管】，按 M 或 Ctrl+Alt+G 才接管鼠标
python -m handmouse --safe-mode          # 不注入任何系统输入，只跑状态机 + 调试窗口
python -m handmouse --diagnostic-seconds 20   # 跑 20 秒后打印指标退出
```

首次运行会在 EXE/仓库同级目录生成 `config.json`（内容与 dataclass 默认值一致）。

> **入口唯一**：第一阶段遗留的 `main.py` 与 `gesture/` 包已删除，仓库里只有 `handmouse`
> 这一个入口，一律用 `python -m handmouse [--safe-mode]`。若需要回看旧代码，见 git 标签
> `pre-cleanup-main-gesture`（`git show pre-cleanup-main-gesture:main.py`）。


**建议第一次按这个顺序验证**（每步可单独中断）：

1. `python -m handmouse --safe-mode` → 确认系统光标**完全不动**，骨架/HUD 正常。
2. `python -m handmouse --diagnostic-seconds 20` → 手放在镜头前，看 `hand_detection_rate` 与延迟。
3. `python -m handmouse` → 默认不接管，按 `M` 接管。
4. **急停演练**：接管后按 `Ctrl+Alt+Esc`，确认左键立刻松开且控制暂停；窗口失焦状态下再按一次仍生效。

## 按键与热键

| 按键 | 作用 |
|---|---|
| `M` | 暂停 / 恢复（接管） |
| `R` | 重置滤波器 |
| `ESC` / `Q` | 退出 |
| `Ctrl+Alt+G` | 接管（窗口失焦也生效） |
| `Ctrl+Alt+Esc` | **急停**：立刻松开左键并暂停控制 |
| `Ctrl+Alt+Q` | 退出 |

全局热键走 `GetAsyncKeyState` 每 30ms 轮询（不用 `RegisterHotKey`：会被 QQ/输入法抢占，
且消息会被 `waitKey` 的 `PeekMessage` 取走）。轮询回调只置标志位，实际注入统一在主循环发起。

## 手势一览

| 手势 | 效果 |
|---|---|
| 食指伸出 | 光标移动（One Euro 滤波 + 插值，`pointer_*` 参数） |
| 捏合（拇指-食指靠近） | `PINCH_ARMED` 锁定落点 → 松开触发左键点击（单击延迟 `double_click_window_ms` 提交） |
| 捏合两下（窗口内、同位置） | 左键双击，打开/启动（第一次捏合绝不提前单击） |
| 捏合后保持 | 左键按住拖拽（`drag_hold_ms`） |
| 握拳保持 | 冻结：松开左键并切换暂停（应急用，`fist_hold_ms`，再握一次恢复） |
| 张开手掌 | 滚轮模式（上下移动手掌滚动） |
| 手掌左右甩动 | 上一页 / 下一页（`page_swipe_threshold`） |
| 手掌旋转 | 滚动（`rotation_gestures_enabled`） |
| 手部局部丢失 | 食指/中指局部续跟踪（`partial_tracking_frames`） |

## CLI 参数

| 参数 | 说明 |
|---|---|
| `--config PATH` | 配置文件路径，默认仓库根 / EXE 同级 `config.json` |
| `--camera N` | 摄像头序号，覆盖 config |
| `--safe-mode` | 不注入任何系统输入 |
| `--auto-control` | 启动即接管（默认先暂停） |
| `--no-snap` | 关闭控件磁吸 |
| `--multi-monitor` | 按虚拟桌面映射坐标 |
| `--virtual-desk` | 实验：单屏也走 `MOUSEEVENTF_VIRTUALDESK` |
| `--margin X Y` | 活动区左右/上下内缩比例 |
| `--diagnostic-seconds S` | 跑 S 秒后打印指标并退出 |

## 关键配置项

配置项很多，完整列表见 `config.json`；下面是最常需要调的。

| 参数 | 默认 | 含义 |
|---|---|---|
| `camera_index` / `camera_width` / `camera_height` / `camera_fps` | 0 / 640 / 480 / 30 | 采集参数 |
| `pointer_stability` / `pointer_response` | 0.58 / 0.82 | 光标抖动抑制与跟手程度 |
| `cursor_interpolation_ms` | 42 | 目标点插值时长，越大越平滑越延迟 |
| `pointer_margin_x` / `pointer_margin_y` | 0.2 / 0.16 | 画面边缘不映射到屏幕边缘 |
| `pinch_engage` / `pinch_release` / `pinch_arm` | 0.42 / 0.55 / 0.72 | 捏合触发 / 松开 / 预锁阈值（手部宽度归一） |
| `drag_hold_ms` / `fist_hold_ms` | 5000 / 500 | 拖拽与握拳冻结的判定时长 |
| `double_click_window_ms` / `double_click_max_drift` | 350 / 0.05 | 双击窗口（ms，单击延迟同值）与两次捏合落点容差（归一化画面坐标） |
| `cursor_snap_enabled` / `cursor_snap_radius_px` | true / 30 | 控件磁吸开关与半径（需 `uia` extra） |
| `multi_monitor` / `virtual_desk_experimental` | false / false | 多屏映射 / 单屏实验开关 |

## 架构

```
camera-capture   → LatestSlot[FramePacket]     容量 1，覆盖即丢，杜绝延迟积压
hand-landmarker  → LatestSlot[DetectionPacket]
gesture-control  → 消费最新检测，驱动状态机并注入输入
                    ↳ 独立守护线程 cursor-snap-uia（≈16Hz）只喂磁吸缓存
```

- **`LatestSlot`**：慢消费者永远拿到最新帧而不是排队帧，推理再慢也不会让光标延迟累积。
- **`ScreenMap` 是唯一坐标真相源**：`mouse.py` 不再读 `GetSystemMetrics`。启动时 `enable_dpi_awareness()`
  逐级降级（`PER_MONITOR_AWARE_V2` → `shcore` → `SetProcessDPIAware`），必须在任何坐标换算之前调用。
- **输入注入**：全部走 `user32.SendInput`（与真实输入同队列、不覆盖用户按键状态），比 `mouse_event` /
  `SetCursorPos` 可靠，也比 PyAutoGUI 少一层依赖和缓动延迟。safe-mode 用 `NullMouseBackend` 记录指令但零注入。
- **UIA 不在热路径**：`CursorSnapService.transform()` 每帧都调，但只做加锁 + 取缓存候选 + 整数换算（O(1) 纯算术）；
  真正跨进程 COM 的 `find_target` 只在守护线程里以 ≈16Hz 跑。UIA 变慢只会让磁吸命中变陈旧，不会造成光标拖影。

## 目录结构

```
handmouse/
  app.py  __main__.py         CLI 入口与主循环
  pipeline.py                 三线程管线 + 快照
  vision.py  partial_tracking.py   摄像头采集 / MediaPipe 推理 / 局部续跟踪
  contracts.py                统一数据契约（手势、指令、快照）
  gestures.py                 鼠标手势状态机
  mouse.py                    SendInput 注入 + NullMouseBackend
  screen_map.py               DPI 感知 + 单屏/虚拟桌面坐标换算
  cursor_snap.py              控件磁吸（uia 惰性导入）
  filters.py  latest_slot.py  滤波 / 最新帧槽
  hotkeys.py                  全局热键轮询
  calibration.py  debug_view.py    校准 / OpenCV 调试窗口
  models/hand_landmarker.task 随包分发的手部模型
  LICENSE.vimoose             上游 MIT 归属
tests/                        pytest 用例
config.json                   ConfigStore 生成的默认配置
handmouse.spec                PyInstaller onefile 打包配置
```

## 测试

```bash
pytest -q
```

覆盖滤波、鼠标手势状态机、点击/拖拽语义、磁吸、局部续跟踪、流水线与 safe-mode 零副作用。

## 打包

```bash
pyinstaller handmouse.spec      # 产物 dist/handmouse.exe（单文件）
```

> 此 spec **尚未实机打包验证**。首次打包后请在无 Python 的干净机器上冒烟四条路径：
> 启动、摄像头采集、调试窗口、光标注入。spec 内的关键约束（模型 datas 目标必须是
> `handmouse/models`、不打包 `config.json`、不排除 cv2/mediapipe）都写在了注释里，改动前先读注释。

## 已知限制

- **`opencv-contrib-python` 剔不掉**：是 MediaPipe 的硬依赖，打包体积由 MediaPipe 决定。
- **多屏未实机验证**：归一化与 `MOUSEEVENTF_VIRTUALDESK` 已按虚拟桌面写好，但只在单屏机上跑过。
- **磁吸默认需要 `uia` extra**：跨进程 UIA 调用在某些窗口上会慢，按可选功能处理。
- **无托盘/设置面板**：只能改 `config.json` 与 CLI flags。

## 许可

本项目原创部分与移植部分均以 MIT 分发。移植自 `KIdiens/Vimoose` 的模块保留上游许可，
见 `handmouse/LICENSE.vimoose`。
