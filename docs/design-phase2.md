# 设计文档：第二阶段（引擎层移植 + 轨迹手势）

> 这是 2026-10 的一次架构决策记录：为什么整体移植 `KIdiens/Vimoose` 的引擎层、
> 而不是继续在自研管线上打补丁，以及移植前做过的验收与踩到的依赖坑。
> 结论里的"第一阶段 `main.py` + `gesture/`"后来已被删除，保留原文不改。

> 本文件取代第一阶段版本。第一阶段 `main.py` + `gesture/` 包保留未动，确认后再决定是否删除。

## Context（为什么做这一阶段）

第一阶段跑通了「食指光标 + 捏合左键 + 握拳冻结 + 三点校准」，但有三个结构性短板：

1. **窗口失焦就脱不了身**——鼠标接管后光标移出 OpenCV 窗口即失去键盘焦点，ESC/M 按不到，只能靠"握拳 0.6 秒"应急。
2. **捏合只有一种语义**——捏合即 `left_down`、松开即 `left_up`，Windows 拖拽判定阈值 `SM_CXDRAG` 仅 4px，想精确点击很容易变成误拖。
3. **没有悬停辅助，也没有"手势触发动作"的能力**——只有"映射光标/按钮"，没有"画个 W 打开记事本"这类主动动作。

本阶段的决策不是继续打补丁，而是**整体移植 `KIdiens/Vimoose`（MIT）的引擎层作为基座**，再在其上增补状态机、轨迹手势与 LAUNCH。理由：Vimoose 已把「MediaPipe Tasks 异步管线 + One Euro 滤波 + SendInput 注入 + 控件磁吸 + 全局热键」做成了分层清晰、测试覆盖完整的引擎，重写一遍是纯浪费。

被否决的备选：纯网页 demo（hand-gesture-grab、指尖万剑）无法稳定控制系统光标；`Gesture-Controlled-Virtual-Mouse` 是 GPL-3.0；NonMouse 是 Electron。

## 已核实的移植前置结论

移植前先在临时 venv（Python 3.12.9）里跑了验收 gate，**全部通过，所以采用整体移植**：

| 验收项 | 结果 |
|---|---|
| 原仓库 pytest | **67 passed**（11 个测试文件） |
| 引擎层能否脱离 PySide6 独立运行 | 能。自写 headless 冒烟脚本加载 `.task` 模型成功、`NullMouseBackend` 零注入，输出 `SMOKE PASS` |
| 模型是否可随仓库分发 | 可。`hand_landmarker.task` 7,819,105 字节 |

三条实测结论，其中第一条与预期冲突，需要明确记录：

1. **`mediapipe` 硬依赖 `opencv-contrib-python`，"core 不含 contrib"无法达成。**实测 `pip install mediapipe==0.10.21` 会自动装上 `opencv-contrib-python 4.11.0.86`——贡献版是 wheel 元数据里的硬依赖，不是可选。只要保留 MediaPipe，就不可能不带 contrib。**对策：不再单独 pin `opencv-python`**（否则两个发行版同时提供 `cv2` 互相覆盖），桌面预览直接用 MediaPipe 带来的那一份 OpenCV。等哪天不用 MediaPipe 了，再回来收这块钱。
2. **`pywin32` 不需要。**引擎层鼠标/键盘注入全部是纯 `ctypes` + `user32.SendInput`（`MOUSEINPUT`/`KEYBDINPUT`/`INPUT` 三个结构体自己声明），代码里没有任何 `win32`/`comtypes` 引用。少一个依赖。
3. **UIA 确实没有进热路径。**`uiautomation` 在 `cursor_snap.py` 里是**函数内惰性导入**；`CursorSnapService.transform()`（每帧都会调）只做「加锁 + 取缓存候选 + 整数换算」，是 O(1) 纯算术；真正会跨进程 COM 的 `find_target` 只跑在独立的守护线程 `cursor-snap-uia` 里，每 `0.06s` 一轮（≈16Hz）。UIA 变慢只会让磁吸命中变陈旧，不会让光标跟手拖影。

## MVP 边界（已拍定）

- **不接 PySide6**。配置走 `config.json` + CLI flags，交互走 OpenCV 调试窗口 + 全局热键。以后真要托盘/设置页再加 `ui` extra，不需要返工。
- **不接 `opencv-contrib`**（见上：MediaPipe 会自动带，不额外引）。
- **`comtypes`/`uiautomation` 是默认关闭的 `uia` extra**，二选一，不叠。
- **多屏只预留接口**：默认 `multi_monitor=false` 按单屏跑，坐标抽象（`ScreenMap`）已经就位，需要时用 `MOUSEEVENTF_VIRTUALDESK` 开实验开关。

## 依赖分层

`pyproject.toml` 里 `dependencies` 刻意留空——引擎层依赖全部下挂在 `core` extra：

| extra | 内容 | 用途 |
|---|---|---|
| `core` | `mediapipe`、`numpy`（自带 `opencv-contrib-python`） | 摄像头采集、手部关键点、光标注入。**可用安装的最小集** |
| `uia` | `uiautomation` | 控件磁吸/找按钮；与 `comtypes` 二选一 |
| `ui` | `PySide6` | 托盘/设置面板，MVP 未实现，先占位 |
| `metrics` | `psutil` | 调试窗口的 CPU/内存读数，缺失时自动跳过采样 |
| `all` | 以上全部 | 完整安装 |

```bash
pip install -e ".[core]"   # 可用安装
pip install -e ".[all]"    # 完整
python -m handmouse --safe-mode
```

## 架构

### 线程模型（三线程 + 两个槽）

```
camera-capture   → LatestSlot[FramePacket]    容量1，覆盖即丢，杜绝延迟积压
hand-landmarker  → LatestSlot[DetectionPacket]
gesture-control  → 消费最新检测，驱动状态机并注入输入
                    ↳ 独立守护线程 cursor-snap-uia（≈16Hz）只喂磁吸缓存
```

`LatestSlot` 是关键设计：慢消费者永远拿到最新帧而不是排队帧，所以推理再慢也不会让光标延迟越来越大。

### 坐标：单一 `ScreenMap`

`screen_map.py` 是**唯一**的坐标真相来源，`mouse.py` 不再读 `GetSystemMetrics`：

- `enable_dpi_awareness()` 逐级降级：`SetProcessDpiAwarenessContext(PER_MONITOR_AWARE_V2)` → `shcore.SetProcessDpiAwareness(2)` → `SetProcessDPIAware()`。**必须在任何坐标换算之前调用**，否则高 DPI/多屏一律偏移。
- `to_pixels()` / `to_normalized()` / `absolute_coordinates()`：后者按 `SM_XVIRTUALSCREEN`/`SM_YVIRTUALSCREEN`/`SM_CXVIRTUALSCREEN`/`SM_CYVIRTUALSCREEN` 归一化到 0–65535。
- `use_virtual_desk` = `multi_monitor or virtual_desk_experimental`，为真时注入补 `MOUSEEVENTF_VIRTUALDESK`。

关于"别在 CLICK/DRAG 里写死主屏坐标"：CLICK/DRAG 本身**只发键码、不含坐标**，所以真正的落点是两处——注入换算（已交给 `ScreenMap`）和磁吸的像素原点（`cursor_snap.py` 的 `origin_x/origin_y` 也已改为从 `ScreenMap` 传入）。两处共用同一个 `ScreenMap` 实例。

### 输入注入：`SendInput` + `NullMouseBackend`

`mouse.py` 全部走 `user32.SendInput`，比 `mouse_event`/`SetCursorPos` 可靠（与真实输入同队列、不覆盖用户按键状态），也比 PyAutoGUI 少一层依赖和缓动延迟。

`NullMouseBackend` 记录收到的指令但不调用任何 Win32 API——`--safe-mode` 下能完整跑通状态机、确认动作序列，同时保证系统光标不被接管。

同文件底部追加了键盘注入区（`parse_hotkey` / `press_hotkey`），供 LAUNCH 的 hotkey 绑定使用，仍是纯 `ctypes`。

### 全局热键

`GetAsyncKeyState` 每 30ms 轮询 `Ctrl+Alt` + `G/Esc/Q`，线程内做去抖（`_pressed` 集合）。**不用 `RegisterHotKey`**：注册会被 QQ/输入法占用而失败，且消息在 `waitKey` 的 `PeekMessage` 里会被取走。轮询回调只置标志位，主循环 `_apply_pending_hotkeys` 统一应用——保证所有输入注入都在同一线程发起。

### 轨迹手势：DTW + 独立状态机

用户画 W/C/E/X/圈/横线 → 出候选 → **捏合才执行**。

- `dtw.py`：弧长重采样（固定 32 点，消速度差）→ 质心 + 最大跨度等比归一化（消位置和大小差）→ 归一化 DTW。**刻意不做旋转归一化**：横线和竖线必须能区分。
- `motion_gestures.py`：`IDLE → ARMED → DRAWING → CANDIDATE → 执行`。进入 ARMED 需要「食指+中指伸出、无名指小指收拢、拇指不贴食指」**且指尖原地悬停 250ms**——只靠姿势会与正常光标控制（中指自然半伸）误撞，"没在移动"这个条件才是分界线。收笔由姿势消失触发。候选出来后**不会自动执行**，必须重新捏合，避免画错一笔就误触发。
- 与鼠标状态机的隔离：`DRAWING` 期间的观测**不交给 `GestureEngine`**（否则绘制中的捏合会被当成左键点击），改用 `engine.hold()` 只释放可能按住的左键并续上 `last_hand_ns`（防误报 LOST）。退出绘制时 `filter.reset()` + `interpolator.clear()`，避免恢复控制的瞬间光标跳一下。

实测匹配质量（模板自匹配 + 抖动）：干净轨迹 6 个形状全部 top-1 且距离 ≈0；加 2% 抖动后最优 0.02–0.05，第二名 0.21–0.50；缩放 0.35×~2.2× 与任意平移下自匹配距离仍为 0。竖线（非内置形状）最优 0.2652 > 阈值 0.18，**会被正确拒绝**。

### LAUNCH 白名单

`launch.py` 的安全边界是硬约束：目标只能来自 `config.launch_whitelist`，识别错一次的代价不能是"打开任意 EXE"。`dry_run`（safe-mode）只记录不执行。默认绑定：

| 手势 | 动作 | 目标 |
|---|---|---|
| W | launch | notepad |
| C | launch | calc |
| E | launch | explorer |
| X | hotkey | ctrl+w |
| CIRCLE | hotkey | win+d |
| LINE | hotkey | alt+tab |

白名单别名：`notepad` / `calc` / `explorer` / `settings`。改习惯直接编辑 `config.json` 的 `motion_bindings`。

## 文件清单

**新增（本项目原创）**

| 文件 | 说明 |
|---|---|
| `handmouse/screen_map.py` | DPI 感知 + 单屏/虚拟桌面统一换算，全项目唯一坐标源 |
| `handmouse/dtw.py` | 重采样 / 归一化 / DTW / 模板与匹配 |
| `handmouse/motion_gestures.py` | 轨迹手势状态机 |
| `handmouse/launch.py` | 白名单校验 + LAUNCH/热键派发 |
| `handmouse/debug_view.py` | OpenCV 调试窗口（骨架、轨迹、7 行 HUD） |
| `handmouse/app.py` / `__main__.py` | CLI 入口与主循环 |
| `config.json` | 由 `ConfigStore.save()` 生成，与 dataclass 默认值天然一致 |
| `pyproject.toml` / `requirements.txt` | extras 分层 |
| `tests/test_dtw.py` / `test_motion_gestures.py` / `test_launch.py` | 新增测试 |

**移植并改名（`gesture_mouse` → `handmouse`，共 12 个模块）**
`config.py`、`contracts.py`、`filters.py`、`latest_slot.py`、`mouse.py`、`cursor_snap.py`、`hotkeys.py`、`vision.py`、`partial_tracking.py`、`gestures.py`、`calibration.py`，外加 `models/hand_landmarker.task` 与 `LICENSE.vimoose`（MIT 归属）。

**移植后定点修改**
`mouse.py`（ScreenMap 接入 + 键盘注入 + 结构体重命名为 `Win32MouseBackend` 常量）、`cursor_snap.py`（像素原点与 ScreenMap 对齐）、`config.py`（整体重写：配置路径改 `PROJECT_ROOT/config.json`、去掉版本迁移）、`contracts.py`（新增 `MOTION_*` 手势、`ActionType`/`ActionCommand`、快照字段）、`gestures.py`（新增 `hold()`）、`pipeline.py`（去 PySide6、psutil 变可选、接入轨迹手势）。

**刻意未移植**
`ui.py`（1395 行 PySide6）、`app.py`（Qt 入口）、`startup.py`（改注册表自启）、`logging_setup.py`（写 appdata）、`tests/test_config.py`（依赖已删除的 `version` 与版本迁移）。

## `config.json` 关键参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `multi_monitor` / `virtual_desk_experimental` | `false` / `false` | 多屏坐标映射 / 单屏也走 VIRTUALDESK 的实验开关 |
| `cursor_snap_enabled` / `cursor_snap_radius_px` | — | 控件磁吸开关与半径（需要 `uia` extra） |
| `motion_gestures_enabled` | `true` | 轨迹手势总开关 |
| `motion_arm_hold_ms` / `motion_arm_max_drift` | `250` / `0.05` | 双指悬停进入绘制的时间与"原地"容差 |
| `motion_min_duration_ms` / `motion_min_path_px` | `250` / `60.0` | 太短/太小的轨迹提前丢弃，不进 DTW |
| `motion_template_points` / `motion_max_candidates` | `32` / `3` | 重采样点数与候选个数 |
| `motion_accept_distance` | `0.18` | 最优候选的 DTW 接受阈值 |
| `motion_confirm_timeout_ms` / `motion_confirm_pinch` | `2500` / `0.42` | 候选等待确认的时限与捏合确认阈值 |
| `motion_hand_lost_ms` | `200` | 绘制中手部丢失多久作废本次轨迹 |
| `motion_bindings` / `launch_whitelist` | 见上表 | 形状→动作绑定 / LAUNCH 允许的目标 |
| `launch_enabled` | `true` | LAUNCH 总开关 |

## 验证

**自动化**：`pytest -q` → **119 passed**（移植来的 59 例 + 新增 60 例）。移植来的用例覆盖原引擎的滤波、手势状态机、磁吸、局部续跟踪、流水线；新增用例覆盖 DTW 的数学性质与 6 个模板回归、轨迹手势全链路（含"绘制期间不产生 MOVE"）、LAUNCH 白名单与 safe-mode 零副作用。

**真实摄像头冒烟**（safe-mode，`--diagnostic-seconds 4`）：

```
capture_fps=30.0  inference_fps=8.2  latency_ms=15.0  p50=15.0  p95=16.0
hand_detection_rate=0.0  dropped_frames=0  cpu_percent=2.0  memory_mb=205.8  error=none
后端=NullMouseBackend  DPI=PerMonitorV2  显示器=1  注入坐标系=(0,0) 1920x1080  virtual_desk=False
```

相机、模型加载、线程启停、坐标抽象、零注入全部正常（`hand_detection_rate=0.0` 是镜头前没放手）。

**手动验证（按顺序，每步可单独停）**

1. `python -m handmouse --safe-mode` → 确认光标**完全不动**，骨架/轨迹/HUD 正常，画 W 后能出候选。
2. `python -m handmouse --diagnostic-seconds 20` → 真实放开手，看 `hand_detection_rate` 与延迟。
3. `python -m handmouse` → 默认**不接管**，按 `M` 或 `Ctrl+Alt+G` 才开始接管鼠标。
4. **急停演练**：接管后按 `Ctrl+Alt+Esc`，断言左键立刻松开、控制暂停；窗口失焦状态下重按一次仍生效（这是本项关键判据）。
5. 轨迹手势：双指悬停 → 画 W → 收指 → 看到候选 → 捏合 → 记事本打开。
6. 按键：`M` 暂停/接管 · `R` 重置滤波器与轨迹状态 · `ESC`/`Q` 退出 · `Ctrl+Alt+Q` 退出。

## 已知取舍与后续

- **contrib 剔不掉**（见上），打包体积由 MediaPipe 决定。
- **多屏未实机验证**：接口与归一化已按虚拟桌面写好，但只在一台单屏机上跑过；`--multi-monitor` / `--virtual-desk` 需要在多屏机器上实测后再转正。
- **磁吸默认关**：需要 `uia` extra 且 UIA 跨进程调用在某些窗口上会慢，先按可选功能处理。
- **`ui`/托盘/设置面板未做**：配置只能改文件，符合 MVP 边界。
- **第一阶段 `main.py` 与 `gesture/` 包仍在仓库里**，已被 `handmouse` 取代，两者不可混用；是否删除待确认。
