# 调参指南

> 所有参数都在 `config.json`，改完**重启程序**生效（没有热重载）。
> 参数总共 100 多个，但你**需要动的通常不超过 6 个**。

---

## 1. 方法论：一次只改一个

手势鼠标是**多判据串联**的系统：进入门 → 姿态判定 → 捏合判定 → 点击确认 → 输入注入。
一次改三个参数，你无法知道是哪个起了作用，也不知道该退回哪个。

规矩：

1. **一次只改一个键**；
2. 改完录一段 20 秒的实机日志（`--safe-mode --diagnostic-seconds 20`）；
3. 变好就留下，变坏就退回，**不要"再试试看"**；
4. 同一个问题改到第三轮还不好 → **换判据，不要在数值上继续磨**
   （项目里滚轮模式就是这么被删掉的）。

`config.json` 里的键名与 `handmouse/config.py` 的 dataclass 字段**一一对应**，
注释都在那边，写得很啰嗦但有用。

---

## 2. 最常需要调的 6 个

| 键 | 默认 | 什么时候调 |
|---|---|---|
| `pointer_stability` | 0.58 | 光标抖 → 调高（0.65~0.75）；觉得迟钝 → 调低 |
| `pointer_response` | 0.82 | 跟手程度。和上面是一对，**先动 stability** |
| `pinch_engage` | 0.42 | 捏不出来 → 调高；一动就误点 → 调低。步进 0.05 |
| `pinch_release` | 0.55 | 松不开（粘着点击）→ 调高；一松手就断开拖拽 → 调低 |
| `cursor_micro_deadband` | 0.0025 | 瞄准小图标时晃 → 调高（0.004~0.006） |
| `pinch_pre_dwell_ms` | 1000 | 停留唤醒磁吸太慢 → 降到 600~800 |

`pinch_engage < pinch_release < pinch_arm` 必须保持递增。程序里没有强制校验，
写反了会出现"永远捏不出来"。

---

## 3. 按症状查表

### 光标

| 症状 | 键 | 方向 |
|---|---|---|
| 停在目标上还在颤 | `pointer_stability` | ↑ |
| 光标慢半拍 | `cursor_interpolation_ms` | ↓（42 → 16~30） |
| 手指不动光标却缓慢漂移 | `cursor_micro_deadband` | ↑ |
| 手要移很大才到屏幕边缘 | `pointer_margin_x` / `pointer_margin_y` | ↓（0.2 → 0.12） |
| 想去到屏幕最边缘却到不了 | `pointer_edge_override` | 确认是 `true` |
| 屏幕上下被切掉一块 | `pointer_margin_top/bottom` | 单独设，覆盖 `pointer_margin_y` |

### 点击

| 症状 | 键 | 方向 |
|---|---|---|
| 捏合毫无反应 | `pinch_engage` | ↑ |
| 手指一动就点了 | `pinch_engage` | ↓ |
| 想点但触发成拖拽 | `drag_hold_ms` | ↑ |
| 点不出来，因为总升级成拖拽 | `pinch_hold_ms` | ↑（100 → 150） |
| 双击打不开文件 | `double_click_window_ms` | ↑（700 → 900） |
| 两次单击粘成双击 | `double_click_max_drift` | ↓ |

### 滚轮 / 翻页

| 症状 | 键 | 方向 |
|---|---|---|
| 双指划一下滚太多 | `swipe_scroll_gain` | ↓（30 → 15） |
| 双指划了不滚 | `swipe_min_delta` | ↓（24 → 16） |
| 滚动手感一顿一顿 | `swipe_hold_ms` | ↓（650 → 500，代价是误触变多） |
| 左右翻页太灵敏 | `swipe_threshold` | ↑（0.12 → 0.18） |
| 完全不要翻页 | `swipe_navigation_enabled` | `false` |

### 磁吸

| 症状 | 键 | 方向 |
|---|---|---|
| 吸不到按钮 | `cursor_snap_radius_px` | ↑（34 → 45） |
| 两个小图标之间弹跳 | `cursor_snap_radius_px` | ↓（34 → 24） |
| 不想吸到窗口外部 | `cursor_snap_inside_only` | `true` |
| 完全不要磁吸 | `cursor_snap_enabled` | `false`，或启动加 `--no-snap` |

### 进入/退出

| 症状 | 键 | 方向 |
|---|---|---|
| 手放松就退出控制态 | `fist_exits_gate` | `false` |
| 握拳退不出去 | `fist_hold_ms` | ↓（900 → 500） |
| 进来太慢（五指要保持太久） | `entry_hold_ms` | ↓（300 → 200） |
| 手背也能顶开闸门 | `entry_palm_nz_min` | ↑（0.3 → 0.45） |
| 不想有进入姿势 | `entry_gate_enabled` | `false`（等于一直可控） |
| 开机对着镜头比五指没反应 | `wake_by_open_palm` / `wake_hold_ms` | 确认 `true` / 800 |

---

## 4. 关于「阈值是给一只手的」

系统的定位是**为一只手校准的鼠标**：作者本人的阈值来自他自己
700+ 帧带标注的采集。换一个人，`pinch_engage` 和 `pointer_stability`
几乎一定需要重新调 —— 这不是缺陷，是刻意的设计取舍（见 README「为什么叫 HandSolo」）。

调参的基本功是**先看清自己的分布**，而不是照抄别人的数字：

```bat
:: ① 采集带标注的个人样本（只开摄像头，绝不移动鼠标）
python -m handmouse.collect
python -m handmouse.collect --only swipe --append     :: 只补采双指
python -m handmouse.collect --seconds 12              :: 每种手势录久一点

:: ② 用样本回放，看某个判据在你自己身上的表现
python scripts\audit_dwell.py
python scripts\audit_swipe_final.py
python scripts\audit_pointer_anchor.py
```

`scripts/audit_*.py` 全部是**离线回放**工具：import 生产代码里的判据函数，
在历史样本上重算命中率与误报率，改判据前后各跑一次就能知道有没有真的变好。
**改判据前先跑审计脚本，不要凭直觉调**——这是这个项目里最值钱的一条经验。

---

## 5. 什么时候该放弃调参

出现下面任一情况，说明问题不在数值上，别继续磨：

- 同一症状改了 3 个不同方向都不改善 → 判据的**观测维度**选错了；
- 两类姿势的取值区间**完全重叠**（例如「拇指张开」与「手指朝下」的 spread
  在 1.1~1.4 重叠）→ 任何单一阈值都分不开，要换维度；
- 只有在**某个特定朝向**下失效 → 二维投影判据的天花板，改用米制 3D 坐标；
- 调好了一个姿势，另一个姿势又坏了 → 阈值在挪动边界，不是在解决问题。
