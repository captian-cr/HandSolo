"""配置：仓库根目录的 ``config.json`` 为唯一真源，CLI flags 可临时覆盖。

设计取舍：MVP 阶段不做设置面板，所以没有 GUI 配置界面。所有现场可调参数
集中在 ``AppConfig``，改完 ``config.json`` 重启即可生效；命令行只提供少量
覆盖项（``--camera`` / ``--margin`` / ``--safe-mode`` 等）。

读取策略：文件缺失或损坏时退回默认值并且**不阻塞启动**，只在控制台提示。
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path


def _base_dir() -> Path:
    """配置文件所在目录。

    打包成单 exe 后 ``__file__`` 指向解压出来的一次性临时目录，写进去的配置
    下次启动就没了，所以冻结模式下改用 exe 自身所在目录。
    """

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _base_dir()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"


@dataclass(slots=True)
class AppConfig:
    """应用可持久化配置。"""

    # ---- 摄像头 ----
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    #: 摄像头拒绝配置的分辨率时，是否按内置档位表（4:3 优先）继续尝试。
    #: 默认开——别人机器上的摄像头五花八门，宁可自动降级也不要打不开。
    #: 关掉它等于"要么按我配的来，要么失败"，只在排查问题时有用。
    camera_allow_fallback: bool = True
    #: 采集后端：``"dshow"``（Windows 老后端，兼容性最好）/ ``"msmf"``
    #: （Media Foundation，较新）/ ``"any"``（让 OpenCV 自己挑）。
    #:
    #: 🔴 **这是显式开关，不是自动探测**：同一台机器两次启动走的路径必须一样，
    #: 否则出问题没法复现。默认 ``dshow``，因为 MSMF 在部分摄像头/驱动上是坏的。
    #: 如果你的摄像头在 dshow 下帧率只有 15~24fps 而 MSMF 能到 30fps，
    #: 就把它改成 ``"msmf"``——实测过两台摄像头都是这个结论。
    #: 非 Windows 平台忽略此值（一律 ``any``）。
    camera_backend: str = "dshow"
    mirror: bool = True
    handedness: str = "Any"
    recognition_zoom: float = 1.15
    #: 推理输入限宽（像素）。0 = 用原尺寸。降下来能显著提高推理帧率，
    #: 归一化坐标不受影响，代价是手指定位精度略降。
    inference_width: int = 0

    # ---- 识别阈值 ----
    min_detection_confidence: float = 0.40
    min_presence_confidence: float = 0.35
    min_tracking_confidence: float = 0.40

    # ---- 光标滤波与映射 ----
    pointer_stability: float = 0.58
    pointer_response: float = 0.82
    #: 光标锚在哪个关键点上：``"index_tip"``（食指尖，landmark 8）或
    #: ``"palm"``（掌心，0/5/9/13/17 的均值 + 缓慢补偿偏移）。
    #:
    #: 2026-10-23 离线实测（``scripts/audit_pointer_anchor.py``，701 帧标注样本）：
    #: 掌心锚点的**形状抖动只有食指尖的 17%~36%**（捏合段最低 17%，
    #: 段内被带跑的距离也只有 33%）。这正是两个老问题的根因：
    #: "磁吸释放瞬间光标跳回随手位置" 与 "捏合时 1.42→0.90 这段光标跟着食指走"。
    #:
    #: 🔴 为什么不是"直接换成掌心"就完事：掌心在画面里的位置比食指尖**偏低**，
    #: 直接映射会让整个光标区域下移、屏幕顶部够不到。所以 ``"palm"`` 模式
    #: 会维护一个 ``(食指尖 - 掌心)`` 的慢速 EMA 并把它加回去
    #: （``pointer_anchor_offset_tau_s``）：拿掌心的**稳定**，同时保住食指尖的
    #: **覆盖范围**。快抖被滤掉、平均位置不变。
    #:
    #: 默认用 ``"palm"``（2026-10-23 起）—— 上面那两个老问题只能在锚点上根治。
    #: 万一实机上觉得"手要移得更多/更少"，改回 ``"index_tip"`` 即可完全恢复
    #: 旧行为（一个键，不改代码）。
    pointer_anchor: str = "palm"
    #: ``pointer_anchor="palm"`` 时，``(食指尖 - 掌心)`` 补偿量的 EMA 时间常数
    #: （秒）。太小会把食指尖的抖动带回来，太大则换姿势（伸指/收指）后光标
    #: 会有一段时间偏着。1.0s 是"能跟上换姿势"与"滤掉抖动"的折中。
    pointer_anchor_offset_tau_s: float = 1.0
    cursor_interpolation_ms: int = 42
    #: 瞄准死区（归一化画面宽度）：光标离"上次真正落点"不到这个距离就**不发
    #: MOVE**，让它钉在原地。用途是 2026-10-22 用户提的「手指动一点点的时候
    #: 光标在抖，我瞄不准小目标」——抖动幅度小于死区就不会传到屏幕上，
    #: 手指一动大就立刻跟上（不做延时、不做插值滞回，所以不影响手感）。
    #: 0.0025 在 1920 宽屏上约 4.8px，够吸掉单目关键点的高频抖动，
    #: 又小到不会让人觉得"粘住"。设 0 关闭。左右方向键 ``; '`` 可现场调。
    cursor_micro_deadband: float = 0.0025
    pointer_margin_x: float = 0.20
    pointer_margin_y: float = 0.16
    #: 四边独立内缩（覆盖 ``pointer_margin_x/y``）。``None`` 表示用对称值。
    #: 2026-10-21 点击标定实测（用户 18 次捏合）：
    #:   他的**舒适**指尖范围只有 x∈[0.208, 0.676]、y∈[0.472, 0.818]，
    #:   而对称 margin 0.15/0.22 要求 x∈[0.15,0.85]、y∈[0.22,0.78]，
    #:   于是**屏幕上方 45%、右侧 25% 根本够不到** ——
    #:   这正是他反馈"要点上面的窗就要把手举很高"和"右下角不准"的同一个根因。
    #: 对称内缩无法表达"上面够不到、下面有余量"这种范围，必须四边分开。
    pointer_margin_top: float | None = None
    pointer_margin_bottom: float | None = None
    pointer_margin_left: float | None = None
    pointer_margin_right: float | None = None
    pointer_edge_override: bool = True

    # ---- 捏合 / 手势 ----
    pinch_engage: float = 0.42
    pinch_release: float = 0.55
    pinch_arm: float = 0.72
    #: 「手指朝下」的判定 margin（按掌宽归一）：食指尖低于食指掌指关节
    #: 超过本值就算朝下。朝下且食指弯曲时不算捏合——那是勾手指，不是捏。
    pinch_down_margin: float = 0.08
    #: 是否启用「朝下 + 食指弯 → 不算捏合」这条抑制。
    #:
    #: 2026-10-21 用户实测反馈：他捏合时 HUD 显示捏合比 0.32（已低于 engage
    #: 0.42），状态却仍是「正在控制光标」—— 说明这条抑制把真捏合也挡掉了。
    #: 根因还是老二维判据：他的侧手投影下 `_finger_extensions` 常把弯着的
    #: 食指判成伸直、又把朝前的食指判成朝下。默认保持 True 不改变他人行为，
    #: 该用户配置里关掉。
    pinch_down_suppress_enabled: bool = True
    #: 单击是否**立即落地**（默认 False，保持旧的"挂起等双击窗口"语义）。
    #:
    #: True 时捏合释放立刻发 LEFT_CLICK，不再等 `double_click_window_ms`；
    #: 双击改由**操作系统**按时间+距离合并（真实鼠标就是这么工作的）。
    #:
    #: 2026-10-21 实测依据：该用户 294 段捏合只产生 28 次点击，其中单击仅 4 次、
    #: 双击 24 次 —— 他等不到半秒的延迟反馈就再捏一次，把单点做成了双击，
    #: 再等不到就按得更久，最后被升级成拖拽（46 次）。
    click_immediate: bool = False
    #: 握拳是否退出控制态（进入门）。
    #:
    #: 2026-10-21 用户实测：日常使用中手一放松就半握，`_is_fist` 判成握拳
    #: → 进入门被关掉 → HUD 报「未进入控制态」、光标定住，他以为卡死了
    #: （31002 帧记录里这种状态 93 次、合计 76 秒）。
    #: 设为 False 后握拳**只冻结本次动作、不退出控制态**，
    #: 退出只能靠 M / Ctrl+Alt+Esc —— 反而更可控。
    fist_exits_gate: bool = True

    # ---- 双指滑动导航（2026-10-21 用户要求）----
    #: 食指+中指伸出、无名指与小指握起 = 滑动姿势。
    #: 上下划滚动内容，左右划翻页。与「只伸食指移动」「五指张开挥动翻页」互不冲突。
    swipe_navigation_enabled: bool = True
    #: 触发一次滑动所需的掌心位移（画面归一化）。掌心比指尖稳，取值偏小即可。
    swipe_threshold: float = 0.12
    #: 起手后多久内的位移才计入（超过就重设锚点，避免慢慢挪也触发）。
    swipe_window_ms: int = 800
    #: 一次上下滑动滚多少格（每格 120 = Windows 的标准一格）。
    #: ⚠️ 2026-10-22 起**纵向滚动已改为连续给量**，此值不再用于上下滚，
    #: 仅保留兼容；上下滚动强度请看 ``swipe_scroll_gain``。
    swipe_wheel_notches: int = 3
    #: 触发后的冷却时间，防止一次挥手连发（主要用于左右翻页去重）。
    swipe_latch_ms: int = 350

    # ---- 双指"滑轮式"连续滚动（2026-10-22 用户要求）----
    #: **滚动强度**：掌心每走 1.0（归一化行程）折算多少个标准滚轮格。
    #: 逐帧结算，所以划得越快/越长 → 每帧给得越多、帧数也越多 → 滚得越多；
    #: 手慢下来 → 立刻变慢（速率→速度成正比，这就是"像鼠标一样的灵敏度"）。
    #: 2026-10-22 用户反馈"要像鼠标一样灵敏" → 18 → 30；还不够就再往大调。
    swipe_scroll_gain: float = 30.0
    #: 攒够多少个滚轮单位才真的发一次（1 格 = 120）。太小会以高频刷小事件，
    #: 太大就会有"一格一格跳"的顿挫感。2026-10-22: 40 → 24，配合 gain=30
    #: 让快划逐帧出量、慢划也顺滑，减少阶梯感。
    swipe_min_delta: int = 24
    #: 逐帧位移的噪声门（归一化）：低于它的帧位移当作手抖，不累计。
    swipe_step_deadzone: float = 0.002
    #: 滚动中双指判据偶发失败的**滞回持有**时长（毫秒）。
    #:
    #: 运动模糊、划动末端手指回弯、掌面轻微翻转都会让判据抖一两帧；
    #: 旧行为是失败的那一帧立刻掉回"只伸食指=移动"，把光标瞬移到食指
    #: 位置——2026-10-22 用户实机反馈"连续滑动时鼠标跟着手指乱飞"就是它。
    #: 现在抖动期间按住双指状态不动（不滚也不移），判据回来无缝继续。
    #: 时长直接决定"双指到底能不能用"：他 701 帧**带 3D** 的新采集数据上
    #: 用 3D 判据做的时序仿真（`scripts/audit_2d_vs_3d_real.py`）——
    #: 450ms 76.0%、**650ms 91.0%**、900ms 93.0%，而"只伸食指/停住/捏合/
    #: 五指张开/握拳"的误触在所有档位都是 0~3.3%（不随持有窗变差）。
    #: 取 650：他实机报的"划着划着就跳回控制光标"就是掉出持有窗导致的。
    #: 代价：故意换成别的姿势后，光标最多再冻 650ms 才恢复——仍小于反向
    #: 滚动要等的 `swipe_window_ms`(800ms)。
    swipe_hold_ms: int = 650
    #: 「停住 → 捏合」的锚定时长（毫秒）：停住光标后要捏合时，先把光标
    #: 钉住这么久，捏合动作的位移就不会把落点带偏。
    pinch_anchor_hold_ms: int = 1000
    #: 捏合趋向预测阈值：捏合比降到本值以下就认为"要捏合了"，提前钉住光标。
    #:
    #: 2026-10-21 实测标定（`logs/pinch-anchor-20261021-150447.csv`，521 帧）：
    #:   * 向下移动全程捏合比中位 1.61、最小 0.648，**整段只有 1 帧低于 0.70**，
    #:     且那一帧 tip_y=0.944（映射后已超界、光标本就钳在屏幕底），冻结无感；
    #:   * 5 次捏合 **5/5** 都在跌破 pinch_engage(0.42) 之前穿过 0.70。
    #: 所以 0.70 既能提前钉住、又几乎不误伤向下移动。
    #: 注意：捏合比在 30fps 下常一帧从 0.66 直落 0.35，**绝对阈值必须留在这个
    #: 过渡带里**，否则来不及触发（用 0.60 时 5 次里只能提前命中 1 次）。
    #: 2026-10-22 由 0.70 提到 **0.85**（他要求"设大一些，因为我每次捏合都是由大
    #: 变小，过程中鼠标已经移动了"）。他自己 701 帧上量化过（锚定每次锁 1000ms，
    #: 统计光标被锁的时间占比，连续 3 帧确认）：
    #:     0.70 → 移动 8.3%／捏合覆盖 86.2%
    #:     0.85 → 移动 9.9%／捏合覆盖 86.2%   ← **覆盖率不变，只多 1.6 点误冻**
    #:     1.00 → 移动 22.3%／停住 6.3%／双指 24.0%   ← 开始明显变差
    #:     1.15 → 双指 50% 被冻           ← 不可用
    #: 所以 0.85 是"再往上就要付代价"的边界。
    #: 🔴 另外试过两种"更聪明"的提前触发，实测都更差，别再试：
    #:   * 单帧大幅下降（快捏）：移动误冻 32.2%
    #:   * 相对下降（近 600ms 最大值 ×0.62）：移动 19~31%、双指 51~60%
    pinch_anchor_ratio: float = 0.90
    right_pinch_approach_delta: float = 0.22
    # 右键捏合总开关。默认关闭：实机中"握拳"与"拇指碰中指"在 2.5D
    # 投影上不可分（握拳时拇指-中指必然靠近，且 MCP-PIP-TIP 夹角会蹭过
    # 判据阈值），导致握拳频繁弹右键。需要右键时改用键盘或鼠标侧键。
    right_click_enabled: bool = False
    # 捏合持续时间门槛：短于此值的快速掠过视为误触，不触发点击/拖拽。
    # 实机觉得太容易误触时调大（如 200ms），觉得反应迟顿时调小。
    pinch_hold_ms: int = 100
    drag_hold_ms: int = 5000
    # 拖拽锚定式光标的位移增益：屏幕行程 = gain × 掌心屏幕位移。
    # 掌心刚体行程比食指尖短，实机觉得拖拽偏慢时只调此值，不改代码。
    drag_gain: float = 1.0
    # 双击（捏合两下）判定窗口：窗口内第二次**同位置**捏合升级为双击。
    #
    # 默认 350 保持旧的"挂起等双击"语义（那条路径下每次单击都要等这么久，
    # 所以线上他开的是 `click_immediate=True`）。
    # 他那份 config.json 里设的是 **700**：用他 31002 帧使用日志还原 458 段
    # 捏合，相邻两次间隔中位 **422ms**，只有 51% 落在 450ms 内。在"立即落地"
    # 路径下这个窗口只用于**升级第二次捏合为双击**，放宽没有单击延迟的代价。
    double_click_window_ms: int = 350
    # 两次捏合落点的最大间距（归一化画面坐标）：超过则按两次独立单击。
    double_click_max_drift: float = 0.05
    fist_hold_ms: int = 500
    hand_lost_ms: int = 200
    scroll_threshold: float = 0.12
    scroll_sensitivity: float = 1.0
    rotation_gestures_enabled: bool = True
    palm_rotation_threshold: float = 0.18
    page_gestures_enabled: bool = True
    page_swipe_threshold: float = 0.70

    # ---- 进入门（测量域闸门）----
    # 全局闸门：未进入控制态时，下游手势（捏合/翻页/拖拽）一律不执行，从根上
    # 挡住"脸被误判成手"产生的幻影动作。默认关，与 virtual_desk_experimental
    # 一样先在一轮引导基线里验证，再决定是否转正。
    entry_gate_enabled: bool = False
    #: 「拇指张开=停住」。默认关：拇指张开度必须按真实手型校准阈值，
    #: 合成测试数据的拇指本来就外展，直接启用会误伤正常指向。
    thumb_stop_enabled: bool = True
    #: 拇指张开度进入/退出阈值。加大更好触发，减小更不容易误触。
    #: 实测：收起 0.6~0.7，张开 0.86~1.44。默认 0.88 偏保守。
    thumb_stop_threshold: float = 0.88
    thumb_resume_threshold: float = 0.75

    # ---- 个人手势样本库（数据驱动识别）----
    # 先用 `python -m handmouse.collect` 采集自己的手势，再把这里打开。
    # 关闭时完全走既有几何判据，行为与以前一致。
    gesture_model_enabled: bool = False    #: 样本库路径。留空 = gesture_data/samples.jsonl。
    gesture_model_path: str = ""
    #: 拒识距离：超过这个值就回答"不认识"，回退几何判据。
    #: 0 = 每次启动按用户自己的样本现算（suggest_reject_distance）。
    gesture_model_reject_distance: float = 0.0
    #: 相对判据：第二近的异类要比最近邻远这么多倍才算明确识别。
    gesture_model_margin_ratio: float = 1.8
    #: 双指（swipe）模型票的**深信距离**：最近邻 ≤ 此值时模型直接采纳、
    #: 几何不参与；超过则落入犹豫区，必须过几何可行性闸（防单指误报）。
    #: 2026-10-22 用户实机标定：真双指 0.53/0.67，单指误报 0.91/0.94，
    #: 0.8 卡在中间。单指若再被深信区误触发，把此值往下调并报 @数字。
    gesture_model_deep_distance: float = 0.8
    # 进入姿势（掌心朝镜 + 五指张开）需连续保持的时间。退出复用既有常数：
    # 握拳退出沿用 fist_hold_ms，手出画退出沿用 hand_lost_ms。
    entry_hold_ms: int = 300
    # 朝向判定的掌面朝向单位量幅度下限（0-5-17 三角法向 z 分量归一化后）。
    # 这是"几何退化帧底线"，不是"掌/背分界"：掌/背靠期望符号区分，本值只用来
    # 丢弃 0/5/17 三点近共线、朝向量沦为噪声的帧。阈值 0.30 的依据是「清掉错号
    # 最严重的 L-back 帧 p10≈0.10，正常帧 |nz| 中位≈0.95 有巨大余量、不误杀」，
    # **不是**"0.30 足以区分掌/背"。它**不解决符号噪声**——右手背的原始符号统计
    # 曾显示约四成帧不匹配，后经特征化确认那部分是翻手/换手的 lag 块（成块翻转、
    # 伴 |nz| 谷），不是逐帧噪声；由符号锁存（entry_palm_nz_commit + 帧数）压掉。
    entry_palm_nz_min: float = 0.30
    # 符号锁存的"参与门"：只有 |nz| ≥ 本值的帧才允许改变已锁定的朝向符号。
    # 与上面的最低置信度门职责不同——|nz| 落在 [min, commit) 区间的帧"看得清
    # （进得去）但不够格改变朝向判定"，构成迟滞。实测右手背换手过渡块
    # |nz|≈0.46~0.56，0.50 门可吃掉其大部分。必须 ≥ entry_palm_nz_min（见 validate）。
    entry_palm_nz_commit: float = 0.50
    # 符号锁存的确认帧数：新符号需连续 K 帧同号且都过 commit 门才被采纳。
    # 8 帧 ≈ 0.5s @8.5fps，覆盖实测最长过渡块（6 帧）留 2 帧余量。
    # K 与 commit 一次只调一个变量，否则无法归因是哪个起作用。
    entry_facing_commit_frames: int = 8

    # ---- KI-002 轨迹纠正（默认关，实机观测到 handedness 翻转频繁时打开） ----
    enable_trajectory_correction: bool = False
    # 同一 wrist 轨迹下标签翻转纠正的 wrist 距离阈值（归一化画面坐标）。
    # 小于此值视为同一追踪目标连续运动，继承上一帧 handedness。
    trajectory_correction_wrist_dist: float = 0.15
    # 轨迹纠正的时间窗口（ms）。超过此窗口未更新视为追踪链断裂，采纳新标签。
    trajectory_correction_window_ms: int = 500

    # ---- 摇杆式光标（默认关，实机验证体感后再决定转正） ----
    # 与绝对映射的本质区别：输入是食指尖偏离画面中心的方向+幅度，
    # 输出是速度，位置靠帧间累加。手臂可保持稳定，只动手指。
    cursor_joystick_enabled: bool = False
    # 中心死区半径（画面宽度比例）：偏离小于此值时光标不动，过滤生理手抖。
    cursor_deadzone_radius: float = 0.06
    # 满速时光标每秒移动的归一化距离（1.0 = 1 秒从屏左移到屏右）。
    cursor_max_speed: float = 4.0
    # 用户自定义速度倍数：填 1.5 整体快 50%，填 2.0 翻倍。
    # 运行中按 + / - 实时微调（步长 0.2），适合就好，无需改配置文件。
    cursor_speed_multiplier: float = 1.0

    # ---- 局部续跟踪 ----
    partial_finger_tracking: bool = True
    # 0 表示只按光流质量决定续跟踪寿命；正数用于诊断或保守限时。
    partial_tracking_frames: int = 0

    # ---- 控件磁吸（需要 handmouse[uia]）----
    cursor_snap_enabled: bool = True
    # 2026-10-20 由 35 降到 22：用户反馈磁吸太黏，光标路过就被拽走。
    # 释放半径 = 本值 × release_multiplier(1.8) ≈ 40px，仍能稳住已吸住的目标。
    #: 2026-10-22 用户要求「把鼠标能操作的一切都设上磁吸」后由 22 提到 34：
    #: 他是隔空指，落点比真鼠标散得多，22px 的捕获圈太小、经常够不着。
    #: 释放半径 = 本值 × release_multiplier(1.8) ≈ 61px，稳住后不会轻易弹出。
    cursor_snap_radius_px: int = 34
    #: 磁吸速度门：光标移动速度超过本值（像素/秒）时不做吸附。
    #: 快速掠过时吸附会不停拽目标，只有慢下来精调才该吸。
    #: 2026-10-22 由 1500 提到 2600：实测他的光标常在 1500~2500 px/s，
    #: 旧值等于"移动中完全关掉吸附"，只剩静止时才有，观感就是"没有磁吸"。
    cursor_snap_max_speed_px_s: float = 2600.0
    #: 只在光标**已进入**控件矩形内部时才吸附（2026-10-21 用户要求）。
    #:
    #: 旧行为是从矩形外 radius_px 就开始往里拽，表现是"路过就被吸走"。
    #: 现在的捏合趋向锚定已经能在点之前把光标钉住，磁吸不必再负责"够不着"，
    #: 只需负责"落点居中"，所以改成进框才吸。
    cursor_snap_inside_only: bool = True
    #: 吸附时把光标钉到控件**正中心**，而不是 UIA 的 GetClickablePoint。
    #: 中心的落点可预期；clickable point 有时偏到控件一角。
    cursor_snap_to_center: bool = True
    #: 吸附时是否在全屏透明层上画动态彩色光环（点击穿透，不挡操作）。
    snap_halo_enabled: bool = True
    #: 磁吸是否只在"预捏合窗口"内开启（2026-10-21 用户选定方案）。
    #:
    #: ``True``（默认）= 光标自由移动时**完全不磁吸**，只在"原地停留满
    #: pinch_pre_dwell_ms，或已触发捏合趋向锚定"这一小段才把光标拉向控件。
    #: 平时光标完全跟手，点按钮前一刻才轻轻吸附，既不乱吸也不影响手感。
    #:
    #: ``False`` = 只要光标在控件上就一直吸。
    #:
    #: 🔴 2026-10-22 **保持 ``True``**。我一度改成 False（想着"随时吸"更好瞄），
    #: 被他纠正：**磁吸会拉低光标帧率**，所以故意只在"停在某处一秒、
    #: 准备捏合"这一小段才启用。这段开关就是他要的"捏合趋向里才吸"。
    #: 真正该改的不是"吸不吸"，而是**吸谁**（文字不该吸、标题栏按钮该吸）。
    cursor_snap_windowed: bool = True

    # ---- 预捏合（停留 = 要捏合了）----
    #: 光标在**原地停留**超过这么久（毫秒），即视为"我要捏合了"——
    #: 这时才开启磁吸（2026-10-21 用户提议：停留是比拇指距离更稳的
    #: "要点击"信号，可让磁吸只在临门一脚时工作，平时不黏光标）。
    #: 仅作为触发时机，不改动你已有的磁吸半径/容差等手感参数。
    pinch_pre_dwell_ms: int = 1000
    #: 判定"原地停留"的**激活**半径（低通后的屏幕归一化坐标，0~1）。
    #:
    #: 🔴 2026-10-23 实测改定：原值 0.005（≈9.6px）**物理上不可达**。
    #: 用他自己 460 秒真实使用日志（14270 个 pointer 帧）回放：
    #: 整段里最安静的 1 秒，从锚点的漂移还有 **24.8px**（p1=33.8px），
    #: 而"静止"时逐帧抖动就有 2~4px——锚点式累计位移会一路累加，
    #: 9.6px 几十毫秒就越界，计时永远清零 → dwell **一次都没亮过**
    #: （回放实测激活 0 次、占比 0.0%），磁吸窗口因此从未打开。
    #:
    #: 现在改成两级：
    #: * **激活**：先对位置做 `pinch_pre_dwell_tau_s` 的指数低通，再比半径；
    #:   低通把生理抖动滤掉，留下真实漂移，所以半径要给到抖动之上。
    #: * **解除**：另用一个**短窗**位移判据（`pinch_pre_dwell_release_*`），
    #:   故意用比激活更短的时间尺度 —— 否则"要等一秒才认得出动"，
    #:   光标会被钉住半秒以上。非对称是刻意的。
    #: 0.030 ≈ 屏宽 3%（1920 屏约 58px），回放激活率 2.7 次/分、最长 1.75s。
    pinch_pre_dwell_epsilon: float = 0.030
    #: 停留判据的低通时间常数（秒）。越大越忽略抖动、但跟手越慢。
    #: 0.55 在回放上与 eps=0.030 配合最稳（近旁几档差别很小）。
    pinch_pre_dwell_tau_s: float = 0.55
    #: 解除停留用的**短窗**（毫秒）与窗内允许位移（归一化）。
    #: 100ms 内移动超过 0.025（≈48px，即持续约 480px/s）就立刻解除。
    #: 这一条负责"一动就松"——它的时间尺度必须比激活短得多，否则光标被
    #: 钉住之后要等半秒才认得出他在动，手感变成"黏住了甩不掉"。
    pinch_pre_dwell_release_ms: int = 100
    pinch_pre_dwell_release_epsilon: float = 0.025
    #: 解除停留用的**长窗**：500ms 内移动超过 0.060（≈115px，即持续
    #: 230px/s）也解除。短窗抓不住"缓慢但持续"的漂移，这一条补上。
    pinch_pre_dwell_drift_ms: int = 500
    pinch_pre_dwell_drift_epsilon: float = 0.060
    #: 捏合趋向锚定要**同时**满足"已经停稳这么久"才允许打开磁吸窗口。
    #: 这条是 2026-10-23 修"磁吸左右弹跳"的关键：捏合比在正常移动中
    #: 就会频繁掉到 0.90 以下，若单靠它开窗口，光标会被反复拽到某个控件上、
    #: 一秒后又松回来 —— 他看到的正是"夹在两个小图标中间左右弹跳"。
    #: 现在移动中捏合只**冻结**光标（原来的防漂移作用不变），不触发磁吸。
    pinch_snap_still_ms: int = 200
    #: 停留满 `pinch_pre_dwell_ms` 之后，是否**同时冻结光标**。
    #:
    #: 🔴 停留其实做两件事，语义完全不同，2026-10-23 拆成可分别关闭：
    #:
    #: * **开磁吸窗口**（`snap_engaged`）：吸到控件中心才需要它，必须留着；
    #: * **冻结光标**（`cursor_frozen`）：防止"要点击了手还在漂"。但它是
    #:   2026-10-22 才加的行为，而现在 dwell 判据修好之后**它第一次真的会触发**
    #:   ——用户很可能觉得"我只是停了一下，光标怎么就冻了"。
    #:
    #: 关掉这个开关只影响冻结，磁吸窗口照旧按停留打开（这是两者的关键区别）。
    #: 默认 True = 保持 2026-10-22 定下的手感不变。
    pinch_pre_dwell_freeze: bool = True
    #: 光标旁的小彩色圆环：绿=控制中 橙=捏合 蓝=拖拽 红=未进入控制态 灰=暂停/丢手。
    #: 真实使用时没有预览窗，靠它用颜色判断状态（2026-10-21 用户要求）。
    state_indicator_enabled: bool = True
    #: 暂停状态下，对镜头比**五指张开**并保持 `wake_hold_ms` 就恢复控制。
    #:
    #: 2026-10-23 用户要求：「只要我们对镜头比五根手指就能启动」。
    #: 配合"开机自启 + 默认暂停"就是他要的免命令行启动。
    #: ⚠️ 两条代价必须让他知道：① 程序要**常驻**（摄像头指示灯常亮）；
    #: ② 暂停时仍以 10Hz 提交识别（`vision` 里限流），单帧开销很低但不是零。
    wake_by_open_palm: bool = True
    #: 五指张开要保持多久才算"我要开始用"（毫秒）。太短会被路过的手骗到。
    wake_hold_ms: int = 800
    #: 窗口底部白色悬浮球：拉它来扩展/缩放窗口（2026-10-22 用户要求）。
    #:
    #: 球画在点击穿透层上，命中判定由我们自己算；**抓住球的那次按下会被吞掉**，
    #: 不发给操作系统，所以拉球不会顺手点到球下面的东西。
    #: 向左拉=左边向外扩，向右拉=右边向外扩，向上拉=整体缩小，向下拉=整体放大，
    #: 松手带弹簧回弹；只有前台窗口有球，静止 2 秒淡出。
    window_ball_enabled: bool = True

    # ---- 使用记录（排查"哪里反复点不中"）----
    #: 把逐帧的控制信号（光标位置、手势、捏合比、点击）追加写进
    #: ``logs/usage-<时间戳>.jsonl``，供 `scripts/analyze_usage.py` 分析。
    #: **不记录图像、不记录窗口标题**，只有控制量。
    usage_log_enabled: bool = False

    # ---- 首次设置 ----
    #: 是否已完成首次向导（标点 + 量速率 + 过一遍手势）。
    #: 🔴 默认 **False**：新用户下载后必须先录自己的习惯，否则用的全是作者
    #: 按自己的手标定的阈值——那正是"换个人就不好用"的根源。
    #: 启动时不阻断（只提示），因为录一次要两分钟，强制会把人赶走。
    onboarding_done: bool = False
    #: 首次设置提醒是否已经提示过一次。**只提示一次**，用户不做就不再打扰
    #: （用户原话：「采集只采集一次啊，不要每次开启都采集一遍」）。
    #: 想重新录：自己跑 `--onboarding`（重录手势）或 `--recalibrate`（只重标定）。
    onboarding_reminded: bool = False

    # ---- 屏幕 / 多屏（ScreenMap）----
    multi_monitor: bool = False
    # 实验开关：multi_monitor=False 时也走 MOUSEEVENTF_VIRTUALDESK，
    # 用于在多屏机器上先验证官方推荐路径，再决定是否把 multi_monitor 打开。
    virtual_desk_experimental: bool = False

    @classmethod
    def from_mapping(cls, raw: dict[str, object]) -> "AppConfig":
        """忽略未知字段，避免旧配置文件带崩新版本。"""

        allowed = {item.name for item in fields(cls)}
        values = {key: value for key, value in raw.items() if key in allowed}
        return cls(**values)  # type: ignore[arg-type]

    def effective_margins(self) -> tuple[float, float, float, float]:
        """活动区的 (left, top, right, bottom) 内缩量。

        四边独立值优先；某一侧没设（``None``）就回退到对应的对称值。
        映射层、HUD、手感快捷键都必须走这个函数，否则会各算各的。
        """

        return (
            self.pointer_margin_x
            if self.pointer_margin_left is None
            else float(self.pointer_margin_left),
            self.pointer_margin_y
            if self.pointer_margin_top is None
            else float(self.pointer_margin_top),
            self.pointer_margin_x
            if self.pointer_margin_right is None
            else float(self.pointer_margin_right),
            self.pointer_margin_y
            if self.pointer_margin_bottom is None
            else float(self.pointer_margin_bottom),
        )

    def validate(self) -> None:
        """保存或使用前的防御性校验，全部就地钳到安全区间。"""

        self.camera_index = max(0, int(self.camera_index))
        self.camera_width = max(320, int(self.camera_width))
        self.camera_height = max(240, int(self.camera_height))
        self.camera_fps = max(5, min(60, int(self.camera_fps)))
        self.camera_allow_fallback = bool(
            getattr(self, "camera_allow_fallback", True),
        )
        self.onboarding_done = bool(getattr(self, "onboarding_done", False))
        self.onboarding_reminded = bool(getattr(self, "onboarding_reminded", False))
        # 未知取值一律退回 dshow：宁可走兼容性最好的那条路，也不要因为拼错
        # 一个字母就换到可能坏掉的后端上（表现是"摄像头打不开"）。
        backend = str(getattr(self, "camera_backend", "dshow") or "").strip().lower()
        self.camera_backend = backend if backend in {"dshow", "msmf", "any"} else "dshow"
        self.recognition_zoom = max(1.0, min(1.35, float(self.recognition_zoom)))
        self.inference_width = max(0, int(getattr(self, "inference_width", 0)))
        self.pointer_stability = max(0.0, min(1.0, float(self.pointer_stability)))
        self.pointer_response = max(0.0, min(1.0, float(self.pointer_response)))
        # 未知取值一律退回默认 "index_tip"：宁可不动映射，也不能静默进 palm
        # 模式——那是改所有 MOVE 坐标的开关，拼错一个字母就该当没写。
        anchor = str(getattr(self, "pointer_anchor", "index_tip") or "").strip().lower()
        self.pointer_anchor = anchor if anchor in {"index_tip", "palm"} else "index_tip"
        self.pointer_anchor_offset_tau_s = max(
            0.10,
            min(5.0, float(getattr(self, "pointer_anchor_offset_tau_s", 1.0))),
        )
        self.cursor_interpolation_ms = max(12, min(60, int(self.cursor_interpolation_ms)))
        self.pointer_margin_x = max(0.0, min(0.4, float(self.pointer_margin_x)))
        self.pointer_margin_y = max(0.0, min(0.4, float(self.pointer_margin_y)))
        # 四边独立内缩：None 保持 None（= 回退到对称值），有值就钳到安全区间。
        # 上限取 0.5（比对称的 0.4 宽）：实测该用户纵向舒适区只到 y≈0.47，
        # 需要 margin_top≈0.45 才能把屏幕上沿纳入可达范围，0.4 会被截断。
        for name in (
            "pointer_margin_top",
            "pointer_margin_bottom",
            "pointer_margin_left",
            "pointer_margin_right",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            setattr(self, name, max(0.0, min(0.5, float(value))))
        # 上下（或左右）加起来必须留出正的活动区，否则映射退化。
        if self.pointer_margin_top is not None and self.pointer_margin_bottom is not None:
            if self.pointer_margin_top + self.pointer_margin_bottom > 0.9:
                self.pointer_margin_bottom = max(0.0, 0.9 - self.pointer_margin_top)
        if self.pointer_margin_left is not None and self.pointer_margin_right is not None:
            if self.pointer_margin_left + self.pointer_margin_right > 0.9:
                self.pointer_margin_right = max(0.0, 0.9 - self.pointer_margin_left)
        self.pinch_engage = max(0.1, min(1.0, float(self.pinch_engage)))
        self.pinch_release = max(self.pinch_engage + 0.02, min(1.2, float(self.pinch_release)))
        self.pinch_arm = max(self.pinch_release + 0.05, min(1.4, float(self.pinch_arm)))
        # 停留（预捏合）判据：激活用低通半径，解除用短窗位移，尺度别搞反。
        self.pinch_pre_dwell_ms = max(200, min(3000, int(self.pinch_pre_dwell_ms)))
        self.pinch_pre_dwell_epsilon = max(
            0.002,
            min(0.30, float(self.pinch_pre_dwell_epsilon)),
        )
        self.pinch_pre_dwell_tau_s = max(0.05, min(3.0, float(self.pinch_pre_dwell_tau_s)))
        self.pinch_pre_dwell_release_ms = max(
            40,
            min(2000, int(self.pinch_pre_dwell_release_ms)),
        )
        self.pinch_pre_dwell_release_epsilon = max(
            0.005,
            min(0.60, float(self.pinch_pre_dwell_release_epsilon)),
        )
        self.pinch_pre_dwell_drift_ms = max(
            100,
            min(3000, int(self.pinch_pre_dwell_drift_ms)),
        )
        self.pinch_pre_dwell_drift_epsilon = max(
            0.005,
            min(0.60, float(self.pinch_pre_dwell_drift_epsilon)),
        )
        self.pinch_snap_still_ms = max(0, min(2000, int(self.pinch_snap_still_ms)))
        self.right_pinch_approach_delta = max(
            0.10,
            min(0.60, float(self.right_pinch_approach_delta)),
        )
        self.drag_hold_ms = max(300, min(10_000, int(self.drag_hold_ms)))
        self.pinch_hold_ms = max(0, min(1000, int(self.pinch_hold_ms)))
        self.drag_gain = max(0.1, min(5.0, float(self.drag_gain)))
        self.double_click_window_ms = max(
            100, min(1000, int(self.double_click_window_ms)),
        )
        self.double_click_max_drift = max(
            0.01, min(0.5, float(self.double_click_max_drift)),
        )
        self.fist_hold_ms = max(300, min(1500, int(self.fist_hold_ms)))
        self.hand_lost_ms = max(100, min(1000, int(self.hand_lost_ms)))
        self.scroll_threshold = max(0.03, min(0.6, float(self.scroll_threshold)))
        self.scroll_sensitivity = max(0.2, min(4.0, float(self.scroll_sensitivity)))
        self.palm_rotation_threshold = max(0.08, min(0.60, float(self.palm_rotation_threshold)))
        self.page_swipe_threshold = max(0.35, min(1.5, float(self.page_swipe_threshold)))
        self.entry_hold_ms = max(80, min(1500, int(self.entry_hold_ms)))
        self.entry_palm_nz_min = max(0.0, min(1.0, float(self.entry_palm_nz_min)))
        self.entry_palm_nz_commit = max(0.1, min(0.95, float(self.entry_palm_nz_commit)))
        # 参与门必须 ≥ 进入门：否则"够格改朝向"比"进得去"还松，迟滞消失——配置
        # 写错会静默失效，故在此强制拉齐。
        self.entry_palm_nz_commit = max(self.entry_palm_nz_commit, self.entry_palm_nz_min)
        self.entry_facing_commit_frames = max(2, min(30, int(self.entry_facing_commit_frames)))
        self.partial_tracking_frames = max(0, min(600, int(self.partial_tracking_frames)))
        self.min_detection_confidence = max(0.25, min(0.90, float(self.min_detection_confidence)))
        self.min_presence_confidence = max(0.25, min(0.90, float(self.min_presence_confidence)))
        self.min_tracking_confidence = max(0.25, min(0.90, float(self.min_tracking_confidence)))
        self.trajectory_correction_wrist_dist = max(
            0.05, min(0.5, float(self.trajectory_correction_wrist_dist)),
        )
        self.trajectory_correction_window_ms = max(
            100, min(2000, int(self.trajectory_correction_window_ms)),
        )
        self.cursor_deadzone_radius = max(0.0, min(0.3, float(self.cursor_deadzone_radius)))
        self.cursor_max_speed = max(0.1, min(10.0, float(self.cursor_max_speed)))
        self.cursor_speed_multiplier = max(0.2, min(10.0, float(self.cursor_speed_multiplier)))
        self.cursor_snap_radius_px = max(12, min(80, int(self.cursor_snap_radius_px)))
        if self.handedness not in {"Left", "Right", "Any"}:
            self.handedness = "Right"


class ConfigStore:
    """以原子替换方式管理 JSON 配置。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_CONFIG_PATH

    def load(self) -> AppConfig:
        if not self.path.exists():
            return AppConfig()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            config = AppConfig.from_mapping(raw)
            config.validate()
            return config
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"[config] 读取 {self.path} 失败，使用默认值：{exc}")
            return AppConfig()

    def save(self, config: AppConfig) -> None:
        config.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(asdict(config), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def reset(self) -> AppConfig:
        config = AppConfig()
        self.save(config)
        return config
