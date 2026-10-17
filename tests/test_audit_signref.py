"""sign-ref 审计脚本的合成日志测试。

fixture 由 LogBuilder 逐帧拼出与真实结构化日志同格式的文本，
不依赖实机日志文件，保证测试可重复。

与真实日志一致的两点模型（231852 实物确认）：
- 无手间隙日志整段静默（采集层不产包），段界 = 时间空洞；
  空洞后重入首帧带 state=lost（引擎失联迁移），且该帧仍有 palm 数据。
- 新格式 palm_presented 在闸门开后恒 False（① 语义变更），姿势分类
  只能用 nz 符号对 expected_sign(hn, mirror) 的匹配率；fixture 里各段
  nz/hn 取值已按真实符号约定排布（mirror=True：右掌 hn=Left nz+、
  右背 hn=Left nz−、左掌 hn=Right nz−、左背 hn=Right nz+）。
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "audit_signref.py"

BASE = datetime(2026, 10, 17, 10, 0, 0)
FRAME_DT = 0.033  # 约 30fps，timedelta 微秒级精确


def _ts(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S,") + f"{t.microsecond // 1000:03d}"


class LogBuilder:
    """按真实 sign-ref 日志的行格式逐帧拼接文本。"""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.t = BASE
        self._ts_seq = 0
        self._reentry_lost = False

    def header(self, old: bool = False, env: bool = True) -> "LogBuilder":
        if old:
            self.lines.append(
                f"{_ts(self.t)} [sign-ref运行头] mirror=True entry_palm_sin_min=0.10"
                " （仅参照模式：只加诊断日志、不改任何判定）"
            )
        else:
            self.lines.append(
                f"{_ts(self.t)} [sign-ref运行头] mirror=True entry_palm_nz_min=0.30"
                " entry_palm_nz_commit=0.50 entry_facing_commit_frames=8"
                " （仅参照模式：只加诊断日志、不改任何判定）"
            )
        if env:
            self.lines.append(
                f"{_ts(self.t)} [环境] commit=abc1234 config_sha256=deadbeef0000"
                " camera=0@640x480@30fps mirror=True det=0.4 pres=0.35 track=0.4"
                " snap=True safe=True"
            )
        return self

    def frame(
        self,
        gate: bool = False,
        state: str = "idle",
        palm: bool | None = None,
        hn: str = "Left",
        score: float | None = 0.95,
        nz: float = 0.85,
        old: bool = False,
    ) -> "LogBuilder":
        self.t += timedelta(seconds=FRAME_DT)
        self._ts_seq += 1
        if self._reentry_lost:
            state = "lost"
            self._reentry_lost = False
        self.lines.append(
            f"{_ts(self.t)} geom ts={self._ts_seq} arrival=0 arrive_ms=16"
            f" gate_active={gate} entry_started=False held_ms=0 flips=0"
            f" state={state} gesture={state}"
        )
        if palm is None:
            return self
        self.lines.append(
            f"{_ts(self.t)}   palm_width=0.1500 span=0.5000"
            f" frontal_ratio=0.3000(min=0.25) → palm_presented={palm}"
            f" hn={hn} valid=True"
        )
        if old:
            self.lines.append(
                f"{_ts(self.t)}   [sign-ref] cross=-0.0067 sin=0.605(min=0.10)"
                f" hn={hn} frontal=0.3000 mirror=True（仅参照模式，不影响判定）"
            )
        else:
            score_field = f" hn_score={score:.4f}" if score is not None else ""
            self.lines.append(
                f"{_ts(self.t)}   [sign-ref] nz_unit={nz:.6f} |nz|={abs(nz):.4f}(min=0.30)"
                f" hn={hn}{score_field} frontal=0.3000 mirror=True"
                "（仅参照模式，不影响判定）"
            )
        return self

    def segment(
        self,
        frames: int,
        palm_presented,
        gate_open_at: int | None = None,
        hn: str = "Left",
        nz=0.85,
        score: float | None = 0.95,
        old: bool = False,
    ) -> "LogBuilder":
        for i in range(frames):
            gate = gate_open_at is not None and i >= gate_open_at
            pp = palm_presented(i) if callable(palm_presented) else palm_presented
            frame_nz = nz(i) if callable(nz) else nz
            self.frame(gate=gate, palm=pp, hn=hn, nz=frame_nz, score=score, old=old)
        return self

    def gap(self, hole: float = 4.0, handlost: bool = True) -> "LogBuilder":
        """无手间隙：日志静默 hole 秒；handlost=True 时重入首帧 state=lost。"""

        self.t += timedelta(seconds=hole)
        self._reentry_lost = handlost
        return self

    def write(self, path: Path, encoding: str = "utf-8") -> Path:
        path.write_text("\n".join(self.lines) + "\n", encoding=encoding)
        return path


def run_audit(log_path: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--log", str(log_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _full_run(old: bool = False, env: bool = True) -> LogBuilder:
    b = LogBuilder().header(old=old, env=env)
    b.segment(60, True, gate_open_at=20, hn="Left", nz=0.85, old=old)   # seg1 右掌
    b.gap()
    b.segment(60, False, hn="Left", nz=-0.85, old=old)                  # seg2 右背
    b.gap()
    b.segment(60, True, gate_open_at=25, hn="Right", nz=-0.85, old=old)  # seg3 左掌
    b.gap()
    b.segment(60, False, hn="Right", nz=0.85, old=old)                  # seg4 左背
    return b


def test_full_run_all_pass(tmp_path: Path) -> None:
    out = run_audit(_full_run().write(tmp_path / "sign-ref-t.log"))
    # 运行头指纹
    assert "| abc1234 | False | deadbeef0000 | True | entry_palm_nz_min=0.30 | 新(nz_unit/hn_score) |" in out
    # 间隙自动检测：重入首帧 lost 记 1 帧，空洞 4.0+0.033≈4.03s
    assert "| 1 | 1 | 4.03 | ✓ | seg2 |" in out
    assert "| 3 | 1 | 4.03 | ✓ | seg4 |" in out
    # 分段：期望/实测/帧数 + 开闸延迟（20 帧 × 33ms = 0.66s）
    assert "| seg1 | 掌 | 掌 | 60 |" in out
    assert "| seg2 | 背 | 背 | 60 |" in out
    assert "0.66" in out
    # 首帧判据含 nz_unit 与 hn_score
    assert "nz=+0.850" in out and "Left@0.950" in out
    # 汇总：手背段硬条件 + 加权拒绝率 + 掌段开闸
    assert "gate_active 合计 0 帧（硬条件 ✓）" in out
    assert "加权拒绝率 100.0%" in out
    assert "2/2 开闸" in out
    assert "SKIP 段：无" in out
    assert out.count("| PASS |") == 4


def test_single_lost_frame_gap_is_clean(tmp_path: Path) -> None:
    """边界：重入首帧仅 1 帧 lost（次帧即回画有手）→ gap 判干净，不误杀。

    状态机要 hand_count=0 持续 hand_lost_ms(200ms) 才迁移到 lost，故 1 帧
    lost 已自证 ≥200ms 无手，不额外要求 lost 跨帧时长（8.5fps 下“1 帧
    lost + 立即回画”的干净 gap 真实存在）。此前由 test_full_run_all_pass
    的 “| 1 | 1 | … | ✓” 断言隐式覆盖，显式化防重构回归。"""
    b = LogBuilder().header()
    b.segment(60, True, gate_open_at=20, hn="Left", nz=0.85)
    b.gap()  # 空洞 4.0s + 重入首帧恰好 1 帧 state=lost，次帧起恢复 idle
    b.segment(60, False, hn="Left", nz=-0.85)
    out = run_audit(b.write(tmp_path / "sign-ref-t.log"))
    assert "| 1 | 1 | 4.03 | ✓ | seg2 |" in out
    assert "| seg2 | 背 | 背 | 60 |" in out and "| PASS |" in out
    assert "SKIP 段：无" in out


def test_brief_reentry_merges_segments_and_skips(tmp_path: Path) -> None:
    """gap3 手回画太早（空洞 0.3s < GAP_MIN_S，hand_lost 未触发）：
    seg3/seg4 融合为一个检测连续段，掌/背判据各半 → 混合 SKIP，
    其帧不得进入手背段分母（H4 '假 100%' 口径陷阱）。"""
    b = LogBuilder().header()
    b.segment(60, True, gate_open_at=20, hn="Left", nz=0.85)
    b.gap()
    b.segment(60, False, hn="Left", nz=-0.85)
    b.gap()
    b.segment(60, True, gate_open_at=25, hn="Right", nz=-0.85)
    b.gap(hole=0.3, handlost=False)  # 回画太早：不构成段界
    # 被污染的 seg4：锁存未清零 gate 常开 + palm 恒 True。若计入分母
    # 会虚增 gate 帧数、拉低真实拒绝率——必须随混合段一起剔除。
    b.segment(60, True, gate_open_at=0, hn="Right", nz=0.85)
    out = run_audit(b.write(tmp_path / "sign-ref-t.log"))
    assert "| seg3 | 掌 | 混合 | 120 |" in out
    assert "SKIP" in out and "姿势混合" in out
    # 汇总只计 seg2：污染段的 60 帧 gate_on 不得混入
    assert "gate_active 合计 0 帧（硬条件 ✓）" in out
    assert "加权拒绝率 100.0%" in out
    assert "SKIP 段：seg3" in out
    assert "检测连续段数 3 ≠ 协议期望 4" in out


def test_hole_without_handlost_marks_skip(tmp_path: Path) -> None:
    """空洞够长但重入首帧非 lost（hand_lost 未触发）→ 下一段 SKIP。"""
    b = LogBuilder().header()
    b.segment(60, True, gate_open_at=20, hn="Left", nz=0.85)
    b.gap()
    b.segment(60, False, hn="Left", nz=-0.85)
    b.gap()
    b.segment(60, True, gate_open_at=25, hn="Right", nz=-0.85)
    b.gap(hole=4.0, handlost=False)  # 空洞成立但引擎未失联迁移
    b.segment(60, False, hn="Right", nz=0.85)
    out = run_audit(b.write(tmp_path / "sign-ref-t.log"))
    assert "✗ | seg4 |" in out
    assert "hand_lost 未触发" in out
    assert "SKIP 段：seg4" in out
    # 手背段分母只剩 seg2
    assert "gate_active 合计 0 帧（硬条件 ✓）" in out


def test_utf16_log_parsed(tmp_path: Path) -> None:
    # PowerShell Tee-Object 落盘的 stdout 是 UTF-16LE；结构化日志虽为 utf-8，
    # 脚本仍须 BOM sniff，防止指错文件时整份解析为空。
    path = _full_run().write(tmp_path / "sign-ref-t.log", encoding="utf-16")
    out = run_audit(path)
    assert "| seg4 |" in out
    assert out.count("| PASS |") == 4


def test_old_format_marks_na_and_parses_cross_sin(tmp_path: Path) -> None:
    # 旧格式（150336 及更早）：cross/sin 观测量、无 hn_score、无 [环境] 指纹行；
    # palm_presented 早于语义变更，仍是原始几何值，可用于分类。
    out = run_audit(_full_run(old=True, env=False).write(tmp_path / "sign-ref-t.log"))
    assert "| N/A | N/A | N/A | True | entry_palm_sin_min=0.10 | 旧(cross/sin) |" in out
    assert "cross=-0.0067 sin=0.605 Left" in out
    assert out.count("| PASS |") == 4
    assert "gate_active 合计 0 帧（硬条件 ✓）" in out


def test_mid_format_without_hn_score(tmp_path: Path) -> None:
    # 中格式（语义变更后至 3c09481 前）：nz_unit 已有、无 hn_score 字段。
    b = LogBuilder().header(env=False)
    b.segment(60, True, gate_open_at=20, hn="Left", nz=0.85, score=None)
    b.gap()
    b.segment(60, False, hn="Left", nz=-0.85, score=None)
    b.gap()
    b.segment(60, True, gate_open_at=25, hn="Right", nz=-0.85, score=None)
    b.gap()
    b.segment(60, False, hn="Right", nz=0.85, score=None)
    out = run_audit(b.write(tmp_path / "sign-ref-t.log"))
    assert "新(nz_unit) |" in out
    # 首帧判据有 nz 与 hn，无 @score
    assert "nz=+0.850" in out and "@0.950" not in out
    assert out.count("| PASS |") == 4


def test_palm_segment_never_opens_is_fail(tmp_path: Path) -> None:
    b = LogBuilder().header()
    b.segment(60, True, gate_open_at=None, hn="Left", nz=0.85)  # 掌段未开闸
    b.gap()
    b.segment(60, False, hn="Left", nz=-0.85)
    b.gap()
    b.segment(60, True, gate_open_at=25, hn="Right", nz=-0.85)
    b.gap()
    b.segment(60, False, hn="Right", nz=0.85)
    out = run_audit(b.write(tmp_path / "sign-ref-t.log"))
    assert "FAIL: 掌段未开闸" in out
    assert "1/2 开闸" in out


def test_back_segment_with_gate_frames_is_fail(tmp_path: Path) -> None:
    b = LogBuilder().header()
    b.segment(60, True, gate_open_at=20, hn="Left", nz=0.85)
    b.gap()
    b.segment(60, False, gate_open_at=40, hn="Left", nz=-0.85)  # 背段误开 20 帧
    b.gap()
    b.segment(60, True, gate_open_at=25, hn="Right", nz=-0.85)
    b.gap()
    b.segment(60, False, hn="Right", nz=0.85)
    out = run_audit(b.write(tmp_path / "sign-ref-t.log"))
    assert "FAIL: 手背段 gate_active 20 帧" in out
    assert "gate_active 合计 20 帧（硬条件 ✗）" in out
    # 实测分类仍为背（nz 判据与 gate 无关），与期望一致 → 无分类 ⚠
    assert "实测分类" not in out


def test_old_format_mixed_segment_is_skip(tmp_path: Path) -> None:
    b = LogBuilder().header(old=True)
    b.segment(60, True, gate_open_at=20, hn="Left", old=True)
    b.gap()
    # 旧格式的混合路径：palm_presented 各半（两姿势融合段）。
    b.segment(60, lambda i: i % 2 == 0, hn="Left", old=True)
    b.gap()
    b.segment(60, True, gate_open_at=25, hn="Right", old=True)
    b.gap()
    b.segment(60, False, hn="Right", old=True)
    out = run_audit(b.write(tmp_path / "sign-ref-t.log"))
    assert "| seg2 | 背 | 混合 |" in out
    assert "姿势混合" in out
    assert "加权拒绝率 100.0%" in out  # 分母只剩 seg4
    assert "SKIP 段：seg2" in out
