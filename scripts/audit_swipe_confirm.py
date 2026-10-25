"""第五版审计：带**连续帧确认**的时序网格搜索（最终定门限用）。

第四版发现的关键事实：单帧判据命中率在 35%~96% 之间抖动（不是某些姿势永不成立），
所以真正的杠杆是**时序**：
  * 进入 SWIPE 要求连续 N 帧成立（N=2 就能把孤立的单帧误报几乎清零）
  * 进入之后用 hold 窗口吞掉抖动（不掉回「只伸食指」= 光标不瞬移）

本脚本以**真实帧序**为目标函数做搜索：
  * 主目标：swipe 会话里处于 SWIPE 的帧占比（可用度）尽量高
  * 约束：move / stop / pinch 的误报帧占比尽量低（move 是"光标乱飞"的来源）

运行：python scripts/audit_swipe_temporal.py 之后跑本脚本
      python scripts/audit_swipe_confirm.py
"""

from __future__ import annotations

import itertools

from audit_swipe_temporal import feats, load_sequences  # noqa: E402


def make(ti, tm, tro, tpo, ang):
    def rule(x):
        if not (x["ti"] >= ti and x["tm"] >= tm and x["tr"] <= tro and x["tp"] <= tpo):
            return False
        if ang and x["a_m"] < 135.0:
            return False
        return True

    return rule


def simulate(seq, rule, hold_ms, confirm):
    """返回 (active 帧数, 总帧数, 进入次数)。confirm=连续成立帧数要求。"""
    hold_ns = hold_ms * 1_000_000
    streak = 0
    active = False
    last_hit = None
    count = 0
    entries = 0
    for ts, x in seq:
        hit = rule(x)
        streak = streak + 1 if hit else 0
        if hit:
            last_hit = ts
        if not active:
            if streak >= confirm:
                active = True
                entries += 1
            else:
                continue
        else:
            if last_hit is None or ts - last_hit > hold_ns:
                active = False
                continue
        count += 1
    return count, len(seq), entries


def score(seqs, rule, hold_ms, confirm):
    out = {}
    for label in ("swipe", "move", "stop", "pinch", "open", "fist"):
        tot = hit = 0
        for (lb, _src), s in seqs.items():
            if lb != label or len(s) < 5:
                continue
            h, n, _e = simulate(s, rule, hold_ms, confirm)
            hit += h
            tot += n
        out[label] = hit / tot if tot else 0.0
    return out


def main():
    seqs = load_sequences()
    grids = {
        "ti": (0.6, 0.7, 0.75, 0.8, 0.85),
        "tm": (0.6, 0.65, 0.7, 0.75, 0.8, 0.85),
        "tro": (0.45, 0.5, 0.55, 0.6, 0.65),
        "confirm": (1, 2, 3),
        "hold": (250, 400),
    }
    rows = []
    for ti, tm, tro, confirm, hold in itertools.product(
        grids["ti"], grids["tm"], grids["tro"], grids["confirm"], grids["hold"]
    ):
        rule = make(ti, tm, tro, tro, False)
        r = score(seqs, rule, hold, confirm)
        rows.append((r, ti, tm, tro, confirm, hold))

    # 约束：move <= 5%，stop <= 2%，pinch <= 4%，open <= 2%；按 swipe 可用度排序
    ok = [x for x in rows if x[0]["move"] <= 0.05 and x[0]["stop"] <= 0.02 and x[0]["pinch"] <= 0.04 and x[0]["open"] <= 0.02]
    ok.sort(key=lambda x: -x[0]["swipe"])
    print("=== 约束内 top 20（move<=5% stop<=2% pinch<=4% open<=2%）===")
    for r, ti, tm, tro, confirm, hold in ok[:20]:
        print(
            f"  ti>={ti} tm>={tm} tr,tp<={tro} confirm={confirm} hold={hold}  ->  "
            + "  ".join(f"{lb} {r[lb]*100:5.1f}%" for lb in ("swipe", "move", "stop", "pinch", "open", "fist"))
        )
    if not ok:
        print("  无可行解，放宽约束")
        rows.sort(key=lambda x: -(x[0]["swipe"] - x[0]["move"] * 3))
        for r, ti, tm, tro, confirm, hold in rows[:20]:
            print(
                f"  ti>={ti} tm>={tm} tr,tp<={tro} confirm={confirm} hold={hold}  ->  "
                + "  ".join(f"{lb} {r[lb]*100:5.1f}%" for lb in ("swipe", "move", "stop", "pinch", "open", "fist"))
            )


if __name__ == "__main__":
    main()
