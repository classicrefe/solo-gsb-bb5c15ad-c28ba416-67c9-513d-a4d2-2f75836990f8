"""两源管混样配比（blend plan）求解。

从一次**成功**模拟的末态中任选两支源管，按目标总体积 ``target_volume``
与目标浓度 ``target_concentrations`` 反推两路取样量 ``x1``、``x2``：

* 体积守恒：``x1 + x2 = target_volume``；
* 每种标记物质量守恒：``c1[m]·x1 + c2[m]·x2 = target[m]·target_volume``，
  其中 ``c1`` / ``c2`` 为两源管末态浓度；源管或目标中缺失的标记物一律
  按 0 处理（目标未列出的标记物即要求混合后为 0）。

未知量只有两个，标记物方程构成超定系统：任一 ``c1[m] != c2[m]`` 的
标记物都给出一个候选 ``x1``，所有候选必须在浮点容差内一致，否则判定
各标记物方程自相矛盾（``BLEND_TARGET_INCONSISTENT``）；
``c1[m] == c2[m]`` 的标记物不提供拆分信息，但要求目标浓度等于该共同
浓度（两管都不含某标记物而目标要求非零时同样在此被拦截）。解出后校验：

* ``x1``、``x2`` 非负——为负说明目标浓度落在两源管可混合范围之外
  （``BLEND_TARGET_UNREACHABLE``）；
* 分别不超过两管末态可用体积（``BLEND_INSUFFICIENT_VOLUME``）。

若所有标记物两管浓度都相同（且目标与之相容），拆分方式不唯一，此时按
“优先从第一支源管取”的确定规则给出 ``x1 = min(target_volume, 可用1)``。

配比只是基于已保存模拟末态的计算结果，不落库、不改变任何模拟记录。
"""
from __future__ import annotations

from typing import Optional

from .engine import _clean

# 方程一致性 / 可行性判断的相对容差
SOLVE_EPS = 1e-9


class BlendPlanError(Exception):
    """配比求解失败：携带结构化错误码、可读信息与定位。"""

    def __init__(self, code: str, message: str, location: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.location = location


def _is_close(a: float, b: float, scale: float) -> bool:
    return abs(a - b) <= SOLVE_EPS * max(1.0, scale)


def solve_blend_plan(
    tube_a: dict,
    tube_b: dict,
    target_volume: float,
    target_concentrations: dict[str, float],
) -> dict:
    """求解两路取样量，返回可直接序列化的配比方案。

    ``tube_a`` / ``tube_b`` 为模拟末态管视图（含 ``id`` / ``volume`` /
    ``concentrations``）；``target_concentrations`` 未列出的标记物按 0
    处理。求解失败抛出 :class:`BlendPlanError`。
    """
    volume = target_volume
    markers = sorted(
        set(tube_a["concentrations"])
        | set(tube_b["concentrations"])
        | set(target_concentrations)
    )
    conc_a = {name: tube_a["concentrations"].get(name, 0.0) for name in markers}
    conc_b = {name: tube_b["concentrations"].get(name, 0.0) for name in markers}
    target = {name: target_concentrations.get(name, 0.0) for name in markers}

    # —— 逐标记物联立：c_a·x1 + c_b·x2 = t·V 且 x1 + x2 = V ——
    x1: Optional[float] = None
    x1_from: Optional[str] = None
    for name in markers:
        denom = conc_a[name] - conc_b[name]
        conc_scale = max(abs(conc_a[name]), abs(conc_b[name]))
        if abs(denom) <= SOLVE_EPS * max(1.0, conc_scale):
            # 两管该标记物浓度相同：不提供拆分信息，
            # 但目标必须等于该共同浓度（含两管都不含该标记物、均为 0 的情形）
            if not _is_close(
                target[name], conc_a[name], max(conc_scale, abs(target[name]))
            ):
                raise BlendPlanError(
                    "BLEND_TARGET_INCONSISTENT",
                    f"标记物 {name!r} 在源管 {tube_a['id']!r} 与 {tube_b['id']!r} 中"
                    f"浓度均为 {_clean(conc_a[name])}，任意配比都无法达到目标浓度 "
                    f"{_clean(target[name])}",
                    f"target_concentrations.{name}",
                )
            continue
        candidate = (target[name] - conc_b[name]) * volume / denom
        if x1 is None:
            x1 = candidate
            x1_from = name
        elif not _is_close(candidate, x1, max(abs(candidate), abs(x1), volume)):
            raise BlendPlanError(
                "BLEND_TARGET_INCONSISTENT",
                f"各标记物方程不一致：按标记物 {x1_from!r} 需从 {tube_a['id']!r} "
                f"取样 {_clean(x1)}，按标记物 {name!r} 需取样 {_clean(candidate)}"
                f"（x1 + x2 = target_volume = {_clean(volume)} 下两者必须相等）",
                f"target_concentrations.{name}",
            )

    if x1 is None:
        # 所有标记物两管浓度一致：拆分不唯一，按确定规则优先从第一支源管取
        x1 = min(volume, tube_a["volume"])
    x2 = volume - x1

    # —— 残差复核（浮点兜底）：每一行质量守恒方程都必须成立 ——
    for name in markers:
        residual = conc_a[name] * x1 + conc_b[name] * x2 - target[name] * volume
        scale = max(
            abs(conc_a[name] * x1),
            abs(conc_b[name] * x2),
            abs(target[name] * volume),
        )
        if abs(residual) > SOLVE_EPS * max(1.0, scale):
            raise BlendPlanError(
                "BLEND_TARGET_INCONSISTENT",
                f"标记物 {name!r} 的质量守恒方程无法满足（残差 "
                f"{_clean(residual)}）",
                f"target_concentrations.{name}",
            )

    # —— 可行性：非负 + 不超过两管末态可用体积 ——
    feas_tol = SOLVE_EPS * max(1.0, volume)
    if x1 < -feas_tol or x2 < -feas_tol:
        raise BlendPlanError(
            "BLEND_TARGET_UNREACHABLE",
            f"目标浓度超出源管 {tube_a['id']!r} 与 {tube_b['id']!r} 的可混合"
            f"范围：解得两路取样量 {_clean(x1)} / {_clean(x2)} 含负值",
            "target_concentrations",
        )
    # 容差内的负值噪声先夹取到 0
    x1 = max(x1, 0.0)
    x2 = max(x2, 0.0)

    for index, (tube, take) in enumerate(((tube_a, x1), (tube_b, x2))):
        if take > tube["volume"] + feas_tol:
            raise BlendPlanError(
                "BLEND_INSUFFICIENT_VOLUME",
                f"配比需从 {tube['id']!r} 取样 {_clean(take)}，但该管末态仅剩 "
                f"{_clean(tube['volume'])}",
                f"source_tubes[{index}]",
            )
    # 容差内的超量噪声夹取到可用体积
    x1 = min(x1, tube_a["volume"])
    x2 = min(x2, tube_b["volume"])

    expected = {
        name: _clean((conc_a[name] * x1 + conc_b[name] * x2) / volume)
        for name in markers
    }
    return {
        "aliquots": [
            {"tube_id": tube_a["id"], "volume": _clean(x1)},
            {"tube_id": tube_b["id"], "volume": _clean(x2)},
        ],
        "expected_concentrations": expected,
        "remaining_volumes": {
            tube_a["id"]: _clean(tube_a["volume"] - x1),
            tube_b["id"]: _clean(tube_b["volume"] - x2),
        },
    }
