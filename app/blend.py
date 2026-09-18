"""两源管混样反解（blend plan）。

给定某次**成功**模拟末态中的两支源管、目标总体积 ``target_volume`` 与目标
各标记物浓度，联立以下方程反推两路取样量 ``v_a`` / ``v_b``：

* 体积守恒：``v_a + v_b = target_volume``；
* 每种标记物质量守恒：``c_a[m] * v_a + c_b[m] * v_b = t[m] * target_volume``
  （源管或目标中缺失的标记物浓度一律按 0 处理；目标未列出的标记物同样
  视为目标 0，即源管里存在的任何标记物都必须能被两路取样解释）。

两个未知数、1 + N 条方程：取两管浓度**相对差最大**的标记物方程与体积方程
联立求解（数值上最能把两路贡献区分开），再回代校验其余所有标记物方程；
任一方程残差超差即判定目标自相矛盾（``BLEND_TARGET_INCONSISTENT``）。
所有标记物在两管中浓度（近似）相同时方程组退化，任意取样比例都满足目标，
无法唯一确定取样量（``BLEND_UNDERDETERMINED``）。

求解后依次校验：取样量非负（``BLEND_NEGATIVE_ALIQUOT``，即目标浓度落在
两管浓度范围之外）、不超过源管末态可用体积（``BLEND_INSUFFICIENT_VOLUME``）。

本模块只做纯计算：不读取 / 修改数据库，也不改变已保存的模拟记录。
"""
from __future__ import annotations

from .engine import ZERO_EPS, _clean

# 方程一致性 / 退化判断的相对容差
REL_TOL = 1e-9


def _failed(code: str, message: str, location: str) -> dict:
    return {
        "status": "failed",
        "aliquots": [],
        "expected_concentrations": {},
        "error": {"code": code, "message": message, "location": location},
    }


def _marker_location(name: str, target_concentrations: dict[str, float]) -> str:
    """定位到具体目标浓度字段；目标未列出该标记物时退化为整个对象。"""
    if name in target_concentrations:
        return f"body.target_concentrations.{name}"
    return "body.target_concentrations"


def _relative_span(ca: float, cb: float) -> float:
    """两管某标记物浓度的相对差（0 表示两管浓度不可区分）。"""
    denom = abs(ca) + abs(cb)
    if denom <= 0.0:
        return 0.0
    return abs(ca - cb) / denom


def plan_blend(
    tube_a: dict,
    tube_b: dict,
    target_volume: float,
    target_concentrations: dict[str, float],
) -> dict:
    """求解两路取样量，返回可直接序列化的结果字典（含 status / error）。

    ``tube_a`` / ``tube_b`` 为模拟末态视图（含 ``id`` / ``volume`` /
    ``concentrations``）；``aliquots`` 顺序与传入的源管顺序一致。
    """
    conc_a: dict[str, float] = tube_a["concentrations"]
    conc_b: dict[str, float] = tube_b["concentrations"]
    all_markers = sorted(set(conc_a) | set(conc_b) | set(target_concentrations))

    # 每个标记物一条质量守恒方程：(名称, c_a, c_b, 目标质量)
    equations = [
        (
            name,
            conc_a.get(name, 0.0),
            conc_b.get(name, 0.0),
            target_concentrations.get(name, 0.0) * target_volume,
        )
        for name in all_markers
    ]

    # 主方程：两管浓度相对差最大的标记物
    pivot = max(equations, key=lambda item: _relative_span(item[1], item[2]))
    pivot_name, pivot_ca, pivot_cb, pivot_rhs = pivot

    if _relative_span(pivot_ca, pivot_cb) <= REL_TOL:
        # 所有标记物在两管中的浓度（近似）相同，方程组退化：
        # 先确认各目标浓度与源管浓度一致，否则目标本身矛盾
        for name, ca, _cb, rhs in equations:
            implied = ca * target_volume
            if abs(implied - rhs) > REL_TOL * max(abs(implied), abs(rhs)) + ZERO_EPS:
                return _failed(
                    "BLEND_TARGET_INCONSISTENT",
                    f"标记物 {name!r} 的目标浓度 {_clean(rhs / target_volume)} 与两支"
                    f"源管矛盾：两管该标记物浓度均约为 {_clean(ca)}，任何取样比例"
                    f"都只能配出 {_clean(ca)}",
                    _marker_location(name, target_concentrations),
                )
        return _failed(
            "BLEND_UNDERDETERMINED",
            "两支源管的所有标记物浓度都相同，任意取样比例均满足目标，"
            "无法唯一确定两路取样量",
            "body.sources",
        )

    # 与体积方程联立：v_a + v_b = V 且 c_a*v_a + c_b*v_b = rhs
    v_a = (pivot_rhs - pivot_cb * target_volume) / (pivot_ca - pivot_cb)
    v_b = target_volume - v_a

    # 回代校验所有标记物方程（含目标未列出、按 0 处理的标记物）
    for name, ca, cb, rhs in equations:
        implied = ca * v_a + cb * v_b
        if abs(implied - rhs) <= REL_TOL * max(abs(implied), abs(rhs)) + ZERO_EPS:
            continue
        return _failed(
            "BLEND_TARGET_INCONSISTENT",
            f"按标记物 {pivot_name!r} 解得两路取样量后，标记物 {name!r} 的质量"
            f"守恒不成立：混合后质量为 {_clean(implied)}，目标要求 {_clean(rhs)}"
            f"（浓度 {_clean(implied / target_volume)} ≠ "
            f"{_clean(rhs / target_volume)}）",
            _marker_location(name, target_concentrations),
        )

    # 取样量非负：目标浓度必须落在两管浓度之间
    neg_tol = REL_TOL * target_volume + ZERO_EPS
    if v_a < -neg_tol or v_b < -neg_tol:
        bad_tube = tube_a["id"] if v_a < -neg_tol else tube_b["id"]
        return _failed(
            "BLEND_NEGATIVE_ALIQUOT",
            f"按标记物 {pivot_name!r} 的目标浓度 "
            f"{_clean(pivot_rhs / target_volume)} 求解需要从管 {bad_tube!r} 取样 "
            f"{_clean(min(v_a, v_b))}（负值）：目标浓度不在两管浓度区间 "
            f"[{_clean(min(pivot_ca, pivot_cb))}, {_clean(max(pivot_ca, pivot_cb))}] 内",
            _marker_location(pivot_name, target_concentrations),
        )
    # 浮点噪声级别的负值按 0 处理
    if v_a < 0.0:
        v_a, v_b = 0.0, target_volume
    elif v_b < 0.0:
        v_a, v_b = target_volume, 0.0

    # 不超过源管末态可用体积
    for index, (tube, volume) in enumerate(((tube_a, v_a), (tube_b, v_b))):
        available = tube["volume"]
        if volume > available + REL_TOL * max(available, target_volume) + ZERO_EPS:
            return _failed(
                "BLEND_INSUFFICIENT_VOLUME",
                f"需要从管 {tube['id']!r} 取样 {_clean(volume)}，"
                f"但该管在模拟末态仅有 {_clean(available)}",
                f"body.sources[{index}]",
            )

    expected = {
        name: _clean((ca * v_a + cb * v_b) / target_volume)
        for name, ca, cb, _rhs in equations
    }
    aliquots = [
        {
            "source": tube_a["id"],
            "volume": _clean(v_a),
            "remaining_volume": _clean(max(tube_a["volume"] - v_a, 0.0)),
        },
        {
            "source": tube_b["id"],
            "volume": _clean(v_b),
            "remaining_volume": _clean(max(tube_b["volume"] - v_b, 0.0)),
        },
    ]
    return {
        "status": "success",
        "aliquots": aliquots,
        "expected_concentrations": expected,
        "error": None,
    }
