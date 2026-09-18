"""FastAPI 应用入口与路由。"""
from __future__ import annotations

import sqlite3
from typing import Optional

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Query, status
from fastapi.responses import JSONResponse, RedirectResponse

from . import blend, database, engine
from .models import (
    BlendPlanRequest,
    BlendPlanResult,
    SimulationList,
    SimulationRequest,
    SimulationResult,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时建表（不存在则创建），避免首次请求才暴露路径权限问题
    conn = database.connect()
    conn.close()
    yield


app = FastAPI(
    title="实验室样本混样追踪 API",
    description=(
        "提交初始样本（体积 + 标记物浓度，可选 max_volume 管容量）和按时间"
        "排序的转移操作，服务逐步计算混合浓度与来源占比，并对含移液损耗"
        "（可选 delivered_amount）的系统按“所有管内 + 累计 waste”口径做"
        "守恒校验，每次模拟记录持久化到 SQLite。execution_mode=time_batch 时，"
        "同一时刻的连续转移按批原子执行：吸取均基于批次开始快照，"
        "先按源管汇总校验可用量，再按批末净体积统一校验各管 max_volume，"
        "然后统一写入，结果与数组顺序无关；任一管超容整批不落地。"
        "此外支持从成功模拟的末态反推混样方案"
        "（POST /api/simulations/{sim_id}/blend-plan）：给定目标体积与"
        "目标各标记物浓度，联立所有标记物的质量守恒方程求解两支源管的"
        "两路取样量，并返回预计浓度与取样后剩余体积。"
    ),
    version="1.4.0",
    lifespan=lifespan,
)


def get_db() -> sqlite3.Connection:
    conn = database.connect()
    try:
        yield conn
    finally:
        conn.close()


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["系统"])
def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/api/simulations",
    tags=["模拟"],
    response_model=SimulationResult,
    responses={
        201: {"model": SimulationResult, "description": "模拟成功并已保存"},
        400: {"model": SimulationResult, "description": "业务规则失败（超量 / 超容量 / 时间逆序 / 未知样本）"},
        422: {"description": "请求体格式 / 字段校验失败（FastAPI 标准错误结构）"},
    },
    status_code=status.HTTP_201_CREATED,
    summary="提交一次混样模拟",
)
def create_simulation(
    request: SimulationRequest,
    db: sqlite3.Connection = Depends(get_db),
) -> JSONResponse:
    """执行模拟并落库。

    返回 ``201`` 表示业务模拟成功；当转移超量、注入后超过管容量
    （max_volume）、时间逆序或引用未知样本等业务规则失败时返回 ``400``，
    响应体中 ``status=failed`` 且 ``error`` 给出错误码、可读信息和精确定位
    （如 ``transfers[2].source``、``transfers[i].delivered_amount``）。
    """
    result = engine.run_simulation(request)
    record = database.insert_simulation(db, result)

    http_status = (
        status.HTTP_201_CREATED
        if record["status"] == "success"
        else status.HTTP_400_BAD_REQUEST
    )
    return JSONResponse(content=record, status_code=http_status)


@app.get(
    "/api/simulations/{sim_id}",
    tags=["模拟"],
    response_model=SimulationResult,
    responses={
        400: {"model": SimulationResult, "description": "该次模拟的业务结果为失败"},
        404: {"description": "模拟记录不存在"},
    },
    summary="查询某次模拟的完整结果",
)
def get_simulation(
    sim_id: str, db: sqlite3.Connection = Depends(get_db)
) -> JSONResponse:
    record = database.get_simulation(db, sim_id)
    if record is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "detail": {
                    "code": "SIMULATION_NOT_FOUND",
                    "message": f"未找到 id 为 {sim_id!r} 的模拟记录",
                    "location": f"path.sim_id",
                }
            },
        )
    http_status = (
        status.HTTP_200_OK
        if record["status"] == "success"
        else status.HTTP_400_BAD_REQUEST
    )
    return JSONResponse(content=record, status_code=http_status)


@app.get(
    "/api/simulations",
    tags=["模拟"],
    response_model=SimulationList,
    summary="列出历史模拟记录（按创建时间倒序）",
)
def list_simulations(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    status_filter: Optional[str] = Query(
        default=None,
        alias="status",
        pattern="^(success|failed)$",
        description="按状态过滤：success 或 failed",
    ),
    db: sqlite3.Connection = Depends(get_db),
) -> SimulationList:
    items = database.list_simulations(db, limit=limit, offset=offset)
    if status_filter:
        items = [item for item in items if item["status"] == status_filter]
    return SimulationList(items=items, limit=limit, offset=offset)


@app.post(
    "/api/simulations/{sim_id}/blend-plan",
    tags=["混样规划"],
    response_model=BlendPlanResult,
    responses={
        200: {"model": BlendPlanResult, "description": "求解成功，返回两路取样方案"},
        400: {
            "model": BlendPlanResult,
            "description": "目标无解 / 取样量超可用体积 / 模拟未成功 / 源管不存在",
        },
        404: {"description": "模拟记录不存在"},
        422: {"description": "请求体格式 / 字段校验失败（FastAPI 标准错误结构）"},
    },
    summary="从成功模拟末态反推两源管混样取样方案",
)
def create_blend_plan(
    sim_id: str,
    request: BlendPlanRequest,
    db: sqlite3.Connection = Depends(get_db),
) -> JSONResponse:
    """联立质量守恒方程反推两路取样量（纯计算，不修改模拟记录）。

    从指定**成功**模拟的末态中取 ``sources`` 两支源管，联立体积守恒
    （两路取样量之和 = ``target_volume``）与每种标记物的质量守恒方程求解；
    缺失标记物按 0 处理。成功返回 ``200`` 与两路取样量、预计各标记物浓度、
    取样后剩余体积；目标无解（方程不一致 / 取样量为负 / 超出可用体积）、
    模拟未成功或源管不存在时返回 ``400``，``error`` 给出错误码与定位信息
    （如 ``body.target_concentrations.GFP``、``body.sources[0]``）。
    """
    record = database.get_simulation(db, sim_id)
    if record is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "detail": {
                    "code": "SIMULATION_NOT_FOUND",
                    "message": f"未找到 id 为 {sim_id!r} 的模拟记录",
                    "location": "path.sim_id",
                }
            },
        )

    def _envelope(result: dict) -> dict:
        return {
            "simulation_id": sim_id,
            "target_volume": request.target_volume,
            "target_concentrations": request.target_concentrations,
            **result,
        }

    def _reject(code: str, message: str, location: str) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_envelope(
                {
                    "status": "failed",
                    "aliquots": [],
                    "expected_concentrations": {},
                    "error": {"code": code, "message": message, "location": location},
                }
            ),
        )

    if record["status"] != "success":
        return _reject(
            "SIMULATION_NOT_SUCCESSFUL",
            f"模拟 {sim_id!r} 的状态为 {record['status']!r}，"
            "只有成功的模拟才能用于混样规划",
            "path.sim_id",
        )

    final_states = {tube["id"]: tube for tube in record["final_states"]}
    for index, source in enumerate(request.sources):
        if source not in final_states:
            return _reject(
                "SOURCE_TUBE_NOT_FOUND",
                f"模拟 {sim_id!r} 的末态中不存在管 {source!r}"
                f"（可用管：{sorted(final_states)}）",
                f"body.sources[{index}]",
            )

    result = blend.plan_blend(
        final_states[request.sources[0]],
        final_states[request.sources[1]],
        request.target_volume,
        request.target_concentrations,
    )
    http_status = (
        status.HTTP_200_OK
        if result["status"] == "success"
        else status.HTTP_400_BAD_REQUEST
    )
    return JSONResponse(content=_envelope(result), status_code=http_status)
