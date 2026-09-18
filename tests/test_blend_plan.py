"""POST /api/simulations/{sim_id}/blend-plan 接口测试。

覆盖：成功求解（含缺失标记物按 0、拆分不唯一的确定规则）、结构化错误
（模拟 / 源管不存在、模拟未成功、方程不一致、负取样量、余量不足）、
请求校验 422、OpenAPI 文档与“不改变已保存模拟”。
"""
import pytest


def _create_simulation(client, payload):
    response = client.post("/api/simulations", json=payload)
    assert response.status_code == 201
    return response.json()["id"]


@pytest.fixture()
def mixed_sim_id(client):
    """S1 末态 80 @ GFP10；S2 末态 100 @ GFP2 / RFP3.2。"""
    return _create_simulation(
        client,
        {
            "name": "blend-plan-末态",
            "samples": [
                {"id": "S1", "initial_volume": 100.0, "markers": {"GFP": 10.0}},
                {"id": "S2", "initial_volume": 80.0, "markers": {"RFP": 4.0}},
            ],
            "transfers": [
                {"time": 1, "source": "S1", "target": "S2", "amount": 20.0},
            ],
        },
    )


def test_blend_plan_success(client, mixed_sim_id):
    response = client.post(
        f"/api/simulations/{mixed_sim_id}/blend-plan",
        json={
            "source_tubes": ["S1", "S2"],
            "target_volume": 20.0,
            "target_concentrations": {"GFP": 6.0, "RFP": 1.6},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["simulation_id"] == mixed_sim_id
    assert body["source_tubes"] == ["S1", "S2"]
    assert body["target_volume"] == 20.0
    assert body["target_concentrations"] == {"GFP": 6.0, "RFP": 1.6}
    assert body["aliquots"] == [
        {"tube_id": "S1", "volume": 10.0},
        {"tube_id": "S2", "volume": 10.0},
    ]
    # 预计浓度覆盖全部标记物并与目标一致
    assert body["expected_concentrations"] == {"GFP": 6.0, "RFP": 1.6}
    # 取样后剩余体积（基于末态 80 / 100）
    assert body["remaining_volumes"] == {"S1": 70.0, "S2": 90.0}


def test_blend_plan_does_not_modify_simulation(client, mixed_sim_id):
    before = client.get(f"/api/simulations/{mixed_sim_id}").json()
    client.post(
        f"/api/simulations/{mixed_sim_id}/blend-plan",
        json={
            "source_tubes": ["S1", "S2"],
            "target_volume": 20.0,
            "target_concentrations": {"GFP": 6.0, "RFP": 1.6},
        },
    )
    after = client.get(f"/api/simulations/{mixed_sim_id}").json()
    assert after == before


def test_blend_plan_missing_marker_treated_as_zero(client):
    # 目标未列出 RFP（按 0 处理）：只能全部从不含 RFP 的 S1 取
    sim_id = _create_simulation(
        client,
        {
            "samples": [
                {"id": "S1", "initial_volume": 50.0, "markers": {"GFP": 10.0}},
                {
                    "id": "S2",
                    "initial_volume": 60.0,
                    "markers": {"GFP": 4.0, "RFP": 5.0},
                },
            ],
            "transfers": [],
        },
    )
    response = client.post(
        f"/api/simulations/{sim_id}/blend-plan",
        json={
            "source_tubes": ["S1", "S2"],
            "target_volume": 20.0,
            "target_concentrations": {"GFP": 10.0},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["aliquots"] == [
        {"tube_id": "S1", "volume": 20.0},
        {"tube_id": "S2", "volume": 0.0},
    ]
    assert body["expected_concentrations"] == {"GFP": 10.0, "RFP": 0.0}
    assert body["remaining_volumes"] == {"S1": 30.0, "S2": 60.0}


def test_blend_plan_implicit_zero_marker_inconsistent(client, mixed_sim_id):
    # 目标只给 GFP=5：GFP 方程要求 x1=7.5，RFP 隐式为 0 要求 x1=20 → 不一致
    response = client.post(
        f"/api/simulations/{mixed_sim_id}/blend-plan",
        json={
            "source_tubes": ["S1", "S2"],
            "target_volume": 20.0,
            "target_concentrations": {"GFP": 5.0},
        },
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "BLEND_TARGET_INCONSISTENT"
    assert detail["location"] == "target_concentrations.RFP"
    assert "RFP" in detail["message"]


def test_blend_plan_inconsistent_markers_rejected(client, mixed_sim_id):
    # GFP 方程要求 x1=10，RFP 方程要求 x1=7.5 → 拒绝
    response = client.post(
        f"/api/simulations/{mixed_sim_id}/blend-plan",
        json={
            "source_tubes": ["S1", "S2"],
            "target_volume": 20.0,
            "target_concentrations": {"GFP": 6.0, "RFP": 2.0},
        },
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "BLEND_TARGET_INCONSISTENT"
    assert detail["location"] == "target_concentrations.RFP"
    assert "GFP" in detail["message"] and "RFP" in detail["message"]


def test_blend_plan_target_marker_absent_from_both_tubes(client):
    # 两管都不含 RFP（浓度均为 0），目标却要求 RFP>0 → 无解
    sim_id = _create_simulation(
        client,
        {
            "samples": [
                {"id": "S1", "initial_volume": 50.0, "markers": {"GFP": 10.0}},
                {"id": "S2", "initial_volume": 50.0, "markers": {"GFP": 4.0}},
            ],
            "transfers": [],
        },
    )
    response = client.post(
        f"/api/simulations/{sim_id}/blend-plan",
        json={
            "source_tubes": ["S1", "S2"],
            "target_volume": 20.0,
            "target_concentrations": {"GFP": 7.0, "RFP": 1.0},
        },
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "BLEND_TARGET_INCONSISTENT"
    assert detail["location"] == "target_concentrations.RFP"


def test_blend_plan_unreachable_target_negative_aliquot(client):
    # 目标浓度 12 高于两源管（10 / 4）：数学解含负取样量
    sim_id = _create_simulation(
        client,
        {
            "samples": [
                {"id": "S1", "initial_volume": 50.0, "markers": {"GFP": 10.0}},
                {"id": "S2", "initial_volume": 50.0, "markers": {"GFP": 4.0}},
            ],
            "transfers": [],
        },
    )
    response = client.post(
        f"/api/simulations/{sim_id}/blend-plan",
        json={
            "source_tubes": ["S1", "S2"],
            "target_volume": 20.0,
            "target_concentrations": {"GFP": 12.0},
        },
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "BLEND_TARGET_UNREACHABLE"
    assert detail["location"] == "target_concentrations"


def test_blend_plan_insufficient_volume(client):
    # 配比需从 S1 取 20，但 S1 末态仅剩 15
    sim_id = _create_simulation(
        client,
        {
            "samples": [
                {"id": "S1", "initial_volume": 15.0, "markers": {"GFP": 10.0}},
                {"id": "S2", "initial_volume": 100.0, "markers": {"GFP": 0.0}},
            ],
            "transfers": [],
        },
    )
    response = client.post(
        f"/api/simulations/{sim_id}/blend-plan",
        json={
            "source_tubes": ["S1", "S2"],
            "target_volume": 40.0,
            "target_concentrations": {"GFP": 5.0},
        },
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "BLEND_INSUFFICIENT_VOLUME"
    assert detail["location"] == "source_tubes[0]"
    assert "'S1'" in detail["message"]


def test_blend_plan_underdetermined_deterministic_split(client):
    # 两管浓度完全一致：拆分不唯一，按“优先从第一支源管取”给出确定解
    sim_id = _create_simulation(
        client,
        {
            "samples": [
                {"id": "S1", "initial_volume": 30.0, "markers": {"GFP": 5.0}},
                {"id": "S2", "initial_volume": 10.0, "markers": {"GFP": 5.0}},
            ],
            "transfers": [],
        },
    )
    response = client.post(
        f"/api/simulations/{sim_id}/blend-plan",
        json={
            "source_tubes": ["S1", "S2"],
            "target_volume": 35.0,
            "target_concentrations": {"GFP": 5.0},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["aliquots"] == [
        {"tube_id": "S1", "volume": 30.0},
        {"tube_id": "S2", "volume": 5.0},
    ]
    assert body["expected_concentrations"] == {"GFP": 5.0}
    assert body["remaining_volumes"] == {"S1": 0.0, "S2": 5.0}


def test_blend_plan_simulation_not_found(client):
    response = client.post(
        "/api/simulations/does-not-exist/blend-plan",
        json={
            "source_tubes": ["S1", "S2"],
            "target_volume": 10.0,
            "target_concentrations": {"GFP": 1.0},
        },
    )
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["code"] == "SIMULATION_NOT_FOUND"
    assert detail["location"] == "path.sim_id"


def test_blend_plan_failed_simulation_rejected(client):
    failed = client.post(
        "/api/simulations",
        json={
            "samples": [
                {"id": "S1", "initial_volume": 5.0, "concentration": 1.0},
                {"id": "S2", "initial_volume": 5.0, "concentration": 0.0},
            ],
            "transfers": [
                {"time": 1, "source": "S1", "target": "S2", "amount": 100.0},
            ],
        },
    )
    assert failed.status_code == 400
    sim_id = failed.json()["id"]

    response = client.post(
        f"/api/simulations/{sim_id}/blend-plan",
        json={
            "source_tubes": ["S1", "S2"],
            "target_volume": 2.0,
            "target_concentrations": {"default": 0.5},
        },
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "SIMULATION_NOT_SUCCESSFUL"
    assert detail["location"] == "path.sim_id"


def test_blend_plan_unknown_source_tube(client, mixed_sim_id):
    response = client.post(
        f"/api/simulations/{mixed_sim_id}/blend-plan",
        json={
            "source_tubes": ["S1", "S9"],
            "target_volume": 10.0,
            "target_concentrations": {"GFP": 5.0},
        },
    )
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["code"] == "SOURCE_TUBE_NOT_FOUND"
    assert detail["location"] == "source_tubes[1]"
    assert "'S9'" in detail["message"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_tubes": ["S1", "S1"]},               # 两支源管相同
        {"source_tubes": ["S1"]},                     # 数量不足
        {"source_tubes": ["S1", "S2", "S3"]},         # 数量超出
        {"source_tubes": ["S1", "  "]},               # 空字符串 id
        {"target_volume": 0},                         # 目标体积非正
        {"target_volume": -5},
        {"target_concentrations": {}},                # 至少一种标记物
        {"target_concentrations": {"GFP": -1.0}},     # 目标浓度为负
        {"target_concentrations": {"": 1.0}},         # 空标记物名
    ],
)
def test_blend_plan_validation_422(client, mixed_sim_id, overrides):
    request = {
        "source_tubes": ["S1", "S2"],
        "target_volume": 20.0,
        "target_concentrations": {"GFP": 6.0, "RFP": 1.6},
    }
    request.update(overrides)
    response = client.post(
        f"/api/simulations/{mixed_sim_id}/blend-plan", json=request
    )
    assert response.status_code == 422


def test_blend_plan_extra_field_rejected(client, mixed_sim_id):
    response = client.post(
        f"/api/simulations/{mixed_sim_id}/blend-plan",
        json={
            "source_tubes": ["S1", "S2"],
            "target_volume": 20.0,
            "target_concentrations": {"GFP": 6.0},
            "delivered_amount": 1.0,
        },
    )
    assert response.status_code == 422


def test_openapi_documents_blend_plan(client):
    schema = client.get("/openapi.json").json()
    path = schema["paths"]["/api/simulations/{sim_id}/blend-plan"]["post"]
    assert path["responses"]["200"]["description"]
    request_props = schema["components"]["schemas"]["BlendPlanRequest"][
        "properties"
    ]
    assert {"source_tubes", "target_volume", "target_concentrations"} <= set(
        request_props
    )
    result_props = schema["components"]["schemas"]["BlendPlanResult"][
        "properties"
    ]
    assert {
        "aliquots",
        "expected_concentrations",
        "remaining_volumes",
    } <= set(result_props)
