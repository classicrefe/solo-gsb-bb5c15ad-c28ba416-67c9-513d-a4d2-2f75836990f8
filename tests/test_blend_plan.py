"""blend-plan 接口与求解器测试：两源管混样反解、各类无解场景与错误定位。"""
import pytest

from app.blend import plan_blend


def create_sim(client, samples, transfers=None, name="blend-src"):
    payload = {"name": name, "samples": samples, "transfers": transfers or []}
    response = client.post("/api/simulations", json=payload)
    assert response.status_code == 201
    return response.json()["id"]


def blend(client, sim_id, payload):
    return client.post(f"/api/simulations/{sim_id}/blend-plan", json=payload)


# ---------- 成功场景 ----------


def test_blend_plan_single_marker_success(client):
    sim_id = create_sim(
        client,
        [
            {"id": "A", "initial_volume": 100.0, "concentration": 10.0},
            {"id": "B", "initial_volume": 100.0, "concentration": 0.0},
        ],
    )
    response = blend(
        client,
        sim_id,
        {
            "sources": ["A", "B"],
            "target_volume": 50.0,
            "target_concentrations": {"default": 4.0},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["simulation_id"] == sim_id
    assert body["error"] is None
    assert body["target_volume"] == 50.0

    aliquots = {a["source"]: a for a in body["aliquots"]}
    assert aliquots["A"]["volume"] == 20.0
    assert aliquots["A"]["remaining_volume"] == 80.0
    assert aliquots["B"]["volume"] == 30.0
    assert aliquots["B"]["remaining_volume"] == 70.0
    assert body["expected_concentrations"] == {"default": 4.0}


def test_blend_plan_multi_marker_success(client):
    # 两管标记物互补：各自缺失的标记物按 0 处理
    sim_id = create_sim(
        client,
        [
            {"id": "S1", "initial_volume": 100.0, "markers": {"GFP": 10.0}},
            {"id": "S2", "initial_volume": 100.0, "markers": {"RFP": 5.0}},
        ],
    )
    response = blend(
        client,
        sim_id,
        {
            "sources": ["S1", "S2"],
            "target_volume": 40.0,
            "target_concentrations": {"GFP": 5.0, "RFP": 2.5},
        },
    )
    assert response.status_code == 200
    body = response.json()
    aliquots = {a["source"]: a for a in body["aliquots"]}
    assert aliquots["S1"]["volume"] == 20.0
    assert aliquots["S1"]["remaining_volume"] == 80.0
    assert aliquots["S2"]["volume"] == 20.0
    assert aliquots["S2"]["remaining_volume"] == 80.0
    assert body["expected_concentrations"] == {"GFP": 5.0, "RFP": 2.5}


def test_blend_plan_uses_final_state(client):
    # 末态：A 50 @ GFP10/RFP0；B 150 @ GFP 10/3、RFP 16/3
    sim_id = create_sim(
        client,
        samples=[
            {"id": "A", "initial_volume": 100.0, "markers": {"GFP": 10.0, "RFP": 0.0}},
            {"id": "B", "initial_volume": 100.0, "markers": {"GFP": 0.0, "RFP": 8.0}},
        ],
        transfers=[{"time": 1, "source": "A", "target": "B", "amount": 50.0}],
    )
    response = blend(
        client,
        sim_id,
        {
            "sources": ["A", "B"],
            "target_volume": 60.0,
            "target_concentrations": {"GFP": 5.0, "RFP": 4.0},
        },
    )
    assert response.status_code == 200
    body = response.json()
    aliquots = {a["source"]: a for a in body["aliquots"]}
    assert aliquots["A"]["volume"] == pytest.approx(15.0)
    assert aliquots["A"]["remaining_volume"] == pytest.approx(35.0)
    assert aliquots["B"]["volume"] == pytest.approx(45.0)
    assert aliquots["B"]["remaining_volume"] == pytest.approx(105.0)
    assert body["expected_concentrations"]["GFP"] == pytest.approx(5.0)
    assert body["expected_concentrations"]["RFP"] == pytest.approx(4.0)


def test_blend_plan_zero_aliquot_allowed(client):
    # 目标浓度恰等于 A 管浓度：只需从 A 取样，B 取 0
    sim_id = create_sim(
        client,
        [
            {"id": "A", "initial_volume": 100.0, "concentration": 10.0},
            {"id": "B", "initial_volume": 100.0, "concentration": 0.0},
        ],
    )
    response = blend(
        client,
        sim_id,
        {
            "sources": ["A", "B"],
            "target_volume": 30.0,
            "target_concentrations": {"default": 10.0},
        },
    )
    assert response.status_code == 200
    aliquots = {a["source"]: a for a in response.json()["aliquots"]}
    assert aliquots["A"]["volume"] == 30.0
    assert aliquots["B"]["volume"] == 0.0
    assert aliquots["B"]["remaining_volume"] == 100.0


def test_blend_plan_missing_marker_in_target_treated_as_zero(client):
    # 目标只列 GFP 且为 0：含 RFP 的 A 管必须取 0，等效全从 B 取
    sim_id = create_sim(
        client,
        [
            {"id": "A", "initial_volume": 100.0, "markers": {"GFP": 10.0, "RFP": 3.0}},
            {"id": "B", "initial_volume": 100.0, "markers": {"GFP": 0.0, "RFP": 0.0}},
        ],
    )
    response = blend(
        client,
        sim_id,
        {
            "sources": ["A", "B"],
            "target_volume": 10.0,
            "target_concentrations": {"GFP": 0.0},
        },
    )
    assert response.status_code == 200
    body = response.json()
    aliquots = {a["source"]: a for a in body["aliquots"]}
    assert aliquots["A"]["volume"] == 0.0
    assert aliquots["B"]["volume"] == 10.0
    assert body["expected_concentrations"] == {"GFP": 0.0, "RFP": 0.0}


def test_blend_plan_does_not_modify_simulation(client):
    sim_id = create_sim(
        client,
        [
            {"id": "A", "initial_volume": 100.0, "concentration": 10.0},
            {"id": "B", "initial_volume": 100.0, "concentration": 0.0},
        ],
    )
    before = client.get(f"/api/simulations/{sim_id}").json()
    response = blend(
        client,
        sim_id,
        {
            "sources": ["A", "B"],
            "target_volume": 50.0,
            "target_concentrations": {"default": 4.0},
        },
    )
    assert response.status_code == 200
    after = client.get(f"/api/simulations/{sim_id}").json()
    assert after["final_states"] == before["final_states"]


# ---------- 目标无解 / 业务失败（400，带定位） ----------


def test_blend_plan_inconsistent_markers_rejected(client):
    # GFP 要求 S1/S2 各取 20，RFP 要求只从 S2 取 8：方程互相矛盾
    sim_id = create_sim(
        client,
        [
            {"id": "S1", "initial_volume": 100.0, "markers": {"GFP": 10.0}},
            {"id": "S2", "initial_volume": 100.0, "markers": {"RFP": 5.0}},
        ],
    )
    response = blend(
        client,
        sim_id,
        {
            "sources": ["S1", "S2"],
            "target_volume": 40.0,
            "target_concentrations": {"GFP": 5.0, "RFP": 1.0},
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "failed"
    assert body["aliquots"] == []
    assert body["expected_concentrations"] == {}
    error = body["error"]
    assert error["code"] == "BLEND_TARGET_INCONSISTENT"
    assert error["location"] == "body.target_concentrations.RFP"
    assert "'RFP'" in error["message"]


def test_blend_plan_unlisted_source_marker_makes_target_infeasible(client):
    # A 管含 RFP 而目标未列出（按 0 处理）：任何正的 A 取样都会带入 RFP
    sim_id = create_sim(
        client,
        [
            {"id": "A", "initial_volume": 100.0, "markers": {"GFP": 10.0, "RFP": 3.0}},
            {"id": "B", "initial_volume": 100.0, "markers": {"GFP": 0.0, "RFP": 0.0}},
        ],
    )
    response = blend(
        client,
        sim_id,
        {
            "sources": ["A", "B"],
            "target_volume": 10.0,
            "target_concentrations": {"GFP": 5.0},
        },
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "BLEND_TARGET_INCONSISTENT"
    # 目标中不存在 RFP 字段，定位退化为整个 target_concentrations
    assert error["location"] == "body.target_concentrations"
    assert "'RFP'" in error["message"]


def test_blend_plan_target_marker_absent_from_both_tubes(client):
    sim_id = create_sim(
        client,
        [
            {"id": "A", "initial_volume": 100.0, "concentration": 10.0},
            {"id": "B", "initial_volume": 100.0, "concentration": 0.0},
        ],
    )
    response = blend(
        client,
        sim_id,
        {
            "sources": ["A", "B"],
            "target_volume": 10.0,
            "target_concentrations": {"GFP": 1.0},
        },
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "BLEND_TARGET_INCONSISTENT"
    assert error["location"] == "body.target_concentrations.GFP"


def test_blend_plan_negative_aliquot_rejected(client):
    # 目标浓度 12 超出两管浓度区间 [0, 10]：解出 B 取样为负
    sim_id = create_sim(
        client,
        [
            {"id": "A", "initial_volume": 100.0, "concentration": 10.0},
            {"id": "B", "initial_volume": 100.0, "concentration": 0.0},
        ],
    )
    response = blend(
        client,
        sim_id,
        {
            "sources": ["A", "B"],
            "target_volume": 50.0,
            "target_concentrations": {"default": 12.0},
        },
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "BLEND_NEGATIVE_ALIQUOT"
    assert error["location"] == "body.target_concentrations.default"
    assert "'B'" in error["message"]


def test_blend_plan_exceeds_available_volume(client):
    # 需要 A 提供 120，但末态仅有 100
    sim_id = create_sim(
        client,
        [
            {"id": "A", "initial_volume": 100.0, "concentration": 10.0},
            {"id": "B", "initial_volume": 100.0, "concentration": 0.0},
        ],
    )
    response = blend(
        client,
        sim_id,
        {
            "sources": ["A", "B"],
            "target_volume": 150.0,
            "target_concentrations": {"default": 8.0},
        },
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "BLEND_INSUFFICIENT_VOLUME"
    assert error["location"] == "body.sources[0]"
    assert "'A'" in error["message"]


def test_blend_plan_underdetermined_when_tubes_identical(client):
    sim_id = create_sim(
        client,
        [
            {"id": "A", "initial_volume": 100.0, "concentration": 5.0},
            {"id": "B", "initial_volume": 100.0, "concentration": 5.0},
        ],
    )
    response = blend(
        client,
        sim_id,
        {
            "sources": ["A", "B"],
            "target_volume": 40.0,
            "target_concentrations": {"default": 5.0},
        },
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "BLEND_UNDERDETERMINED"
    assert error["location"] == "body.sources"


def test_blend_plan_degenerate_but_inconsistent(client):
    # 两管浓度相同（均为 5），目标 6 怎么配都达不到
    sim_id = create_sim(
        client,
        [
            {"id": "A", "initial_volume": 100.0, "concentration": 5.0},
            {"id": "B", "initial_volume": 100.0, "concentration": 5.0},
        ],
    )
    response = blend(
        client,
        sim_id,
        {
            "sources": ["A", "B"],
            "target_volume": 40.0,
            "target_concentrations": {"default": 6.0},
        },
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "BLEND_TARGET_INCONSISTENT"
    assert error["location"] == "body.target_concentrations.default"


# ---------- 模拟 / 源管定位错误 ----------


def test_blend_plan_simulation_not_found(client):
    response = blend(
        client,
        "does-not-exist",
        {
            "sources": ["A", "B"],
            "target_volume": 10.0,
            "target_concentrations": {"default": 1.0},
        },
    )
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["code"] == "SIMULATION_NOT_FOUND"
    assert detail["location"] == "path.sim_id"


def test_blend_plan_requires_successful_simulation(client):
    failed_payload = {
        "samples": [
            {"id": "A", "initial_volume": 5.0, "concentration": 1.0},
            {"id": "B", "initial_volume": 5.0, "concentration": 0.0},
        ],
        "transfers": [{"time": 1, "source": "A", "target": "B", "amount": 100.0}],
    }
    created = client.post("/api/simulations", json=failed_payload)
    assert created.status_code == 400
    sim_id = created.json()["id"]

    response = blend(
        client,
        sim_id,
        {
            "sources": ["A", "B"],
            "target_volume": 1.0,
            "target_concentrations": {"default": 0.5},
        },
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "SIMULATION_NOT_SUCCESSFUL"
    assert error["location"] == "path.sim_id"


def test_blend_plan_unknown_source_tube(client):
    sim_id = create_sim(
        client,
        [
            {"id": "A", "initial_volume": 100.0, "concentration": 10.0},
            {"id": "B", "initial_volume": 100.0, "concentration": 0.0},
        ],
    )
    response = blend(
        client,
        sim_id,
        {
            "sources": ["A", "Z"],
            "target_volume": 10.0,
            "target_concentrations": {"default": 5.0},
        },
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "SOURCE_TUBE_NOT_FOUND"
    assert error["location"] == "body.sources[1]"
    assert "'Z'" in error["message"]


# ---------- 请求体校验（422） ----------


def test_blend_plan_same_source_rejected_422(client):
    response = blend(
        client,
        "any-id",
        {
            "sources": ["A", "A"],
            "target_volume": 10.0,
            "target_concentrations": {"default": 1.0},
        },
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"sources": ["A", "B"], "target_volume": 0, "target_concentrations": {"default": 1.0}},
        {"sources": ["A", "B"], "target_volume": -5, "target_concentrations": {"default": 1.0}},
        {"sources": ["A", "B"], "target_volume": 10.0, "target_concentrations": {}},
        {"sources": ["A", "B"], "target_volume": 10.0, "target_concentrations": {"GFP": -1.0}},
        {"sources": ["A"], "target_volume": 10.0, "target_concentrations": {"default": 1.0}},
        {"sources": ["A", "B", "C"], "target_volume": 10.0, "target_concentrations": {"default": 1.0}},
        {
            "sources": ["A", "B"],
            "target_volume": 10.0,
            "target_concentrations": {"default": 1.0},
            "unexpected": 1,
        },
    ],
)
def test_blend_plan_request_validation_422(client, payload):
    response = blend(client, "any-id", payload)
    assert response.status_code == 422


def test_openapi_documents_blend_plan(client):
    schema = client.get("/openapi.json").json()
    assert "/api/simulations/{sim_id}/blend-plan" in schema["paths"]
    request_props = schema["components"]["schemas"]["BlendPlanRequest"]["properties"]
    assert {"sources", "target_volume", "target_concentrations"} <= set(request_props)
    result_props = schema["components"]["schemas"]["BlendPlanResult"]["properties"]
    assert {"aliquots", "expected_concentrations", "error"} <= set(result_props)


# ---------- 求解器单元测试 ----------


def test_plan_blend_direct_success():
    tube_a = {"id": "A", "volume": 100.0, "concentrations": {"default": 10.0}}
    tube_b = {"id": "B", "volume": 100.0, "concentrations": {"default": 0.0}}
    result = plan_blend(tube_a, tube_b, 50.0, {"default": 4.0})
    assert result["status"] == "success"
    assert result["aliquots"][0] == {
        "source": "A",
        "volume": 20.0,
        "remaining_volume": 80.0,
    }
    assert result["aliquots"][1]["volume"] == 30.0
    assert result["expected_concentrations"] == {"default": 4.0}


def test_plan_blend_direct_underdetermined():
    tube_a = {"id": "A", "volume": 10.0, "concentrations": {"default": 5.0}}
    tube_b = {"id": "B", "volume": 10.0, "concentrations": {"default": 5.0}}
    result = plan_blend(tube_a, tube_b, 10.0, {"default": 5.0})
    assert result["status"] == "failed"
    assert result["error"]["code"] == "BLEND_UNDERDETERMINED"


def test_plan_blend_direct_empty_source_insufficient():
    # A 已抽空（末态体积 0）：解出需要 A 提供 25，判定超可用体积
    tube_a = {"id": "A", "volume": 0.0, "concentrations": {"default": 0.0}}
    tube_b = {"id": "B", "volume": 100.0, "concentrations": {"default": 4.0}}
    result = plan_blend(tube_a, tube_b, 50.0, {"default": 2.0})
    assert result["status"] == "failed"
    assert result["error"]["code"] == "BLEND_INSUFFICIENT_VOLUME"
    assert result["error"]["location"] == "body.sources[0]"
