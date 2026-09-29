"""HTTP 接口测试。

全部通过 ``TestClient`` 走真实的应用生命周期（lifespan 里装配依赖），
但**不联网、不依赖任何 Key** —— 这正是"零配置可跑"这个设计目标
在测试层面的体现：CI 上不需要配任何 secret。

接口里最值得守住的三点：
  1. ``/api/config`` 绝不能回显 API Key 原文；
  2. 上传接口的边界（空文件、超长、不支持的格式）必须给出可操作的中文提示；
  3. ``/api/recompute`` 只重算不重解析 —— 它的毫秒级响应是前端拖滑条的体验基础。
"""

from __future__ import annotations

import json

import pytest

from .helpers import SIMPLE_JD, SIMPLE_RESUME

RESUME_TEXT = SIMPLE_RESUME


# ===========================================================================
# 元信息
# ===========================================================================


class TestMeta:
    def test_health(self, client) -> None:
        data = client.get("/api/health").json()
        assert data["status"] == "ok"
        assert data["version"]
        assert data["llm_enabled"] is False      # 测试环境强制无 Key
        assert data["skill_count"] > 50

    def test_config_shape(self, client) -> None:
        data = client.get("/api/config").json()
        assert data["llm_enabled"] is False
        assert data["engine"] == "rule"
        assert data["skill_count"] > 50
        assert len(data["dimensions"]) == 6
        assert len(data["default_weights"]) == 6
        assert data["seniority_options"]
        assert data["max_upload_mb"] >= 1

    def test_config_never_leaks_api_key(self, client) -> None:
        """**安全底线**：无论配没配 Key，接口都不能把它吐出来。"""
        raw = client.get("/api/config").text
        assert "llm_api_key" not in raw
        assert "api_key" not in json.loads(raw)

    def test_config_dimensions_have_ui_metadata(self, client) -> None:
        """前端要拿 label / description 渲染滑条与说明气泡，缺一个就会显示空白。"""
        for dim in client.get("/api/config").json()["dimensions"]:
            assert dim["key"] and dim["label"] and dim["description"]
            assert 0 <= dim["default_weight"] <= 1

    def test_stats_starts_empty(self, client) -> None:
        data = client.get("/api/stats").json()
        assert data["matches_run"] == 0
        assert data["resumes_stored"] == 0
        assert data["llm_calls"] == 0
        assert data["skills_in_dict"] > 50

    def test_stats_counts_matches(self, client) -> None:
        before = client.get("/api/stats").json()["matches_run"]
        client.post("/api/match", json={"jd_text": SIMPLE_JD, "resume_text": RESUME_TEXT})
        assert client.get("/api/stats").json()["matches_run"] == before + 1

    def test_index_page_served(self, client) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "匹配" in resp.text
        assert "<script" in resp.text

    def test_favicon_returns_no_content(self, client) -> None:
        assert client.get("/favicon.ico").status_code == 204

    def test_openapi_schema_lists_core_endpoints(self, client) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        for path in ("/api/match", "/api/match/stream", "/api/recompute", "/api/resumes"):
            assert path in paths


class TestSkillDictionary:
    def test_list_all(self, client) -> None:
        data = client.get("/api/skills").json()
        assert data["total"] > 50
        assert data["items"]
        assert data["categories"]

    def test_search_by_alias(self, client) -> None:
        """用户输入 js 应当能看到它指向 JavaScript —— 这是"可解释"的一部分。"""
        data = client.get("/api/skills", params={"q": "js"}).json()
        canonicals = {item["canonical"] for item in data["items"]}
        assert "JavaScript" in canonicals

    def test_search_k8s(self, client) -> None:
        data = client.get("/api/skills", params={"q": "k8s"}).json()
        assert "Kubernetes" in {item["canonical"] for item in data["items"]}

    def test_search_is_case_insensitive(self, client) -> None:
        lower = client.get("/api/skills", params={"q": "python"}).json()["matched"]
        upper = client.get("/api/skills", params={"q": "PYTHON"}).json()["matched"]
        assert lower == upper > 0

    def test_search_no_result(self, client) -> None:
        data = client.get("/api/skills", params={"q": "绝对不存在的技能名"}).json()
        assert data["items"] == []

    def test_filter_by_category(self, client) -> None:
        data = client.get("/api/skills", params={"category": "编程语言"}).json()
        assert data["items"]
        assert all(item["category"] == "编程语言" for item in data["items"])

    def test_limit_is_clamped(self, client) -> None:
        assert len(client.get("/api/skills", params={"limit": 0}).json()["items"]) == 1
        assert len(client.get("/api/skills", params={"limit": 99999}).json()["items"]) <= 1000

    def test_item_has_aliases(self, client) -> None:
        data = client.get("/api/skills", params={"q": "Python"}).json()
        item = next(i for i in data["items"] if i["canonical"] == "Python")
        assert item["aliases"]
        assert item["alias_count"] == len(item["aliases"])


class TestExamples:
    def test_list(self, client) -> None:
        items = client.get("/api/examples").json()
        keys = {item["key"] for item in items}
        assert {"jd_ai_app_engineer", "resume_zhangsan"} <= keys
        assert all(item["kind"] in {"jd", "resume"} for item in items)

    def test_detail(self, client) -> None:
        data = client.get("/api/examples/jd_ai_app_engineer").json()
        assert data["kind"] == "jd"
        assert len(data["text"]) > 100

    def test_unknown_key_404(self, client) -> None:
        assert client.get("/api/examples/不存在").status_code == 404

    def test_all_listed_examples_are_readable(self, client) -> None:
        """目录里列出的每个示例都必须真的读得到 —— 否则前端"一键填充"会静默失败。"""
        for item in client.get("/api/examples").json():
            resp = client.get(f"/api/examples/{item['key']}")
            assert resp.status_code == 200, item["key"]
            assert resp.json()["text"].strip()


# ===========================================================================
# 简历管理
# ===========================================================================


class TestResumeCrud:
    def test_paste_then_list_then_get_then_delete(self, client) -> None:
        created = client.post("/api/resumes", data={"text": RESUME_TEXT, "name": "王五"}).json()
        assert created["resume_id"]
        assert created["name"] == "王五"
        assert created["char_count"] > 0
        assert created["preview"]

        listed = client.get("/api/resumes").json()
        assert [item["resume_id"] for item in listed] == [created["resume_id"]]

        detail = client.get(f"/api/resumes/{created['resume_id']}").json()
        assert detail["text"] == RESUME_TEXT.strip()

        assert client.delete(f"/api/resumes/{created['resume_id']}").json()["deleted"] is True
        assert client.get("/api/resumes").json() == []

    def test_name_guessed_when_not_given(self, client) -> None:
        created = client.post("/api/resumes", data={"text": RESUME_TEXT}).json()
        assert created["name"] == "王五"

    def test_json_body_supported(self, client) -> None:
        """脚本调用走 JSON 更方便，接口应当兼容。"""
        created = client.post("/api/resumes", json={"text": RESUME_TEXT, "name": "脚本上传"}).json()
        assert created["name"] == "脚本上传"

    def test_upload_text_file(self, client) -> None:
        created = client.post(
            "/api/resumes",
            files={"file": ("resume.txt", RESUME_TEXT.encode("utf-8"), "text/plain")},
        ).json()
        assert created["char_count"] > 0
        # 没显式给名字时，用文件名兜底
        assert created["name"] == "resume"

    def test_upload_gbk_file(self, client) -> None:
        """国内用户用记事本存的 txt 经常是 GBK，必须能识别。"""
        created = client.post(
            "/api/resumes",
            files={"file": ("gbk.txt", RESUME_TEXT.encode("gbk"), "text/plain")},
        ).json()
        assert created["char_count"] > 0

    def test_upload_utf8_bom_file(self, client) -> None:
        """带 BOM 的 UTF-8 若按普通 utf-8 解码，会在首字段前留下 \\ufeff，
        进而让第一个字段永远匹配不到（在项目 1 里踩过这个坑）。"""
        created = client.post(
            "/api/resumes",
            files={"file": ("bom.txt", RESUME_TEXT.encode("utf-8-sig"), "text/plain")},
        ).json()
        assert created["char_count"] == len(RESUME_TEXT.strip())

    def test_upload_empty_file_rejected(self, client) -> None:
        resp = client.post("/api/resumes", files={"file": ("empty.txt", b"", "text/plain")})
        assert resp.status_code == 400
        assert "空" in resp.json()["detail"]

    def test_upload_unsupported_format_rejected(self, client) -> None:
        resp = client.post(
            "/api/resumes", files={"file": ("resume.exe", b"binary", "application/octet-stream")}
        )
        assert resp.status_code == 400
        # 提示必须告诉用户支持哪些格式
        assert "txt" in resp.json()["detail"]

    def test_missing_file_and_text_rejected(self, client) -> None:
        resp = client.post("/api/resumes", data={"name": "只有名字"})
        assert resp.status_code == 400

    def test_oversized_text_rejected(self, client) -> None:
        from app.store import MAX_RESUME_CHARS

        resp = client.post("/api/resumes", data={"text": "x" * (MAX_RESUME_CHARS + 10)})
        assert resp.status_code == 413

    def test_delete_missing_returns_404(self, client) -> None:
        assert client.delete("/api/resumes/aaaaaaaaaaaa").status_code == 404

    def test_get_missing_returns_404(self, client) -> None:
        assert client.get("/api/resumes/aaaaaaaaaaaa").status_code == 404

    def test_invalid_id_is_not_a_path_traversal(self, client) -> None:
        """URL 里的路径穿越尝试不能读到磁盘上的其它文件。"""
        for evil in ("..%2F..%2Fetc%2Fpasswd", "aaaaaaaaaaaa%2F..%2F..", "../../secrets"):
            resp = client.get(f"/api/resumes/{evil}")
            assert resp.status_code in (400, 404)

    def test_list_is_newest_first(self, client) -> None:
        first = client.post("/api/resumes", data={"text": RESUME_TEXT, "name": "第一份"}).json()
        second = client.post("/api/resumes", data={"text": RESUME_TEXT, "name": "第二份"}).json()
        listed = client.get("/api/resumes").json()
        assert [item["resume_id"] for item in listed] == [second["resume_id"], first["resume_id"]]


# ===========================================================================
# 解析
# ===========================================================================


class TestParseEndpoints:
    def test_parse_jd(self, client) -> None:
        data = client.post("/api/parse/jd", json={"text": SIMPLE_JD}).json()
        assert data["engine"] == "rule"       # 测试环境无 Key，走规则引擎
        assert data["jd"]["title"]
        assert data["jd"]["must_have"]
        assert data["llm_error"] == ""

    def test_parse_jd_requires_text(self, client) -> None:
        assert client.post("/api/parse/jd", json={"text": "   "}).status_code == 400
        assert client.post("/api/parse/jd", json={}).status_code == 400

    def test_parse_resume(self, client) -> None:
        data = client.post("/api/parse/resume", json={"text": RESUME_TEXT}).json()
        assert data["resume"]["name"] == "王五"
        assert data["resume"]["skill_names"]
        assert data["engine"] == "rule"

    def test_parse_resume_requires_text(self, client) -> None:
        assert client.post("/api/parse/resume", json={}).status_code == 400


# ===========================================================================
# 匹配
# ===========================================================================


class TestMatchEndpoint:
    def test_match_returns_full_payload(self, client) -> None:
        data = client.post("/api/match", json={"jd_text": SIMPLE_JD, "resume_text": RESUME_TEXT}).json()
        assert data["jd"]["title"]
        assert data["resume"]["name"] == "王五"
        assert data["report"]["dimensions"]
        assert data["elapsed_ms"] >= 0
        assert 0 <= data["report"]["overall"] <= 100

    def test_match_report_has_explanations(self, client) -> None:
        """报告的每一部分都要有内容 —— 空白项会让前端整块渲染不出来。"""
        report = client.post(
            "/api/match", json={"jd_text": SIMPLE_JD, "resume_text": RESUME_TEXT}
        ).json()["report"]
        assert report["verdict"]
        assert report["must_ratio"]
        assert report["engine"] == "rule"
        for dim in report["dimensions"]:
            assert dim["detail"], f"{dim['key']} 缺少说明"
            assert dim["label"]
        for question in report["interview_questions"]:
            assert question["question"] and question["rationale"]

    def test_match_requires_jd(self, client) -> None:
        resp = client.post("/api/match", json={"jd_text": "", "resume_text": RESUME_TEXT})
        assert resp.status_code == 422

    def test_match_requires_resume_source(self, client) -> None:
        resp = client.post("/api/match", json={"jd_text": SIMPLE_JD})
        assert resp.status_code == 400
        assert "resume" in resp.json()["detail"]

    def test_match_by_resume_id(self, client) -> None:
        created = client.post("/api/resumes", data={"text": RESUME_TEXT}).json()
        data = client.post(
            "/api/match", json={"jd_text": SIMPLE_JD, "resume_id": created["resume_id"]}
        ).json()
        assert data["resume"]["name"] == "王五"

    def test_match_by_missing_resume_id_404(self, client) -> None:
        resp = client.post("/api/match", json={"jd_text": SIMPLE_JD, "resume_id": "aaaaaaaaaaaa"})
        assert resp.status_code == 404

    def test_custom_weights_change_score(self, client) -> None:
        payload = {"jd_text": SIMPLE_JD, "resume_text": RESUME_TEXT}
        base = client.post("/api/match", json=payload).json()["report"]["overall"]

        from app.schemas import DIMENSION_ORDER

        only_must = {key: 0.0 for key in DIMENSION_ORDER}
        only_must["must_have"] = 1.0
        tweaked = client.post("/api/match", json={**payload, "weights": only_must}).json()["report"]["overall"]
        assert tweaked != base

    def test_questions_can_be_disabled(self, client) -> None:
        data = client.post(
            "/api/match", json={"jd_text": SIMPLE_JD, "resume_text": RESUME_TEXT, "want_questions": False}
        ).json()
        assert data["report"]["interview_questions"] == []

    def test_question_count_is_clamped_by_schema(self, client) -> None:
        resp = client.post(
            "/api/match",
            json={"jd_text": SIMPLE_JD, "resume_text": RESUME_TEXT, "question_count": 999},
        )
        assert resp.status_code == 422


class TestMatchStream:
    def _collect(self, client, payload: dict) -> list[tuple[str, dict]]:
        with client.stream("POST", "/api/match/stream", json=payload) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            body = "".join(resp.iter_text())

        events: list[tuple[str, dict]] = []
        for block in body.split("\n\n"):
            name = data = None
            for line in block.split("\n"):
                if line.startswith("event: "):
                    name = line[7:].strip()
                elif line.startswith("data: "):
                    data = line[6:]
            if name and data is not None:
                events.append((name, json.loads(data)))
        return events

    def test_event_sequence(self, client) -> None:
        events = self._collect(client, {"jd_text": SIMPLE_JD, "resume_text": RESUME_TEXT})
        names = [name for name, _ in events]

        assert names[0] == "start"
        assert names[-1] == "[DONE]"
        assert "done" in names
        # 进度事件必须按 解析JD -> 解析简历 -> 打分 的顺序推
        stages = [payload["stage"] for name, payload in events if name == "stage"]
        assert stages == ["jd", "resume", "score"]

    def test_progress_is_monotonic(self, client) -> None:
        events = self._collect(client, {"jd_text": SIMPLE_JD, "resume_text": RESUME_TEXT})
        progress = [payload["progress"] for name, payload in events if name == "stage"]
        assert progress == sorted(progress)

    def test_parsed_events_carry_summary(self, client) -> None:
        events = self._collect(client, {"jd_text": SIMPLE_JD, "resume_text": RESUME_TEXT})
        parsed = {payload["target"]: payload for name, payload in events if name == "parsed"}
        assert parsed["jd"]["title"]
        assert parsed["jd"]["must_have"]
        assert parsed["resume"]["name"] == "王五"
        assert parsed["resume"]["skill_count"] > 0

    def test_scored_event_has_six_dimensions(self, client) -> None:
        events = self._collect(client, {"jd_text": SIMPLE_JD, "resume_text": RESUME_TEXT})
        scored = next(payload for name, payload in events if name == "scored")
        assert len(scored["dimensions"]) == 6
        assert scored["verdict"]

    def test_done_event_is_a_full_match_response(self, client) -> None:
        events = self._collect(client, {"jd_text": SIMPLE_JD, "resume_text": RESUME_TEXT})
        done = next(payload for name, payload in events if name == "done")
        assert set(done) >= {"jd", "resume", "report", "elapsed_ms"}
        assert done["report"]["dimensions"]

    def test_request_validation_error_is_a_plain_404(self, client) -> None:
        """**请求参数**错误在建立 SSE 流之前就被拦下，返回普通 404。

        区分这两类错误很重要：
        - 参数错误 -> 普通 HTTP 错误码（前端按 `res.ok` 分支处理即可）；
        - 流水线内部错误 -> SSE `error` 事件（流已经建好，不能断流）。
        """
        resp = client.post("/api/match/stream", json={"jd_text": SIMPLE_JD, "resume_id": "aaaaaaaaaaaa"})
        assert resp.status_code == 404
        assert "不存在" in resp.json()["detail"]

    def test_error_event_on_pipeline_failure(self, client, monkeypatch) -> None:
        """流水线内部崩溃时必须推 `error` 事件并正常收尾，而不是断流。

        断流的话前端只能看到"连接中断"，用户完全不知道发生了什么。
        """
        import app.main as main

        async def boom(*_args, **_kwargs):
            raise RuntimeError("模拟流水线崩溃")

        monkeypatch.setattr(main, "run_pipeline", boom)

        events = self._collect(client, {"jd_text": SIMPLE_JD, "resume_text": RESUME_TEXT})
        names = [name for name, _ in events]
        assert "error" in names
        assert names[-1] == "[DONE]"      # 依然要正常收尾

        error = next(payload for name, payload in events if name == "error")
        assert error["kind"] == "internal"
        assert "崩溃" in error["message"]


class TestBatchMatch:
    def test_sorted_desc_by_overall(self, client, other_resume_text) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        jd = (root / "examples" / "jd_ai_app_engineer.txt").read_text(encoding="utf-8")

        data = client.post(
            "/api/match/batch",
            json={
                "jd_text": jd,
                "resumes": [
                    {"name": "张三", "resume_text": RESUME_TEXT},
                    {"name": "李四", "resume_text": other_resume_text},
                ],
            },
        ).json()

        assert data["total"] == 2
        scores = [item["overall"] for item in data["items"]]
        assert scores == sorted(scores, reverse=True)
        assert data["jd_title"]

    def test_requires_at_least_one_resume(self, client) -> None:
        assert client.post("/api/match/batch", json={"jd_text": SIMPLE_JD, "resumes": []}).status_code == 422

    def test_blank_entries_are_skipped(self, client) -> None:
        """条目里混进空简历（多行文本框常见）时跳过，而不是整批失败。"""
        data = client.post(
            "/api/match/batch",
            json={"jd_text": SIMPLE_JD, "resumes": [{"name": "空", "resume_text": "  "}]},
        )
        assert data.status_code in (400, 422) or data.json()["total"] == 0

    def test_item_fields_for_hr_view(self, client) -> None:
        data = client.post(
            "/api/match/batch",
            json={"jd_text": SIMPLE_JD, "resumes": [{"name": "王五", "resume_text": RESUME_TEXT}]},
        ).json()
        item = data["items"][0]
        assert item["name"] == "王五"
        assert item["must_ratio"]
        assert item["verdict"]
        assert isinstance(item["missing_must"], list)


class TestRecompute:
    def test_recompute_matches_initial_score(self, client) -> None:
        """不传权重时，重算结果必须和首次匹配完全一致。"""
        first = client.post("/api/match", json={"jd_text": SIMPLE_JD, "resume_text": RESUME_TEXT}).json()
        again = client.post(
            "/api/recompute",
            json={"jd": first["jd"], "resume": first["resume"], "want_questions": False},
        ).json()
        assert again["report"]["overall"] == first["report"]["overall"]

    def test_recompute_with_new_weights(self, client) -> None:
        from app.schemas import DIMENSION_ORDER

        first = client.post("/api/match", json={"jd_text": SIMPLE_JD, "resume_text": RESUME_TEXT}).json()
        only_nice = {key: 0.0 for key in DIMENSION_ORDER}
        only_nice["nice_to_have"] = 1.0

        again = client.post(
            "/api/recompute",
            json={"jd": first["jd"], "resume": first["resume"], "weights": only_nice},
        ).json()

        nice_dim = next(d for d in again["report"]["dimensions"] if d["key"] == "nice_to_have")
        assert again["report"]["overall"] == pytest.approx(nice_dim["score"], abs=0.05)

    def test_recompute_preserves_parsed_inputs(self, client) -> None:
        """重算不能改动解析结果 —— 它只换"怎么看"，不换"看到了什么"。"""
        first = client.post("/api/match", json={"jd_text": SIMPLE_JD, "resume_text": RESUME_TEXT}).json()
        again = client.post("/api/recompute", json={"jd": first["jd"], "resume": first["resume"]}).json()
        assert again["jd"] == first["jd"]
        assert again["resume"] == first["resume"]

    def test_recompute_can_enable_questions(self, client) -> None:
        first = client.post(
            "/api/match",
            json={"jd_text": SIMPLE_JD, "resume_text": RESUME_TEXT, "want_questions": False},
        ).json()
        again = client.post(
            "/api/recompute",
            json={"jd": first["jd"], "resume": first["resume"], "want_questions": True, "question_count": 3},
        ).json()
        assert 0 < len(again["report"]["interview_questions"]) <= 3

    def test_recompute_accepts_empty_parsed_objects(self, client) -> None:
        """``ParsedJD`` / ``ParsedResume`` 的字段都有默认值，所以空对象也是合法输入。

        此时所有维度都不适用 —— 接口要能正常返回，并在 risks 里说明
        "分数不代表真实匹配度"，而不是抛 500。
        """
        resp = client.post("/api/recompute", json={"jd": {}, "resume": {}})
        assert resp.status_code == 200
        report = resp.json()["report"]
        assert report["overall"] == 0.0
        assert any("不代表真实匹配度" in risk for risk in report["risks"])

    def test_recompute_rejects_malformed_payload(self, client) -> None:
        """字段类型完全不对时必须是 422（校验层拦下），而不是 500。"""
        assert client.post("/api/recompute", json={"jd": "不是对象", "resume": {}}).status_code == 422
        assert client.post("/api/recompute", json={}).status_code == 422
        assert (
            client.post(
                "/api/recompute", json={"jd": {"must_have": "应该是数组"}, "resume": {}}
            ).status_code
            == 422
        )


class TestLlmPing:
    def test_offline_mode_reports_rule_engine(self, client) -> None:
        data = client.post("/api/llm/ping").json()
        assert data["ok"] is False
        assert data["kind"] == "config"
        assert "规则引擎" in data["message"]
        assert data["latency_ms"] == 0
