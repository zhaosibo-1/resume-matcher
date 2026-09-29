"""前后端字段契约测试。

**为什么需要这一层？**
前端是一个零构建的静态页面，后端改个字段名，前端**不会报错** ——
它只会渲染出 `undefined`：分数显示成 "undefined 分"、技能矩阵整块空白。
这类问题在浏览器里不容易第一时间发现，但在接口层非常容易拦住。

测试分两部分：
  1. **动态部分**：真的调接口，断言前端确实会读到的字段都在响应里；
  2. **静态部分**：扫描 ``web/index.html`` 里的 ``xxx.field`` 访问，
     逐个核对它是否属于对应的数据结构 —— 后端重命名字段时会立刻失败。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.schemas import (
    BatchMatchItem,
    ConfigResponse,
    ExampleDetail,
    InterviewQuestion,
    MatchReport,
    MatchResponse,
    ParsedJD,
    ParsedResume,
    ResumeListItem,
    ResumeUploadResponse,
    StatsResponse,
)

from .helpers import SIMPLE_JD, SIMPLE_RESUME

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = PROJECT_ROOT / "web" / "index.html"


def fields_of(model: type) -> set[str]:
    return set(model.model_fields)


#: 静态扫描时，每个前端变量对应的数据结构。
#: 用 ``frozenset`` 表示"多选一"（该变量在不同函数里承载过不同结构）。
VARIABLE_SCHEMAS: dict[str, frozenset[type]] = {
    "cfg": frozenset({ConfigResponse}),
    "report": frozenset({MatchReport}),
    "st": frozenset({StatsResponse}),
    "q": frozenset({InterviewQuestion}),
    # 同一份 jd / resume 变量既承载解析结果，也承载 /api/examples/{key} 的返回值
    "jd": frozenset({ParsedJD, ExampleDetail}),
    "resume": frozenset({ParsedResume, ExampleDetail}),
    "item": frozenset({ResumeListItem, BatchMatchItem}),
    "lastResult": frozenset({MatchResponse}),
}

#: ``data`` 在 SSE 回调里承载了多种负载（parsed / scored / done / 词典 / 示例 / 错误体），
#: 所以对它做并集校验：只要属于任何一个契约就不算漂移。
DATA_SCHEMAS: tuple[type, ...] = (
    MatchResponse,
    ParsedJD,
    ParsedResume,
    MatchReport,
    ConfigResponse,
    StatsResponse,
    ExampleDetail,
    ResumeListItem,
    ResumeUploadResponse,
    BatchMatchItem,
    InterviewQuestion,
)

#: SSE 事件的额外字段（不在任何响应模型里，是事件负载特有的）
SSE_ONLY_FIELDS = frozenset(
    {"stage", "progress", "label", "target", "kind", "message", "status", "parsed", "scored", "done"}
)

#: /api/skills 返回的条目不是 pydantic 模型，这里显式列出它的字段。
#: 「显式登记」而不是「不管」，是因为它一旦改名，前端同样会静默渲染成空白。
SKILL_DICT_FIELDS = frozenset({"canonical", "category", "aliases", "parents", "alias_count"})

#: 几个接口返回的是随手拼的 dict（不是模型），同样要登记字段名。
#: 另外 FastAPI 的错误响应体统一是 ``{"detail": ...}``，前端拿它做错误提示。
LOOSE_ENDPOINT_FIELDS = (
    frozenset({"total", "matched", "query", "category", "items", "categories"}),  # /api/skills
    frozenset({"detail"}),                                                        # 错误响应体
)

#: 少数变量在不同函数里承载过"模型 + 字典"两种结构，这里补充非模型字段
EXTRA_ALLOWED: dict[str, frozenset[str]] = {
    # 词条列表：ResumeListItem（简历列表）/ /api/skills 的条目
    "item": SKILL_DICT_FIELDS,
}

#: 非后端字段：这些是 DOM / JS 内置成员。
#: 典型来源：``document.querySelectorAll(".dim").forEach((d) => d.classList...)`` ——
#: 同一个变量名 ``d`` 在别处又代表 API 返回的维度对象，只能靠白名单区分。
DOM_MEMBERS = frozenset(
    {
        "getAttribute",
        "setAttribute",
        "classList",
        "style",
        "value",
        "textContent",
        "innerHTML",
        "querySelector",
        "querySelectorAll",
        "addEventListener",
        "parentElement",
        "insertAdjacentHTML",
        "disabled",
        "click",
        "files",
        "target",
        "key",
        "checked",
        "remove",
    }
)

SCRIPT_RE = re.compile(r"<script>(.*?)</script>", re.DOTALL)


def load_script() -> str:
    html = INDEX_HTML.read_text(encoding="utf-8")
    blocks = SCRIPT_RE.findall(html)
    assert blocks, "web/index.html 里没有找到 <script> 块"
    return "\n".join(blocks)


def accesses(script: str, variable: str) -> set[str]:
    """取出 ``variable.field`` 形式的所有字段名。"""
    pattern = r"(?<![A-Za-z0-9_$])" + re.escape(variable) + r"\.([A-Za-z_$][A-Za-z0-9_$]*)"
    return set(re.findall(pattern, script))


# ===========================================================================
# 动态：接口真的返回了前端要读的字段
# ===========================================================================


class TestLiveResponseContract:
    def test_config_fields_used_by_frontend(self, client) -> None:
        data = client.get("/api/config").json()
        for field in ("engine", "skill_count", "llm_enabled", "model", "provider", "categories"):
            assert field in data, f"/api/config 缺少前端要读的字段：{field}"
        assert isinstance(data["categories"], list) and data["categories"]
        assert isinstance(data["skill_count"], int)
        # 前端 `cfg.default_weights[key]` 逐项读取，必须能按下标取到
        assert isinstance(data["default_weights"], dict) and "must_have" in data["default_weights"]

    def test_stats_fields_used_by_frontend(self, client) -> None:
        data = client.get("/api/stats").json()
        assert isinstance(data["matches_run"], int)

    def test_match_response_fields_used_by_frontend(self, client) -> None:
        data = client.post("/api/match", json={"jd_text": SIMPLE_JD, "resume_text": SIMPLE_RESUME}).json()

        assert set(data) >= {"jd", "resume", "report", "elapsed_ms"}
        report = data["report"]
        for field in (
            "overall",
            "verdict",
            "verdict_level",
            "must_ratio",
            "engine",
            "matched_skills",
            "missing_must",
            "missing_nice",
            "extra_skills",
            "dimensions",
            "weights",
            "interview_questions",
            "strengths",
            "risks",
            "advice",
        ):
            assert field in report, f"report 缺少前端要读的字段：{field}"

        assert set(data["resume"]) >= {"skill_groups", "skill_names"}
        assert set(data["jd"]) >= {"title", "must_have", "nice_to_have"}

    def test_dimension_fields_used_by_frontend(self, client) -> None:
        report = client.post(
            "/api/match", json={"jd_text": SIMPLE_JD, "resume_text": SIMPLE_RESUME}
        ).json()["report"]
        for dim in report["dimensions"]:
            for field in ("key", "label", "score", "weight", "applicable", "detail", "evidence", "gaps"):
                assert field in dim, f"维度 {dim.get('key')} 缺少字段：{field}"
            # 前端把 d.key 当作 DIM_COLORS / report.weights 的下标
            assert dim["key"] in report["weights"]

    def test_interview_question_fields_used_by_frontend(self, client) -> None:
        report = client.post(
            "/api/match", json={"jd_text": SIMPLE_JD, "resume_text": SIMPLE_RESUME}
        ).json()["report"]
        assert report["interview_questions"], "示例数据应当能生成面试题"
        for question in report["interview_questions"]:
            for field in ("kind", "skill", "question", "rationale", "anchor"):
                assert field in question

    def test_example_detail_fields_used_by_frontend(self, client) -> None:
        """前端 `jd.text` / `resume.text` 走的是示例接口，字段名不能漂。"""
        data = client.get("/api/examples/jd_ai_app_engineer").json()
        assert set(data) >= {"kind", "text", "label"}

    def test_resume_list_fields_used_by_frontend(self, client) -> None:
        client.post("/api/resumes", data={"text": SIMPLE_RESUME})
        item = client.get("/api/resumes").json()[0]
        for field in ("name", "char_count", "resume_id"):
            assert field in item

    def test_skill_dict_fields_used_by_frontend(self, client) -> None:
        item = client.get("/api/skills", params={"q": "Python"}).json()["items"][0]
        for field in ("canonical", "category", "aliases", "parents"):
            assert field in item, f"/api/skills 条目缺少字段：{field}"

    def test_sse_payloads_carry_fields_used_by_frontend(self, client) -> None:
        """SSE 事件负载里前端会读 progress / label / target 等，缺一个进度条就不动。"""
        import json as _json

        with client.stream(
            "POST", "/api/match/stream", json={"jd_text": SIMPLE_JD, "resume_text": SIMPLE_RESUME}
        ) as resp:
            body = "".join(resp.iter_text())

        payloads: dict[str, list[dict]] = {}
        for block in body.split("\n\n"):
            name = data = None
            for line in block.split("\n"):
                if line.startswith("event: "):
                    name = line[7:].strip()
                elif line.startswith("data: "):
                    data = line[6:]
            if name and data:
                payloads.setdefault(name, []).append(_json.loads(data))

        for payload in payloads["stage"]:
            assert "progress" in payload and "label" in payload
        for payload in payloads["parsed"]:
            assert "target" in payload and "engine" in payload

        done = payloads["done"][0]
        assert {"report", "jd", "resume", "elapsed_ms"} <= set(done)


# ===========================================================================
# 静态：前端源码里的字段访问都能在后端契约里找到
# ===========================================================================


class TestStaticFieldScan:
    def test_script_block_is_found_and_non_trivial(self) -> None:
        """先确认扫描本身没坏 —— 否则下面的断言会全部"通过"而毫无意义。"""
        script = load_script()
        assert len(script) > 5000
        assert len(accesses(script, "report")) > 10

    @pytest.mark.parametrize("variable", sorted(VARIABLE_SCHEMAS))
    def test_field_access_exists_in_schema(self, variable: str) -> None:
        script = load_script()
        found = accesses(script, variable)

        allowed: set[str] = set()
        for model in VARIABLE_SCHEMAS[variable]:
            allowed |= fields_of(model)
        allowed |= DOM_MEMBERS
        allowed |= EXTRA_ALLOWED.get(variable, frozenset())

        unknown = found - allowed
        assert not unknown, (
            f"前端在 `{variable}.` 上读取了契约里不存在的字段：{sorted(unknown)}；"
            f"可用字段：{sorted(allowed - DOM_MEMBERS)}。"
            f"如果是后端改了字段名，请同步更新 web/index.html。"
        )

    def test_data_union_scan(self) -> None:
        """``data`` 承载多种负载，做并集校验（能拦住凭空造出来的字段名）。"""
        allowed: set[str] = set(SSE_ONLY_FIELDS) | SKILL_DICT_FIELDS | DOM_MEMBERS
        for model in DATA_SCHEMAS:
            allowed |= fields_of(model)
        for loose in LOOSE_ENDPOINT_FIELDS:
            allowed |= loose

        unknown = accesses(load_script(), "data") - allowed
        assert not unknown, f"前端在 `data.` 上读取了契约里不存在的字段：{sorted(unknown)}"

    def test_resume_skill_groups_is_iterated_by_key(self) -> None:
        """前端 `Object.keys(resume.skill_groups)` 然后按下标取数组，
        所以它必须是 dict[str, list[str]] 而不是 list。"""
        assert "skill_groups" in fields_of(ParsedResume)
        assert ParsedResume.model_fields["skill_groups"].annotation == dict[str, list[str]]

    def test_render_targets_exist_in_html(self) -> None:
        """所有 ``$("xxx")`` 的目标元素都必须在 HTML 里真实存在。

        拼错一个 id，页面对应的整块内容就会静默不渲染 —— 这类问题
        在浏览器里只能靠肉眼发现，放到测试里一行就能守住。
        """
        html = INDEX_HTML.read_text(encoding="utf-8")
        script = load_script()

        referenced = set(re.findall(r'\$\("([A-Za-z0-9_\-]+)"\)', script))
        # 动态拼接的 id（如 "cardScore"）也是常量字面量，正则能覆盖；
        # 这里额外补上模板里写死、但由 JS 动态查询的选择器目标
        referenced |= set(re.findall(r'id="([A-Za-z0-9_\-]+)"', script))

        declared = set(re.findall(r'id="([A-Za-z0-9_\-]+)"', html))
        missing = referenced - declared
        assert not missing, f"JS 引用了 HTML 里不存在的元素 id：{sorted(missing)}"

    def test_dimension_colors_cover_all_dimensions(self) -> None:
        """前端 DIM_COLORS 必须覆盖后端定义的六个维度，否则某个维度会画成默认色。"""
        from app.schemas import DIMENSION_ORDER

        script = load_script()
        block = re.search(r"DIM_COLORS\s*=\s*\{(.*?)\}", script, re.DOTALL)
        assert block, "没找到 DIM_COLORS 定义"

        keys = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*:", block.group(1)))
        assert set(DIMENSION_ORDER) <= keys, f"DIM_COLORS 缺少维度：{set(DIMENSION_ORDER) - keys}"

    def test_api_paths_used_by_frontend_all_exist(self, client) -> None:
        """前端 fetch 的每个接口路径都必须真实存在（静态路径 + 一种占位替换）。"""
        script = load_script()
        paths = set(re.findall(r'fetch\(\s*"(/api/[^"?]+)', script))
        paths |= {path.rstrip("/") for path in re.findall(r'"(/api/[^"?]*)"', script)}

        openapi = set(client.get("/openapi.json").json()["paths"])
        # 去掉路径参数占位符后比对
        normalized = {re.sub(r"\{[^}]+\}", "", path).rstrip("/") for path in openapi}

        for path in paths:
            cleaned = re.sub(r"/$", "", path)
            cleaned = re.sub(r"/[^/]*$", "", cleaned) if cleaned.count("/") > 2 else cleaned
            assert (
                cleaned in normalized
                or cleaned + "/{resume_id}" in openapi
                or cleaned + "/{key}" in openapi
                or any(item.startswith(cleaned) for item in normalized)
            ), f"前端请求了不存在的接口：{path}"
