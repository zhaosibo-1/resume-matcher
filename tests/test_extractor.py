"""抽取层测试：清洗、校验、降级。

抽取层是整个系统里**最容易静默出错**的一层：模型返回脏数据、缺字段、
类型不对、Key 欠费、限流……任何一种情况都不应该让匹配功能挂掉，
而应该安静地退回规则引擎。所以这里的测试重点全在"坏输入"上。

所有用例都用 ``ScriptedLLM`` 替身，**不联网**。
"""

from __future__ import annotations

import pytest

from app.config import (
    PARSER_ENGINE_AUTO,
    PARSER_ENGINE_LLM,
    PARSER_ENGINE_RULE,
    Settings,
)
from app.extractor import (
    MAX_INPUT_CHARS,
    Extractor,
    sanitize_jd_payload,
    sanitize_resume_payload,
)
from app.llm import ERR_AUTH, ERR_BAD_JSON, ERR_NETWORK, LLMError

from .helpers import ScriptedLLM


def make_settings(**kwargs) -> Settings:
    base = dict(
        llm_preset="custom",
        llm_api_key="sk-test",
        llm_base_url="http://mock/v1",
        llm_model="m",
        engine_mode=PARSER_ENGINE_AUTO,
    )
    base.update(kwargs)
    return Settings(**base)


# ===========================================================================
# JD 抽取结果清洗
# ===========================================================================


class TestSanitizeJdPayload:
    def test_happy_path(self) -> None:
        cleaned = sanitize_jd_payload(
            {
                "title": "AI 应用开发工程师",
                "company": "某某科技",
                "skills": ["Python", "RAG", "FastAPI"],
                "bonus_skills": ["Kubernetes"],
                "soft_skills": ["沟通能力"],
                "education": "本科及以上",
                "min_years": 3,
                "domains": ["智能客服"],
                "seniority": "校招",
            }
        )
        assert cleaned["title"] == "AI 应用开发工程师"
        assert cleaned["skills"] == ["Python", "RAG", "FastAPI"]
        assert cleaned["bonus_skills"] == ["Kubernetes"]
        assert cleaned["min_years"] == 3.0
        assert cleaned["seniority"] == "校招"

    def test_unknown_fields_are_dropped(self) -> None:
        """模型爱加字段。加进来的必须丢掉，否则会悄悄污染下游契约。"""
        cleaned = sanitize_jd_payload({"title": "x", "我的额外分析": "很长的一段话", "salary": "20k"})
        assert set(cleaned) == {
            "title",
            "company",
            "skills",
            "bonus_skills",
            "soft_skills",
            "education",
            "min_years",
            "domains",
            "seniority",
        }

    def test_skills_given_as_string(self) -> None:
        """模型用字符串代替数组是最常见的偏差之一。"""
        cleaned = sanitize_jd_payload({"skills": "Python, Java、Go；Rust"})
        assert cleaned["skills"] == ["Python", "Java", "Go", "Rust"]

    def test_skills_deduplicated_case_insensitively(self) -> None:
        cleaned = sanitize_jd_payload({"skills": ["Python", "python", "PYTHON"]})
        assert cleaned["skills"] == ["Python"]

    def test_skill_in_both_lists_is_treated_as_must(self) -> None:
        """同一技能同时出现在硬性和加分里，按更强的那个（硬性）处理。"""
        cleaned = sanitize_jd_payload({"skills": ["Docker"], "bonus_skills": ["docker", "Kubernetes"]})
        assert cleaned["skills"] == ["Docker"]
        assert cleaned["bonus_skills"] == ["Kubernetes"]

    def test_missing_fields_get_safe_defaults(self) -> None:
        cleaned = sanitize_jd_payload({})
        assert cleaned["title"] == ""
        assert cleaned["skills"] == []
        assert cleaned["bonus_skills"] == []
        assert cleaned["domains"] == []
        assert cleaned["min_years"] is None

    def test_none_input_values(self) -> None:
        cleaned = sanitize_jd_payload({"title": None, "skills": None, "min_years": None})
        assert cleaned["title"] == ""
        assert cleaned["skills"] == []
        assert cleaned["min_years"] is None

    @pytest.mark.parametrize("bad", ["三到五年", "经验丰富", {"years": 3}, [3]])
    def test_non_numeric_min_years_becomes_none(self, bad: object) -> None:
        assert sanitize_jd_payload({"min_years": bad})["min_years"] is None

    def test_numeric_string_min_years_is_parsed(self) -> None:
        assert sanitize_jd_payload({"min_years": "3 年"})["min_years"] == 3.0

    @pytest.mark.parametrize("bad", [-1, 999])
    def test_out_of_range_min_years_becomes_none(self, bad: int) -> None:
        """负数或明显离谱的年限一律视为脏数据 —— 宁可不填，也不能带进打分。"""
        assert sanitize_jd_payload({"min_years": bad})["min_years"] is None

    def test_bool_is_not_a_number(self) -> None:
        """Python 里 True == 1，但模型返回布尔值说明它理解错了，不能当数字用。"""
        assert sanitize_jd_payload({"min_years": True})["min_years"] is None

    def test_overlong_string_is_truncated(self) -> None:
        cleaned = sanitize_jd_payload({"title": "岗" * 500})
        assert len(cleaned["title"]) == 60

    def test_nested_object_as_string_field_is_discarded(self) -> None:
        """模型把 title 返回成对象时，不能把 str(dict) 塞进去给用户看。"""
        assert sanitize_jd_payload({"title": {"name": "工程师"}})["title"] == ""

    def test_newlines_in_string_are_collapsed(self) -> None:
        cleaned = sanitize_jd_payload({"title": "AI\n应用\t工程师"})
        assert "\n" not in cleaned["title"]
        assert "\t" not in cleaned["title"]

    def test_quotes_are_stripped(self) -> None:
        assert sanitize_jd_payload({"title": '"AI 工程师"'})["title"] == "AI 工程师"

    def test_skills_list_is_capped(self) -> None:
        cleaned = sanitize_jd_payload({"skills": [f"技能{i}" for i in range(500)]})
        assert len(cleaned["skills"]) == 80

    def test_overlong_skill_item_is_dropped_not_truncated(self) -> None:
        """单个条目超长说明模型把整句话塞进列表了。

        必须**丢弃**而不是截断：截断会留下半句话，它会以"技能"的身份
        参与词典匹配与 must/nice 分桶，既匹配不上任何东西、又在报告里显眼地碍事。
        """
        sentence = "熟悉并能够独立完成基于 RAG 的问答链路搭建与向量库选型调优工作" * 3
        assert len(sentence) > 60  # 前置条件：确实超长

        cleaned = sanitize_jd_payload({"skills": ["Python", sentence]})
        assert cleaned["skills"] == ["Python"]

    def test_length_boundary_is_about_the_original_item(self) -> None:
        """长度判断必须基于**原始条目**。

        这里曾经是死代码：先按 item_limit 截断再比长度，条件永远不成立，
        于是"丢弃超长条目"实际变成了"截断超长条目"。
        """
        exactly_60 = "技" * 60
        assert len(exactly_60) == 60
        assert sanitize_jd_payload({"skills": [exactly_60]})["skills"] == [exactly_60]

        over_60 = "技" * 61
        assert sanitize_jd_payload({"skills": [over_60]})["skills"] == []

    def test_overlong_item_in_comma_string_is_dropped(self) -> None:
        """字符串形式的列表也要走同一套长度判断。"""
        long_name = "基于大模型的智能客服系统全链路开发与部署运维经验总结报告" * 4
        assert len(long_name) > 60

        cleaned = sanitize_jd_payload({"skills": f"Python, {long_name}, Redis"})
        assert cleaned["skills"] == ["Python", "Redis"]


# ===========================================================================
# 简历抽取结果清洗
# ===========================================================================


class TestSanitizeResumePayload:
    def test_happy_path(self) -> None:
        cleaned = sanitize_resume_payload(
            {
                "name": "张三",
                "education": "本科",
                "school": "沈阳大学",
                "major": "人工智能",
                "graduation_year": 2027,
                "skills": ["Python", "PyTorch"],
                "total_years": 1.5,
                "highlights": ["召回率从 62% 提升到 89%"],
            }
        )
        assert cleaned["name"] == "张三"
        assert cleaned["graduation_year"] == 2027
        assert cleaned["skills"] == ["Python", "PyTorch"]
        assert cleaned["total_years"] == 1.5

    def test_graduation_year_out_of_range(self) -> None:
        assert sanitize_resume_payload({"graduation_year": 1800})["graduation_year"] is None
        assert sanitize_resume_payload({"graduation_year": 9999})["graduation_year"] is None

    def test_graduation_year_string(self) -> None:
        assert sanitize_resume_payload({"graduation_year": "2027 年"})["graduation_year"] == 2027

    def test_total_years_bounds(self) -> None:
        assert sanitize_resume_payload({"total_years": -5})["total_years"] is None
        assert sanitize_resume_payload({"total_years": 100})["total_years"] is None

    def test_no_padding_of_missing_values(self) -> None:
        """模型没给的字段不能瞎补 —— "宁可少写，也不要编造"。"""
        cleaned = sanitize_resume_payload({"name": "张三"})
        assert cleaned["skills"] == []
        # 注意是 None，而不是 0.0：0.0 会被下游当成"零经验"这个结论
        assert cleaned["total_years"] is None
        assert cleaned["graduation_year"] is None

    def test_highlights_capped_and_trimmed(self) -> None:
        cleaned = sanitize_resume_payload({"highlights": [f"亮点{i}" for i in range(40)]})
        assert len(cleaned["highlights"]) == 10


# ===========================================================================
# 抽取器（含降级路径）
# ===========================================================================


class TestExtractorRouting:
    """决定"这次到底调不调模型"。走错分支会让 CI 依赖 secret，或让用户白等。"""

    async def test_rule_mode_never_calls_llm(self) -> None:
        client = ScriptedLLM()
        extractor = Extractor(make_settings(engine_mode=PARSER_ENGINE_RULE), client)
        assert await extractor.extract_jd("需要 Python") is None
        assert await extractor.extract_resume("张三") is None
        assert client.calls == 0

    async def test_llm_mode_without_key_reports_config_error(self) -> None:
        """强制 llm 但没 Key：不能静默降级，必须留下明确原因。"""
        client = ScriptedLLM()
        extractor = Extractor(make_settings(engine_mode=PARSER_ENGINE_LLM, llm_api_key=""), client)
        assert await extractor.extract_jd("需要 Python") is None
        assert extractor.last_error_kind == "config"
        assert "LLM_API_KEY" in extractor.last_error
        assert client.calls == 0

    async def test_auto_mode_without_key_is_silent(self) -> None:
        """auto 模式没 Key 是**预期内**的降级，不该报错吓人。"""
        client = ScriptedLLM()
        extractor = Extractor(make_settings(engine_mode=PARSER_ENGINE_AUTO, llm_api_key=""), client)
        assert await extractor.extract_jd("需要 Python") is None
        assert extractor.last_error == ""
        assert client.calls == 0

    async def test_auto_mode_with_key_calls_llm(self) -> None:
        client = ScriptedLLM(responses=['{"title": "AI 工程师"}'])
        extractor = Extractor(make_settings(), client)
        result = await extractor.extract_jd("招聘 AI 工程师")
        assert result is not None
        assert result["title"] == "AI 工程师"
        assert client.calls == 1


class TestExtractorDegradation:
    """**系统的鲁棒性核心**：模型出任何问题，功能都必须完整可用。"""

    @pytest.mark.parametrize(
        ("error", "expected_kind"),
        [
            (LLMError("鉴权失败", kind=ERR_AUTH), ERR_AUTH),
            (LLMError("连不上", kind=ERR_NETWORK), ERR_NETWORK),
            (LLMError("限流", kind="rate_limit"), "rate_limit"),
        ],
    )
    async def test_llm_error_degrades_to_none(self, error: LLMError, expected_kind: str) -> None:
        client = ScriptedLLM(error=error)
        extractor = Extractor(make_settings(), client)
        assert await extractor.extract_jd("需要 Python") is None
        assert extractor.last_error_kind == expected_kind
        assert extractor.last_error  # 必须留下可展示的原因

    async def test_bad_json_degrades(self) -> None:
        client = ScriptedLLM(responses=["对不起，我做不到。"])
        extractor = Extractor(make_settings(), client)
        assert await extractor.extract_jd("需要 Python") is None
        assert extractor.last_error_kind == ERR_BAD_JSON

    async def test_resume_path_degrades_too(self) -> None:
        client = ScriptedLLM(error=LLMError("超时", kind=ERR_NETWORK))
        extractor = Extractor(make_settings(), client)
        assert await extractor.extract_resume("张三") is None
        assert extractor.last_error_kind == ERR_NETWORK

    async def test_error_state_is_cleared_after_success(self) -> None:
        """失败后又成功时，不能一直挂着旧的错误信息误导用户。"""
        client = ScriptedLLM(responses=['{"title": "AI 工程师"}'])
        extractor = Extractor(make_settings(), client)

        extractor.last_error = "上一次的旧错误"
        extractor.last_error_kind = "network"

        await extractor.extract_jd("招聘 AI 工程师")
        assert extractor.last_error == ""
        assert extractor.last_error_kind == ""

    async def test_long_input_is_truncated_before_sending(self) -> None:
        """超长输入要截断再发 —— 成本与上下文长度都得控。"""
        client = ScriptedLLM(responses=['{"title": "x"}'])
        extractor = Extractor(make_settings(), client)
        await extractor.extract_jd("岗" * (MAX_INPUT_CHARS + 5000))
        sent = client.requests[0][1]["content"]
        assert len(sent) < MAX_INPUT_CHARS + 100

    def test_describe_exposes_state(self) -> None:
        extractor = Extractor(make_settings(engine_mode=PARSER_ENGINE_RULE), ScriptedLLM())
        described = extractor.describe()
        assert described["engine_mode"] == PARSER_ENGINE_RULE
        assert described["active_engine"] == PARSER_ENGINE_RULE
        assert described["llm_enabled"] is True
        assert described["last_error"] == ""


# ===========================================================================
# 抽取 + 解析 的端到端衔接
# ===========================================================================


class TestExtractorToParser:
    """验证模型的输出真的能被解析器吃下去 —— 这是两块"接缝"。"""

    async def test_jd_extraction_feeds_parser(self) -> None:
        from app.parser import parse_jd

        client = ScriptedLLM(
            responses=[
                '{"title": "AI 应用开发工程师", "company": "某某科技",'
                ' "skills": ["Python", "LangChain"], "bonus_skills": ["Kubernetes"],'
                ' "education": "本科及以上", "min_years": 3, "seniority": "校招"}'
            ]
        )
        extractor = Extractor(make_settings(), client)
        llm_data = await extractor.extract_jd("（任意 JD 原文）")
        parsed = parse_jd("任职要求：熟悉 Python。", llm_data=llm_data)

        assert parsed.title == "AI 应用开发工程师"
        assert parsed.company == "某某科技"
        assert "LangChain" in parsed.must_have      # 规则层抽不到，靠模型补
        assert "Python" in parsed.must_have
        assert "Kubernetes" in parsed.nice_to_have
        assert parsed.min_years == 3.0
        assert parsed.seniority == "校招"

    async def test_resume_extraction_feeds_parser(self) -> None:
        from app.parser import parse_resume

        client = ScriptedLLM(
            responses=[
                '{"name": "张三", "education": "本科", "school": "沈阳大学",'
                ' "major": "人工智能", "graduation_year": 2027,'
                ' "skills": ["vLLM"], "total_years": 1.2,'
                ' "highlights": ["召回率从 62% 提升到 89%"]}'
            ]
        )
        extractor = Extractor(make_settings(), client)
        llm_data = await extractor.extract_resume("张三\n教育背景\n沈阳大学 人工智能 本科")
        parsed = parse_resume("张三\n教育背景\n沈阳大学 人工智能 本科", llm_data=llm_data)

        assert parsed.name == "张三"
        assert parsed.school == "沈阳大学"
        assert parsed.major == "人工智能"
        assert parsed.graduation_year == 2027
        assert "vLLM" in parsed.skill_names
        assert any("89%" in item for item in parsed.highlights)

    async def test_model_cannot_forge_offsets_through_extractor(self) -> None:
        """模型给的技能若原文里没有，offset 必须为空，而不是编一个位置。"""
        from app.parser import parse_resume

        client = ScriptedLLM(responses=['{"skills": ["一个原文里根本不存在的技能"]}'])
        extractor = Extractor(make_settings(), client)
        llm_data = await extractor.extract_resume("张三\n专业技能\nPython")
        parsed = parse_resume("张三\n专业技能\nPython", llm_data=llm_data)

        forged = next(i for i in parsed.skills if i.canonical == "一个原文里根本不存在的技能")
        assert forged.start is None
        assert forged.evidence == ""
        assert forged.source == "llm"
